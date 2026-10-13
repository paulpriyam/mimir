"""Calibrate the reranker's score threshold in a single pass.

Reranker scores vary enormously by query (0.9998 for one query, 0.00006 for another), so a
fixed threshold is the same mistake as a fixed cosine floor. This runs the deep pipeline once,
caches every score, then reports what each threshold would have done - recall against the
negative false-positive count - so the choice is made from data.

    python scripts/sweep_rerank.py [eval_file] [index_dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

from mimir.config import DEEP_COS_FLOOR, QUERY_PROMPT
from mimir.deep import candidate_text
from mimir.embed import WorkerEmbedder
from mimir.evals import load_eval
from mimir.rerank import Reranker
from mimir.search import Searcher
from mimir.store import Store

EVAL_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evals/media.json")
INDEX = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.home() / ".mimir" / "index.sqlite"
THRESHOLDS = [0.0, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005]


def main() -> int:
    store = Store(INDEX)
    base, queries = load_eval(EVAL_FILE)
    searcher = Searcher(store)
    embedder = WorkerEmbedder(text_only=True)
    reranker = Reranker()
    print(f"eval: {EVAL_FILE}   candidates: cos_floor {DEEP_COS_FLOOR}")

    # One pass: cache the reranked candidates per query.
    cached = []
    try:
        for query in queries:
            vector = embedder.embed_text([query.q], QUERY_PROMPT)[0]
            kinds = {query.kind} if query.kind else None
            hits = searcher.search(query.q, query_vector=vector, kinds=kinds, limit=25,
                                   cos_floor=DEEP_COS_FLOOR)
            docs = [candidate_text(store, hit) for hit in hits]
            scores = reranker.score(query.q, docs) if hits else []
            cached.append((query, hits, scores))
    finally:
        reranker.close()
        embedder.close()

    def matches(hit, target: str) -> bool:
        parts = target.split("@", 1)
        if Path(hit.path).resolve() != (base / parts[0]).resolve():
            return False
        if len(parts) == 1:
            return True
        lo, hi = (float(x) for x in parts[1].split("-", 1))
        if hit.loc is None:
            return False
        end = hit.loc_end if hit.loc_end is not None else hit.loc
        return hit.loc <= hi and end >= lo

    print(f"\n{'threshold':>10}  {'ALL R@1':>7} {'ALL R@5':>7}  {'para R@1':>8} {'para R@5':>8}  "
          f"{'empty':>5} {'later':>5}  {'neg FP':>7}")
    for threshold in THRESHOLDS:
        rows = {"ALL": [0, 0, 0], "paraphrase": [0, 0, 0]}
        empty = later = 0
        fp = neg = 0
        for query, hits, scores in cached:
            kept = sorted(zip(hits, scores), key=lambda pair: -pair[1])
            kept = [hit for hit, score in kept if score >= threshold]
            negative = not query.expect
            if negative:
                neg += 1
                if kept:
                    fp += 1
                continue
            rank = next((i + 1 for i, hit in enumerate(kept)
                         if any(matches(hit, t) for t in query.expect)), None)
            if rank is None:
                empty += 1
            elif rank > 1:
                later += 1
            for key in ("ALL", query.tag):
                if key not in rows:
                    rows[key] = [0, 0, 0]
                rows[key][0] += 1
                if rank == 1:
                    rows[key][1] += 1
                if rank is not None and rank <= 5:
                    rows[key][2] += 1
        a, p = rows["ALL"], rows["paraphrase"]
        an, pn = max(a[0], 1), max(p[0], 1)
        print(f"{threshold:>10.5f}  {a[1] / an:>7.2f} {a[2] / an:>7.2f}  {p[1] / pn:>8.2f} "
              f"{p[2] / pn:>8.2f}  {empty:>5} {later:>5}  {fp:>3}/{neg:<3}")
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
