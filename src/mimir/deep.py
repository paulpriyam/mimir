"""Deep search: a loose gate for recall, then a reranker for precision.

Instant search asks one number to do both jobs, and the eval shows it cannot: to keep absurd
queries out, the cosine floor has to sit at 0.68, which also throws away six of forty-six real
paraphrases. Deep search splits the two:

  1. retrieve with the gate loosened to 0.60, so nothing real is missed (0 empty misses in the
     eval, against 6 at 0.68)
  2. rerank those candidates by reading the query and each candidate together
  3. drop everything the reranker scores below the threshold, so "nothing found" still means it

Cost: one model pass over the candidates, so it is opt-in rather than the default.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from .config import DEEP_COS_FLOOR, RERANK_MIN_SCORE, RERANK_TOP
from .search import Hit, Searcher
from .store import Store

DOC_LIMIT = 1200   # characters of a candidate handed to the reranker

# The order a file's own text is worth to a reranker. A caption or the matched snippet says
# what the thing is; a page of filler dilutes it.
TEXT_PRIORITY = {"caption": 0, "ocr": 1, "transcript": 2, "chunk": 3, "page": 4, "name": 5}


def candidate_text(store: Store, hit: Hit) -> str:
    """What the reranker reads for one candidate.

    This has to be the file's real text, not just the segment the vector search landed on.
    Two bugs lived here, both found by measuring: a PDF page's whole text was handed over, so
    1200 characters of filler buried the one relevant sentence and the cross-encoder scored it
    0.0006; and an image's hit points at the image segment, which has no text at all, so the
    reranker was scoring the file name instead of the caption.
    """
    parts: list[str] = []

    # The matched snippet first: it is already centred on the query's words.
    if hit.excerpt and hit.excerpt.strip():
        parts.append(hit.excerpt.strip())

    row = store.file_by_path(hit.path)
    if row is not None:
        texts = store.db.execute(
            """SELECT s.kind AS kind, coalesce(f.original, f.body) AS text
               FROM segments s JOIN fts f ON f.rowid = s.id
               WHERE s.file = ? AND s.id != ? AND f.original IS NOT NULL""",
            (row["id"], hit.segment_id),
        ).fetchall()
        for item in sorted(texts, key=lambda r: TEXT_PRIORITY.get(r["kind"], 9)):
            text = (item["text"] or "").strip()
            if text and text not in parts:
                parts.append(text)

    joined = "\n".join(parts)
    return joined[:DOC_LIMIT] if joined else Path(hit.path).name


def deep_search(store: Store, searcher: Searcher, reranker, text: str,
                query_vector: np.ndarray | None = None, limit: int = 10,
                candidates: int = RERANK_TOP, min_score: float = RERANK_MIN_SCORE,
                kinds: set[str] | None = None) -> list[Hit]:
    """Retrieve loosely, then let the cross-encoder decide order and whether to answer at all."""
    hits = searcher.search(text, query_vector=query_vector, kinds=kinds, limit=candidates,
                           cos_floor=DEEP_COS_FLOOR)
    if not hits:
        return []
    documents = [candidate_text(store, hit) for hit in hits]
    scores = reranker.score(text, documents)
    ranked = sorted(zip(hits, scores), key=lambda pair: -pair[1])
    kept = [replace(hit, score=score) for hit, score in ranked if score >= min_score]
    return kept[:limit]
