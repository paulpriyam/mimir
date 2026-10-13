"""Sweep the retrieval gate against the eval set, and show the trade honestly.

Two constants decide what counts as a match: Z_GATE (how far the best segment must stand out
on the query's own scale) and COS_FLOOR (how close it must be in absolute terms). They were
chosen when the paraphrase set was 15 queries and there were 2 negatives, which is too little
to fit a threshold against without overfitting. There are now 46 and 8.

    python scripts/sweep_gate.py [eval_file] [index_dir]

Prints paraphrase R@1/R@5 next to the negative false-positive count, so a setting that buys
recall by inventing answers is visible rather than hidden.
"""

from __future__ import annotations

import sys
from pathlib import Path

from mimir.embed import WorkerEmbedder
from mimir.evals import evaluate
from mimir.store import Store

EVAL_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evals/media.json")
INDEX = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.home() / ".mimir" / "index.sqlite"

Z_GATES = [1.5, 2.0, 2.5, 3.0]
COS_FLOORS = [0.60, 0.64, 0.66, 0.68, 0.70, 0.72]


def main() -> int:
    store = Store(INDEX)
    embedder = WorkerEmbedder(text_only=True)
    print(f"eval: {EVAL_FILE}   index: {INDEX}")
    print(f"{'cos_floor':>10} {'z_gate':>7}  {'para R@1':>8} {'para R@5':>8}  "
          f"{'empty':>5} {'later':>5}  {'neg FP':>7}")
    try:
        for cos_floor in COS_FLOORS:
            for z_gate in Z_GATES:
                report = evaluate(store, embedder, EVAL_FILE, z_gate=z_gate, cos_floor=cos_floor)
                para = report.rows["paraphrase"]
                neg = report.rows["negative"]
                n = float(max(para.positives, 1))
                print(f"{cos_floor:>10.2f} {z_gate:>7.1f}  {para.at1 / n:>8.2f} {para.at5 / n:>8.2f}  "
                      f"{para.empty_misses:>5} {para.found_later:>5}  "
                      f"{neg.false_positives:>3}/{neg.negatives:<3}")
            print()
    finally:
        embedder.close()
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
