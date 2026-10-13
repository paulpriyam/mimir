"""The eval harness. This is the thing that decides whether a model or a knob stays.

Format (same shape as DigUp's evals/*.json, plus negatives):

    {
      "base": "~/mimir-testbed",
      "queries": [
        {"q": "the clause about pets in the lease", "expect": ["lease.pdf@3-3"], "tag": "pdf-page"},
        {"q": "INV-20931", "expect": ["invoices/march.txt"], "tag": "exact"},
        {"q": "quantum blockchain yoga", "expect": [], "tag": "negative"}
      ]
    }

`expect` is a list of `path` or `path@start-end` (a PDF page, or seconds into audio/video).
An empty `expect` is a negative: the query must return nothing above the gate. DigUp's evals
have none of these, which is exactly why its "nothing found" behaviour is untested.

Reported per tag and overall: R@1, R@5, R@10, MRR, false-positive rate on negatives, and
p50/p95 latency. Peak RSS and index size are printed too, so cost is never a surprise.
"""

from __future__ import annotations

import json
import os
import resource
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import COS_FLOOR, WORD_WEIGHT, Z_GATE
from .search import Searcher


@dataclass
class EvalQuery:
    q: str
    expect: list[str]
    tag: str = "all"
    kind: str | None = None


@dataclass
class TagRow:
    n: int = 0
    at1: int = 0
    at5: int = 0
    at10: int = 0
    mrr: float = 0.0
    negatives: int = 0
    false_positives: int = 0

    @property
    def positives(self) -> int:
        return self.n - self.negatives

    def add(self, rank: int | None, negative: bool) -> None:
        self.n += 1
        if negative:
            self.negatives += 1
            if rank is not None:
                self.false_positives += 1
            return
        if rank is None:
            return
        if rank <= 1:
            self.at1 += 1
        if rank <= 5:
            self.at5 += 1
        if rank <= 10:
            self.at10 += 1
        self.mrr += 1.0 / rank


@dataclass
class Report:
    rows: dict[str, TagRow] = field(default_factory=dict)
    latencies: list[float] = field(default_factory=list)
    per_query: list[tuple[str, str, int | None, str]] = field(default_factory=list)

    def p(self, q: float) -> float:
        if not self.latencies:
            return 0.0
        ordered = sorted(self.latencies)
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def load_eval(path: str | Path) -> tuple[Path, list[EvalQuery]]:
    path = Path(path)
    data = json.loads(path.read_text())
    base = data.get("base", ".")
    # MIMIR_TESTBED stands in for ~/mimir-testbed, so an eval file stays portable.
    testbed = os.environ.get("MIMIR_TESTBED")
    if testbed and base.startswith("~/mimir-testbed"):
        base = testbed + base[len("~/mimir-testbed"):]
    base_path = Path(base).expanduser()
    if not base_path.is_absolute():
        base_path = (path.parent / base_path).resolve()
    else:
        base_path = base_path.resolve()   # /var -> /private/var, so paths compare equal
    queries = [
        EvalQuery(q=item["q"], expect=item.get("expect", []), tag=item.get("tag", "all"), kind=item.get("kind"))
        for item in data["queries"]
    ]
    return base_path, queries


def _matches(hit, base: Path, target: str) -> bool:
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


def evaluate(store, embedder, eval_path: str | Path, depth: int = 10,
             z_gate: float = Z_GATE, word_weight: float = WORD_WEIGHT,
             cos_floor: float = COS_FLOOR) -> Report:
    base, queries = load_eval(eval_path)
    searcher = Searcher(store)
    report = Report()

    for query in queries:
        started = time.perf_counter()
        vector = embedder.embed_text([query.q])[0] if query.expect is not None else None
        kinds = {query.kind} if query.kind else None
        hits = searcher.search(query.q, query_vector=vector, kinds=kinds, limit=depth,
                               z_gate=z_gate, word_weight=word_weight, cos_floor=cos_floor)
        report.latencies.append(time.perf_counter() - started)

        negative = not query.expect
        if negative:
            rank = None if not hits else 1  # any hit at all is a false positive
        else:
            rank = next((i + 1 for i, hit in enumerate(hits)
                         if any(_matches(hit, base, target) for target in query.expect)), None)

        top = Path(hits[0].path).name if hits else "-"
        # z and cosine of the top hit: what the gate saw, so a false positive is diagnosable.
        detail = ""
        if hits and hits[0].z is not None:
            detail = f" (z {hits[0].z:.1f}, cos {hits[0].cosine:.3f})"
        report.per_query.append((query.tag, query.q, rank, top + detail))
        for key in {query.tag, "ALL"}:
            report.rows.setdefault(key, TagRow()).add(rank, negative)

    return report


def peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / (1024 * 1024) if sys.platform == "darwin" else usage / 1024


def format_report(report: Report, depth: int = 10) -> str:
    lines = []
    for tag, q, rank, top in report.per_query:
        where = f"#{rank}" if rank else "miss"
        lines.append(f"{where:>4}  {tag:<12} \"{q}\"  top: {top}")
    lines.append("")
    lines.append(f"{'tag':<12} {'n':>3}  {'R@1':>5} {'R@5':>5} {f'R@{depth}':>5} {'MRR':>5}  {'neg':>4} {'FP':>4}")
    for tag in sorted(report.rows, key=lambda k: (k != "ALL", k)):
        row = report.rows[tag]
        fpr = f"{row.false_positives}/{row.negatives}" if row.negatives else "-"
        if row.positives:
            n = float(row.positives)
            rates = f"{row.at1 / n:>5.2f} {row.at5 / n:>5.2f} {row.at10 / n:>5.2f} {row.mrr / n:>5.2f}"
        else:
            rates = f"{'-':>5} {'-':>5} {'-':>5} {'-':>5}"
        lines.append(f"{tag:<12} {row.n:>3}  {rates}  {row.negatives:>4} {fpr:>4}")
    if report.latencies:
        lines.append(f"\nlatency p50 {report.p(0.5) * 1000:.0f} ms   p95 {report.p(0.95) * 1000:.0f} ms"
                     f"   peak RSS {peak_rss_mb():.0f} MB")
    return "\n".join(lines)
