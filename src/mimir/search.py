"""Hybrid search: meaning (vectors, on this query's own scale) + words (FTS5).

A file qualifies if its best segment stands out in meaning (z >= z_gate) OR it contains at
least half the query's words. Score = meaning_weight * max(z, 0) + words_weight * coverage^1.5,
plus a small bonus when the query asks for that kind of file ("...in a video").

Exact lookups (codes, IDs, long numbers) flip the weights: meaning half, words double. An
embedding can't hold "INV-20931"; keyword search can.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import numpy as np

from .config import COVERAGE_GATE, WORD_WEIGHT, Z_GATE
from .store import Store
from .vectors import cosine_scores, meaning

_WORD = re.compile(r"[^\w]+", re.UNICODE)

STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "with", "and", "or", "my", "me",
    "that", "this", "these", "those", "from", "about", "where", "what", "which", "who", "when",
    "is", "was", "are", "were", "be", "it", "its", "some", "any", "find", "show",
}

# Kind words describe what to look for, not what is in it: dropped from terms, used as a soft boost.
KIND_WORDS = {
    "photo": "image", "photos": "image", "picture": "image", "pictures": "image", "image": "image",
    "images": "image", "screenshot": "screenshot", "screenshots": "screenshot", "pdf": "pdf",
    "pdfs": "pdf", "document": "doc", "documents": "doc", "note": "doc", "notes": "doc",
    "video": "video", "videos": "video", "clip": "video", "movie": "video", "audio": "audio",
    "song": "audio", "podcast": "audio", "voice": "audio", "sound": "audio", "recording": "audio",
}


def terms(text: str) -> list[str]:
    seen: set[str] = set()
    out = []
    for word in _WORD.split(text.lower()):
        if len(word) >= 2 and word not in STOPWORDS and word not in KIND_WORDS and word not in seen:
            seen.add(word)
            out.append(word)
    return out


def kind_hints(text: str) -> set[str]:
    return {KIND_WORDS[w] for w in _WORD.split(text.lower()) if w in KIND_WORDS}


def is_lookup(words: list[str]) -> bool:
    """Codes and IDs ("K7Q2LM", "TRX50") and long numbers mean nothing to an embedding."""
    return bool(words) and all(
        any(c.isdigit() for c in w) and (any(c.isalpha() for c in w) or len(w) >= 4) for w in words
    )


def patterns(words: list[str], prefix_last: bool = False) -> list[str]:
    return [f'"{w}"' + ("*" if prefix_last and i == len(words) - 1 else "") for i, w in enumerate(words)]


@dataclass
class Hit:
    path: str
    kind: str
    score: float
    z: float | None
    cosine: float | None
    keyword_rank: int | None
    segment_id: int
    loc: float | None
    loc_end: float | None
    excerpt: str | None
    modified: float
    size: int
    info: dict = field(default_factory=dict)


class Searcher:
    def __init__(self, store: Store, load_vectors: bool = True):
        self.store = store
        self.matrix: np.ndarray | None = None
        self.row_segment = np.array([], np.int64)
        self.row_file = np.array([], np.int64)
        self.row_modality = np.array([], object)
        self.file_kinds: dict[int, str] = {}
        self.vectors_loaded = False
        if load_vectors:
            self.reload()

    def reload(self) -> None:
        self.matrix, self.row_segment, self.row_file, self.row_modality = self.store.vector_matrix()
        self.file_kinds = self.store.file_kinds()
        self.vectors_loaded = True

    def release_vectors(self) -> None:
        self.matrix = None
        self.vectors_loaded = False

    def search(self, text: str, query_vector: np.ndarray | None = None, kinds: set[str] | None = None,
               limit: int = 20, z_gate: float = Z_GATE, word_weight: float = WORD_WEIGHT) -> list[Hit]:
        if query_vector is not None and not self.vectors_loaded:
            self.reload()
        words = terms(text)
        allowed = lambda file_id: kinds is None or self.file_kinds.get(file_id) in kinds

        # --- meaning: every vector scored on this query's scale, best segment per file
        best: dict[int, tuple[float, float, int]] = {}
        if query_vector is not None and self.matrix is not None and len(self.row_file):
            scores = cosine_scores(self.matrix.astype(np.float32), query_vector)
            z = meaning(scores, self.row_modality)
            for row in range(len(scores)):
                file_id = int(self.row_file[row])
                if not allowed(file_id):
                    continue
                if z[row] > best.get(file_id, (float("-inf"), 0.0, -1))[0]:
                    best[file_id] = (float(z[row]), float(scores[row]), row)

        # --- words: FTS5, one entry per file (first/best match)
        keyword: dict[int, tuple[int, str | None]] = {}
        coverage: dict[int, float] = {}
        pats = patterns(words)
        if pats:
            query = " OR ".join(pats)
            rows = self.store.db.execute(
                """SELECT s.file AS file, s.id AS sid, snippet(fts, 0, '«', '»', '…', 12) AS snip
                   FROM fts JOIN segments s ON s.id = fts.rowid
                   WHERE fts MATCH ? ORDER BY bm25(fts) LIMIT 1000""",
                (query,),
            ).fetchall()
            for row in rows:
                file_id = row["file"]
                if not allowed(file_id) or file_id in keyword:
                    continue
                keyword[file_id] = (row["sid"], row["snip"])
                if len(keyword) >= 100:
                    break
            for i, pat in enumerate(pats):
                hits = self.store.db.execute(
                    "SELECT DISTINCT s.file AS file FROM fts JOIN segments s ON s.id = fts.rowid WHERE fts MATCH ?",
                    (pat,),
                ).fetchall()
                for row in hits:
                    if row["file"] in keyword:
                        coverage[row["file"]] = coverage.get(row["file"], 0.0) + 1.0 / len(pats)

        # --- fuse
        lookup = is_lookup(words)
        meaning_weight, words_weight = (0.5, word_weight * 2) if lookup else (1.0, word_weight)
        hinted = kind_hints(text)
        candidates = {f for f, (z, _, _) in best.items() if z >= z_gate}
        candidates |= {f for f, share in coverage.items() if share >= COVERAGE_GATE}

        scored: dict[int, float] = {}
        for file_id in candidates:
            score = meaning_weight * max(best.get(file_id, (0.0, 0.0, -1))[0], 0.0)
            score += words_weight * (coverage.get(file_id, 0.0) ** 1.5)
            if self.file_kinds.get(file_id) in hinted:
                score += 1.0
            scored[file_id] = score

        hits: list[Hit] = []
        for file_id, score in sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]:
            row = self.store.db.execute(
                "SELECT path, kind, mtime, size, info FROM files WHERE id = ?", (file_id,)
            ).fetchone()
            semantic = best.get(file_id)
            kw = keyword.get(file_id)
            use_meaning = semantic is not None and (semantic[0] >= z_gate or kw is None)
            segment_id = int(self.row_segment[semantic[2]]) if use_meaning else (kw[0] if kw else None)
            if segment_id is None:
                continue
            seg = self.store.segment(segment_id)
            excerpt = None
            if kw and kw[0] == segment_id and kw[1]:
                excerpt = kw[1]
            if excerpt is None and seg:
                excerpt = seg.excerpt
            hits.append(Hit(
                path=row["path"], kind=row["kind"], score=score,
                z=semantic[0] if semantic else None, cosine=semantic[1] if semantic else None,
                keyword_rank=None, segment_id=segment_id,
                loc=seg.loc if seg else None, loc_end=seg.loc_end if seg else None,
                excerpt=excerpt, modified=row["mtime"], size=row["size"],
                info=json.loads(row["info"] or "{}"),
            ))
        return hits

    @staticmethod
    def strong_count(hits: list[Hit], ratio: float = 0.5) -> int:
        """How many hits (best first) score at least `ratio` of the top hit."""
        if not hits or hits[0].score <= 0:
            return len(hits)
        top = hits[0].score
        return sum(1 for h in hits if h.score >= top * ratio)
