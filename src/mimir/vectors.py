"""Vector math: cosine, the per-query meaning scale, and rank fusion.

The meaning scale is the part worth understanding. Cosine similarities are not comparable
across queries or across modalities (a text query's cosines against images sit at a different
level than against text). So every score is put on the query's own scale: subtract the
modality's average, divide by the standard deviation of all of them. Within a modality the
model's own order is untouched; across modalities they become comparable.

This is what DigUp measured beating reciprocal-rank fusion (Searcher.swift:177): R@1 0.88 vs
0.73 synthetic, 0.86 vs 0.82 real. RRF is kept here only for merging lists that genuinely
live on different scales (the reranker's logits).
"""

from __future__ import annotations

import numpy as np

MIN_MODALITY_GROUP = 20  # smaller groups borrow the global average (DigUp's rule)


def normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize each row. Zero rows are left alone."""
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def cosine_scores(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Cosine of `query` against every row of `matrix`. Rows are assumed L2-normalized."""
    query = np.asarray(query, dtype=np.float32)
    norm = float(np.linalg.norm(query))
    if norm == 0:
        return np.zeros(len(matrix), dtype=np.float32)
    return matrix @ (query / norm)


def modality_means(scores: np.ndarray, modality: np.ndarray, sets_scale: np.ndarray | None = None):
    """The average score of each modality for this query. Small groups take the global average."""
    mask = np.ones(len(scores), dtype=bool) if sets_scale is None else np.asarray(sets_scale, bool)
    scores, modality = scores[mask], modality[mask]
    if len(scores) == 0:
        return {}
    overall = float(scores.mean())
    out: dict[str, float] = {}
    for name in set(modality.tolist()):
        group = scores[modality == name]
        out[name] = float(group.mean()) if len(group) >= MIN_MODALITY_GROUP else overall
    return out


def meaning(scores: np.ndarray, modality: np.ndarray, sets_scale: np.ndarray | None = None) -> np.ndarray:
    """Each vector's score on this query's scale: (cosine - modality mean) / global deviation."""
    scores = np.asarray(scores, dtype=np.float64)
    means = modality_means(scores, modality, sets_scale)
    centered = scores - np.array([means.get(m, 0.0) for m in modality], dtype=np.float64)
    mask = np.ones(len(scores), dtype=bool) if sets_scale is None else np.asarray(sets_scale, bool)
    if not mask.any():
        return np.zeros(len(scores), dtype=np.float64)
    deviation = max(float(centered[mask].std()), 0.02)
    return (centered - float(centered[mask].mean())) / deviation


def rrf(rankings: list[list], k: int = 60) -> dict:
    """Reciprocal rank fusion: merge ranked lists of ids that live on different scales."""
    fused: dict = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + 1.0 / (k + rank)
    return fused
