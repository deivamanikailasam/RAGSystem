"""Fusion of dense (vector) and sparse (BM25) result lists.

Two standard strategies:

* **Reciprocal Rank Fusion (RRF)** — combine by *rank*, not raw score:
  ``score(d) = Σ 1 / (k + rank_i(d))`` over the lists that contain ``d``. It is
  scale-free (dense cosine and BM25 scores live on totally different scales, so
  you can't just add them), robust, and the recommended default. ``k`` (≈60)
  damps the influence of very top ranks.

* **Weighted (convex) combination** — min-max normalize each list's scores to
  ``[0, 1]`` and blend: ``alpha * dense + (1 - alpha) * sparse``. Gives a knob
  (``alpha``) to favor semantic vs lexical matching, at the cost of being
  sensitive to score distributions.

Both take/return ``{vector_id: fused_score}`` so the retriever can rank a merged
candidate set uniformly.
"""

from __future__ import annotations

from collections import defaultdict


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]], k: int = 60
) -> dict[int, float]:
    """RRF over several ranked lists of vector ids (each best-first)."""
    fused: dict[int, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, vector_id in enumerate(ranked):
            fused[vector_id] += 1.0 / (k + rank + 1)
    return dict(fused)


def _minmax(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    values = scores.values()
    lo, hi = min(values), max(values)
    if hi == lo:
        # All equal (or a single item) → treat as fully relevant.
        return {vid: 1.0 for vid in scores}
    span = hi - lo
    return {vid: (s - lo) / span for vid, s in scores.items()}


def weighted_fusion(
    dense: dict[int, float], sparse: dict[int, float], alpha: float = 0.5
) -> dict[int, float]:
    """Convex blend of min-max-normalized dense and sparse scores.

    ``alpha`` is the weight on the dense (vector) side; ``1 - alpha`` on sparse.
    """
    dense_n = _minmax(dense)
    sparse_n = _minmax(sparse)
    fused: dict[int, float] = defaultdict(float)
    for vid, s in dense_n.items():
        fused[vid] += alpha * s
    for vid, s in sparse_n.items():
        fused[vid] += (1 - alpha) * s
    return dict(fused)
