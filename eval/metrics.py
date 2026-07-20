"""Information-retrieval metrics for evaluating retrieval quality.

All functions take a ranked list of retrieved item ids (best first, already
de-duplicated to the unit of judgment — here, document ids) and the *set* of
relevant ids for that query (the relevance judgments / "qrels"). Relevance is
treated as **binary** (relevant or not).

Metric cheat-sheet
------------------
* **Precision@k** — of the top-k results, what fraction are relevant?
  "How much of what I showed is good?"  → penalizes noise.
* **Recall@k** — of all relevant docs, what fraction are in the top-k?
  "How much of the good stuff did I find?"  → penalizes misses.
* **F1@k** — harmonic mean of precision@k and recall@k.
* **MRR** — mean of 1/rank of the *first* relevant result. Rewards putting a
  correct answer near the top.
* **Average Precision (AP)** → **MAP** when meaned over queries — precision
  averaged at every rank where a relevant doc appears; rewards ranking *all*
  relevant docs highly, not just the first.
* **nDCG@k** — discounted cumulative gain normalized by the ideal ordering;
  rewards relevant docs appearing earlier, on a 0–1 scale.
* **Hit@k / Success@k** — 1 if any relevant doc is in the top-k, else 0.

These are pure functions with no dependencies so they are trivially unit-tested
(see tests/test_eval_metrics.py).
"""

from __future__ import annotations

import math
from statistics import mean


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    topk = retrieved[:k]
    hits = sum(1 for d in topk if d in relevant)
    return hits / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    topk = retrieved[:k]
    hits = sum(1 for d in topk if d in relevant)
    return hits / len(relevant)


def f1_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for i, d in enumerate(retrieved, start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def average_precision(retrieved: list[str], relevant: set[str], k: int | None = None) -> float:
    """AP truncated at k; divides by min(|relevant|, k) so a perfect ranking scores 1.0."""
    if not relevant:
        return 0.0
    ranked = retrieved[: k] if k else retrieved
    hits = 0
    score = 0.0
    for i, d in enumerate(ranked, start=1):
        if d in relevant:
            hits += 1
            score += hits / i
    denom = min(len(relevant), k) if k else len(relevant)
    return score / denom if denom else 0.0


def dcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    dcg = 0.0
    for i, d in enumerate(retrieved[:k], start=1):
        if d in relevant:
            dcg += 1.0 / math.log2(i + 1)
    return dcg


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    dcg = dcg_at_k(retrieved, relevant, k)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def hit_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if any(d in relevant for d in retrieved[:k]) else 0.0


# --------------------------------------------------------------------------- #
# Aggregation over a set of queries
# --------------------------------------------------------------------------- #
def evaluate_queries(
    results: list[tuple[list[str], set[str]]],
    ks: tuple[int, ...] = (1, 3, 5),
) -> dict[str, float]:
    """Aggregate (mean) metrics over many queries.

    Args:
        results: one ``(retrieved_doc_ids, relevant_doc_ids)`` pair per query.
        ks: cutoffs at which to report precision/recall/F1/nDCG/hit.

    Returns:
        A flat dict, e.g. ``{"precision@3": .., "recall@3": .., "map": ..}``.
    """
    if not results:
        return {}

    agg: dict[str, float] = {}
    for k in ks:
        agg[f"precision@{k}"] = mean(precision_at_k(r, rel, k) for r, rel in results)
        agg[f"recall@{k}"] = mean(recall_at_k(r, rel, k) for r, rel in results)
        agg[f"f1@{k}"] = mean(f1_at_k(r, rel, k) for r, rel in results)
        agg[f"ndcg@{k}"] = mean(ndcg_at_k(r, rel, k) for r, rel in results)
        agg[f"hit@{k}"] = mean(hit_at_k(r, rel, k) for r, rel in results)

    max_k = max(ks)
    agg["mrr"] = mean(reciprocal_rank(r, rel) for r, rel in results)
    agg["map"] = mean(average_precision(r, rel, max_k) for r, rel in results)
    return {key: round(value, 4) for key, value in agg.items()}
