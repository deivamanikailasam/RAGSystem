"""Unit tests for the IR metric functions, with hand-computed expectations.

Reference ranking used across tests:
    retrieved = [B, A, C, D],  relevant = {A, C}
    - A is at rank 2, C at rank 3, B and D are irrelevant.
"""

from __future__ import annotations

import math

import pytest

from eval.metrics import (
    average_precision,
    evaluate_queries,
    f1_at_k,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

RETRIEVED = ["B", "A", "C", "D"]
RELEVANT = {"A", "C"}


def test_precision_at_k():
    assert precision_at_k(RETRIEVED, RELEVANT, 1) == 0.0          # B
    assert precision_at_k(RETRIEVED, RELEVANT, 2) == pytest.approx(0.5)   # B,A -> 1/2
    assert precision_at_k(RETRIEVED, RELEVANT, 3) == pytest.approx(2 / 3) # B,A,C -> 2/3
    assert precision_at_k(RETRIEVED, RELEVANT, 4) == pytest.approx(0.5)   # 2/4


def test_recall_at_k():
    assert recall_at_k(RETRIEVED, RELEVANT, 1) == 0.0
    assert recall_at_k(RETRIEVED, RELEVANT, 2) == pytest.approx(0.5)  # found A of {A,C}
    assert recall_at_k(RETRIEVED, RELEVANT, 3) == pytest.approx(1.0)  # found A and C


def test_f1_at_k():
    # @3: p=2/3, r=1.0 -> F1 = 2*(2/3)/(2/3+1) = (4/3)/(5/3) = 0.8
    assert f1_at_k(RETRIEVED, RELEVANT, 3) == pytest.approx(0.8)
    assert f1_at_k(RETRIEVED, RELEVANT, 1) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(RETRIEVED, RELEVANT) == pytest.approx(0.5)  # first hit rank 2
    assert reciprocal_rank(["X", "Y"], RELEVANT) == 0.0


def test_average_precision():
    # hits at rank 2 (prec 1/2) and rank 3 (prec 2/3); /|R|=2
    expected = (0.5 + (2 / 3)) / 2
    assert average_precision(RETRIEVED, RELEVANT) == pytest.approx(expected)


def test_ndcg_at_k():
    # DCG@3 = 0 + 1/log2(3) + 1/log2(4); IDCG = 1/log2(2) + 1/log2(3)
    dcg = 1 / math.log2(3) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(RETRIEVED, RELEVANT, 3) == pytest.approx(dcg / idcg)


def test_hit_at_k():
    assert hit_at_k(RETRIEVED, RELEVANT, 1) == 0.0
    assert hit_at_k(RETRIEVED, RELEVANT, 2) == 1.0


def test_perfect_ranking_scores_one():
    retrieved = ["A", "C", "B", "D"]
    assert precision_at_k(retrieved, RELEVANT, 2) == 1.0
    assert recall_at_k(retrieved, RELEVANT, 2) == 1.0
    assert average_precision(retrieved, RELEVANT, 5) == pytest.approx(1.0)
    assert ndcg_at_k(retrieved, RELEVANT, 5) == pytest.approx(1.0)
    assert reciprocal_rank(retrieved, RELEVANT) == 1.0


def test_empty_relevant_is_zero_not_error():
    assert recall_at_k(RETRIEVED, set(), 3) == 0.0
    assert average_precision(RETRIEVED, set(), 3) == 0.0
    assert ndcg_at_k(RETRIEVED, set(), 3) == 0.0


def test_aggregation_means_over_queries():
    # AP/MAP truncate at max(ks); use ks=(2,) so q2's rank-2 hit still counts.
    q1 = (["A", "B"], {"A"})       # P@2=0.5, RR=1,   AP@2=1.0
    q2 = (["B", "A"], {"A"})       # P@2=0.5, RR=0.5, AP@2=0.5
    agg = evaluate_queries([q1, q2], ks=(2,))
    assert agg["precision@2"] == pytest.approx(0.5)   # mean(0.5, 0.5)
    assert agg["mrr"] == pytest.approx(0.75)          # mean(1, 0.5)
    assert agg["map"] == pytest.approx(0.75)          # mean(1.0, 0.5)


def test_aggregation_empty_results():
    assert evaluate_queries([]) == {}
