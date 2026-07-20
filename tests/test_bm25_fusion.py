"""Unit tests for the BM25 index and the fusion functions."""

from __future__ import annotations

from app.core.bm25 import BM25Index, tokenize
from app.core.fusion import reciprocal_rank_fusion, weighted_fusion


def _index() -> BM25Index:
    docs = {
        10: "faiss vector index nearest neighbor search over embeddings",
        11: "refund policy customer thirty days order number support",
        12: "office coffee machine kitchen rota cleaning schedule",
    }
    return BM25Index(
        vector_ids=list(docs.keys()),
        docs_tokens=[tokenize(t) for t in docs.values()],
    )


def test_bm25_ranks_exact_term_match_first():
    idx = _index()
    hits = idx.search("how do I get a refund order number", limit=3)
    assert hits
    assert hits[0][0] == 11  # the refund doc
    assert hits[0][1] > 0


def test_bm25_empty_query_or_index():
    assert _index().search("zzzz nonexistent", limit=5) == []
    empty = BM25Index(vector_ids=[], docs_tokens=[])
    assert empty.search("anything", limit=5) == []


def test_bm25_limit():
    idx = _index()
    # A query touching all docs; limit caps the return count.
    hits = idx.search("search refund coffee", limit=2)
    assert len(hits) <= 2


def test_rrf_rewards_agreement():
    # Doc 1 is top of the dense list and second in sparse -> should win overall.
    dense = [1, 2, 3]
    sparse = [4, 1, 2]
    fused = reciprocal_rank_fusion([dense, sparse], k=60)
    ranked = sorted(fused, key=lambda v: fused[v], reverse=True)
    assert ranked[0] == 1


def test_rrf_includes_items_from_either_list():
    fused = reciprocal_rank_fusion([[1, 2], [3, 4]], k=60)
    assert set(fused.keys()) == {1, 2, 3, 4}


def test_weighted_fusion_alpha_favors_dense():
    dense = {1: 0.9, 2: 0.1}
    sparse = {2: 0.9, 1: 0.1}
    # alpha=1 -> pure dense: doc 1 wins.
    f_dense = weighted_fusion(dense, sparse, alpha=1.0)
    assert max(f_dense, key=f_dense.get) == 1
    # alpha=0 -> pure sparse: doc 2 wins.
    f_sparse = weighted_fusion(dense, sparse, alpha=0.0)
    assert max(f_sparse, key=f_sparse.get) == 2
