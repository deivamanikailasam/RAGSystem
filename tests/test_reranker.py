"""Reranking stage tests."""

from __future__ import annotations

from app.config import Settings
from app.core.docstore import ChunkRecord
from app.core.reranker import (
    LexicalReranker,
    NoOpReranker,
    build_reranker,
)
from app.core.retriever import RetrievedChunk


def _chunk(vector_id: int, text: str, vec_score: float) -> RetrievedChunk:
    rec = ChunkRecord(
        vector_id=vector_id,
        tenant="t",
        doc_id=f"doc{vector_id}",
        version=1,
        chunk_index=0,
        source=f"doc{vector_id}.md",
        text=text,
        metadata={},
    )
    return RetrievedChunk(record=rec, score=vec_score)


# Candidate pool where the BEST lexical match is NOT first in vector order.
def _candidates() -> list[RetrievedChunk]:
    return [
        _chunk(1, "bread flour water yeast baking dough", vec_score=0.9),
        _chunk(2, "documents are split into overlapping chunks before embedding", 0.8),
        _chunk(3, "faiss vector index nearest neighbor search", 0.7),
    ]


def test_noop_preserves_vector_order():
    r = NoOpReranker()
    out = r.rerank("chunking", _candidates(), top_n=3)
    assert [c.record.vector_id for c in out] == [1, 2, 3]
    # rerank_score mirrors vector score for the no-op reranker.
    assert out[0].rerank_score == out[0].score


def test_lexical_promotes_best_match():
    r = LexicalReranker()
    out = r.rerank(
        "how are documents chunked before embedding?", _candidates(), top_n=3
    )
    # Doc 2 has the query terms; it must be promoted above the vector top (doc 1).
    assert out[0].record.vector_id == 2
    assert out[0].rerank_score is not None
    assert out[0].rerank_score > 0


def test_lexical_truncates_to_top_n():
    r = LexicalReranker()
    out = r.rerank("chunks embedding", _candidates(), top_n=1)
    assert len(out) == 1


def test_reranker_handles_empty_candidates():
    assert LexicalReranker().rerank("q", [], top_n=5) == []
    assert NoOpReranker().rerank("q", [], top_n=5) == []


# --- factory / fallback ---------------------------------------------------- #
def test_factory_none_and_lexical():
    assert build_reranker(Settings(rerank_strategy="none")).name == "none"
    assert build_reranker(Settings(rerank_strategy="lexical")).name == "lexical"


def test_factory_cross_encoder_falls_back_without_dependency():
    # sentence-transformers is not installed in the base env -> lexical fallback.
    r = build_reranker(Settings(rerank_strategy="cross_encoder"))
    assert r.name == "lexical"


def test_factory_llm_falls_back_without_key():
    r = build_reranker(Settings(rerank_strategy="llm", openai_api_key=None))
    assert r.name == "lexical"


def test_legacy_rerank_enabled_flag_maps_to_lexical():
    s = Settings(rerank_strategy="none", rerank_enabled=True)
    assert s.effective_rerank_strategy == "lexical"
    assert build_reranker(s).name == "lexical"


def test_unknown_strategy_falls_back_to_lexical():
    assert build_reranker(Settings(rerank_strategy="bogus")).name == "lexical"
