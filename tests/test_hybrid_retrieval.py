"""End-to-end tests for the three retrieval modes and hybrid fusion.

Reranking is disabled here (``rerank_strategy="none"``) so we observe the
stage-1 retrieval/fusion ordering directly, without the reranker reshuffling it.
"""

from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.core.rag import RagEngine

SEM = "Documents are split into overlapping chunks before they are embedded into vectors."
CODE = "Support note: the error code XJ1729 means a checksum mismatch happened on upload."
DISTRACTORS = {
    "d1": "The company holiday schedule lists office closure dates for the year.",
    "d2": "Guidelines for booking meeting rooms and hot-desking in the office.",
}


def _engine(tmp_path: Path, mode: str, fusion: str = "rrf") -> RagEngine:
    engine = RagEngine(
        Settings(
            deployment_mode="single_tenant",
            openai_api_key=None,
            data_dir=tmp_path / f"{mode}-{fusion}",
            chunk_tokens=40,
            chunk_overlap=8,
            retrieval_top_k=5,
            retrieval_mode=mode,
            hybrid_fusion=fusion,
            rerank_strategy="none",
        )
    )
    engine.ingest(tenant="default", text=SEM, source="sem.md", doc_id="sem")
    engine.ingest(tenant="default", text=CODE, source="code.md", doc_id="code")
    for did, text in DISTRACTORS.items():
        engine.ingest(tenant="default", text=text, source=f"{did}.md", doc_id=did)
    return engine


def test_bm25_mode_ranks_exact_keyword_first(tmp_path: Path):
    engine = _engine(tmp_path, "bm25")
    resp = engine.query(tenant="default", question="what does error code XJ1729 mean?")
    assert resp.retrieval_mode == "bm25"
    assert resp.citations[0].doc_id == "code"
    # BM25 mode populates bm25_score but not vector_score.
    assert resp.citations[0].bm25_score is not None
    assert resp.citations[0].vector_score is None


def test_vector_mode_populates_vector_score_only(tmp_path: Path):
    engine = _engine(tmp_path, "vector")
    resp = engine.query(tenant="default", question="how are documents chunked?")
    assert resp.retrieval_mode == "vector"
    assert resp.citations
    assert resp.citations[0].vector_score is not None
    assert resp.citations[0].bm25_score is None


def test_hybrid_fuses_both_signals(tmp_path: Path):
    engine = _engine(tmp_path, "hybrid")
    # Query mixes a semantic phrase and an exact code token.
    resp = engine.query(
        tenant="default", question="overlapping chunks and error code XJ1729"
    )
    assert resp.retrieval_mode == "hybrid"
    cited = {c.doc_id for c in resp.citations}
    # Fusion should surface BOTH the semantic doc and the exact-keyword doc.
    assert {"sem", "code"} <= cited
    # At least one citation carries a BM25 contribution.
    assert any(c.bm25_score is not None for c in resp.citations)


def test_hybrid_recovers_keyword_doc_via_bm25(tmp_path: Path):
    """The exact-keyword doc is retrievable in hybrid mode through the BM25 arm."""
    engine = _engine(tmp_path, "hybrid")
    resp = engine.query(tenant="default", question="XJ1729")
    assert "code" in {c.doc_id for c in resp.citations}


def test_weighted_fusion_smoke(tmp_path: Path):
    engine = _engine(tmp_path, "hybrid", fusion="weighted")
    resp = engine.query(tenant="default", question="error code XJ1729 checksum")
    assert resp.citations
    assert "code" in {c.doc_id for c in resp.citations}


def test_bm25_index_invalidated_on_delete(tmp_path: Path):
    engine = _engine(tmp_path, "bm25")
    # Present before deletion.
    assert "code" in {
        c.doc_id
        for c in engine.query(tenant="default", question="XJ1729").citations
    }
    engine.delete_document(tenant="default", doc_id="code")
    # Gone after deletion (index rebuilt without it).
    assert "code" not in {
        c.doc_id
        for c in engine.query(tenant="default", question="XJ1729").citations
    }


def test_hybrid_and_rerank_compose(tmp_path: Path):
    """Hybrid retrieval + lexical reranking should work together."""
    engine = RagEngine(
        Settings(
            deployment_mode="single_tenant",
            openai_api_key=None,
            data_dir=tmp_path / "compose",
            chunk_tokens=40,
            chunk_overlap=8,
            retrieval_mode="hybrid",
            rerank_strategy="lexical",
        )
    )
    engine.ingest(tenant="default", text=SEM, source="sem.md", doc_id="sem")
    engine.ingest(tenant="default", text=CODE, source="code.md", doc_id="code")
    resp = engine.query(tenant="default", question="error code XJ1729")
    assert resp.retrieval_mode == "hybrid"
    assert resp.reranker == "lexical"
    assert resp.citations[0].doc_id == "code"
    assert resp.citations[0].rerank_score is not None
