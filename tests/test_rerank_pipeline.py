"""End-to-end test that reranking is wired into the query pipeline."""

from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.core.rag import RagEngine

# Doc whose FIRST chunk is generic but a LATER chunk directly answers the query.
LAYERED = (
    "This handbook covers many operational topics for the whole company. "
    "General introduction and table of contents follow in the first section. "
    "Refunds policy: a customer may request a refund within thirty days of "
    "purchase by contacting support with the order number."
)
DISTRACTORS = [
    "Company holiday schedule and office closure dates for the year.",
    "Guidelines for booking meeting rooms and desk hoteling.",
    "Instructions for setting up the office coffee machine and kitchen rota.",
]


def _engine(tmp_path: Path, strategy: str) -> RagEngine:
    return RagEngine(
        Settings(
            deployment_mode="single_tenant",
            openai_api_key=None,
            data_dir=tmp_path / strategy,
            chunk_tokens=25,
            chunk_overlap=5,
            retrieval_top_k=3,
            rerank_candidates=10,
            rerank_strategy=strategy,
        )
    )


def _seed(engine: RagEngine) -> None:
    engine.ingest(tenant="default", text=LAYERED, source="handbook.md", doc_id="hb")
    for i, d in enumerate(DISTRACTORS):
        engine.ingest(tenant="default", text=d, source=f"d{i}.md", doc_id=f"d{i}")


def test_reranker_name_reported_in_response(tmp_path: Path):
    engine = _engine(tmp_path, "lexical")
    _seed(engine)
    resp = engine.query(tenant="default", question="How do I request a refund?")
    assert resp.reranker == "lexical"
    assert resp.rerank_ms >= 0.0


def test_rerank_scores_present_on_citations(tmp_path: Path):
    engine = _engine(tmp_path, "lexical")
    _seed(engine)
    resp = engine.query(tenant="default", question="refund within thirty days")
    assert resp.citations
    assert all(c.rerank_score is not None for c in resp.citations)


def test_none_strategy_leaves_scores_equal_to_vector(tmp_path: Path):
    engine = _engine(tmp_path, "none")
    _seed(engine)
    resp = engine.query(tenant="default", question="refund policy")
    assert resp.reranker == "none"
    for c in resp.citations:
        assert c.rerank_score == c.score


def test_lexical_surfaces_the_answer_chunk(tmp_path: Path):
    engine = _engine(tmp_path, "lexical")
    _seed(engine)
    resp = engine.query(
        tenant="default", question="How can a customer request a refund?"
    )
    # The refund chunk from the handbook should be the top citation.
    assert resp.citations[0].doc_id == "hb"
    assert "refund" in resp.citations[0].snippet.lower()
