"""Multi-turn chat: context tracking, persistence, isolation, condensing."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.core.rag import RagEngine

FAISS_DOC = (
    "FAISS is a library for similarity search over dense vectors. Its IVF index "
    "reduces latency by scanning only a few partitions controlled by nprobe."
)
BREAD_DOC = "Sourdough bread needs flour, water, salt and a fermented starter."


@pytest.fixture
def engine(tmp_path: Path) -> RagEngine:
    eng = RagEngine(
        Settings(
            deployment_mode="single_tenant",
            openai_api_key=None,
            data_dir=tmp_path / "d",
            chunk_tokens=60,
            chunk_overlap=10,
        )
    )
    eng.ingest(tenant="default", text=FAISS_DOC, source="faiss.md", doc_id="faiss")
    eng.ingest(tenant="default", text=BREAD_DOC, source="bread.md", doc_id="bread")
    return eng


def test_new_session_id_minted(engine: RagEngine):
    resp = engine.chat(tenant="default", message="What is FAISS?")
    assert resp.session_id
    assert resp.turn_index == 1  # user=0, assistant=1
    assert resp.citations


def test_history_persisted_and_retrievable(engine: RagEngine):
    r1 = engine.chat(tenant="default", message="What is FAISS?")
    engine.chat(tenant="default", message="what about its latency?",
                session_id=r1.session_id)
    msgs = engine.get_conversation(tenant="default", session_id=r1.session_id)
    # 2 turns -> 4 messages (user/assistant x2), in order.
    assert [m.role for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert msgs[0].content == "What is FAISS?"


def test_followup_condensed_with_context(engine: RagEngine):
    r1 = engine.chat(tenant="default", message="How does FAISS reduce latency?")
    # A bare follow-up; offline condenser should fold in the prior FAISS topic.
    r2 = engine.chat(tenant="default", message="what about nprobe?",
                     session_id=r1.session_id)
    assert "FAISS" in r2.standalone_question or "latency" in r2.standalone_question
    # Still retrieves the FAISS doc, not the bread doc.
    assert r2.citations and r2.citations[0].doc_id == "faiss"


def test_turn_index_increments(engine: RagEngine):
    r1 = engine.chat(tenant="default", message="What is FAISS?")
    r2 = engine.chat(tenant="default", message="tell me more",
                     session_id=r1.session_id)
    assert r2.turn_index == 3  # messages 0,1 (turn1), 2,3 (turn2)


def test_delete_conversation(engine: RagEngine):
    r1 = engine.chat(tenant="default", message="What is FAISS?")
    deleted = engine.delete_conversation(tenant="default", session_id=r1.session_id)
    assert deleted >= 2
    assert engine.get_conversation(tenant="default", session_id=r1.session_id) == []


def test_assistant_citations_stored_in_history(engine: RagEngine):
    r1 = engine.chat(tenant="default", message="What is FAISS?")
    msgs = engine.get_conversation(tenant="default", session_id=r1.session_id)
    assistant = msgs[1]
    assert assistant.role == "assistant"
    assert assistant.citations  # citations persisted with the turn


def test_conversation_tenant_isolation(tmp_path: Path):
    eng = RagEngine(
        Settings(deployment_mode="multi_tenant", openai_api_key=None,
                 data_dir=tmp_path / "d", api_keys="")
    )
    eng.ingest(tenant="a", text=FAISS_DOC, source="f.md", doc_id="faiss")
    r = eng.chat(tenant="a", message="What is FAISS?")
    # Tenant B cannot see tenant A's session, even with the same id.
    assert eng.get_conversation(tenant="b", session_id=r.session_id) == []


def test_condenser_name_reported(engine: RagEngine):
    resp = engine.chat(tenant="default", message="What is FAISS?")
    assert resp.condenser == "heuristic"  # offline default
