"""Context-aware FAQ bot end-to-end tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.core.rag import RagEngine

FAISS_DOC = "FAISS is a similarity-search library that indexes dense vectors for retrieval."


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
    eng.add_faq(tenant="default", question="How do I reset my password?",
                answer="Go to Settings > Security > Reset password.", tags=["auth"])
    return eng


def test_faq_hit_returns_curated_answer(engine: RagEngine):
    r = engine.faq_bot.ask(tenant="default", message="How can I reset my password?")
    assert r.source == "faq"
    assert r.answer == "Go to Settings > Security > Reset password."
    assert r.faq_id and r.score >= 0.45
    assert r.citations[0]["type"] == "faq"


def test_rag_fallback_when_no_faq_matches(engine: RagEngine):
    r = engine.faq_bot.ask(tenant="default", message="How does FAISS index vectors?")
    assert r.source == "rag"
    assert r.citations and r.citations[0]["type"] == "doc"
    assert r.citations[0]["doc_id"] == "faiss"


def test_memory_persists_across_sessions(engine: RagEngine):
    # Session 1: state a fact.
    engine.faq_bot.ask(tenant="default", user_id="u1", session_id="s1",
                       message="Hi, my name is Ada.")
    # Session 2 (new session, same user): the fact is recalled.
    r = engine.faq_bot.ask(tenant="default", user_id="u1", session_id="s2",
                           message="How does FAISS index vectors?")
    assert "name is Ada" in r.memories_used


def test_topic_memory_recorded(engine: RagEngine):
    engine.faq_bot.ask(tenant="default", user_id="u1", message="How does FAISS work?")
    kinds = {m.kind for m in engine.memory.list("default", "u1")}
    assert "topic" in kinds


def test_conversation_history_persisted(engine: RagEngine):
    r = engine.faq_bot.ask(tenant="default", message="How does FAISS index vectors?")
    msgs = engine.get_conversation(tenant="default", session_id=r.session_id)
    assert [m.role for m in msgs] == ["user", "assistant"]


def test_follow_up_condensed(engine: RagEngine):
    r1 = engine.faq_bot.ask(tenant="default", message="How does FAISS reduce latency?")
    r2 = engine.faq_bot.ask(tenant="default", session_id=r1.session_id,
                            message="what about it?")
    assert "FAISS" in r2.standalone_question


def test_user_id_defaults_to_session(engine: RagEngine):
    r = engine.faq_bot.ask(tenant="default", session_id="s9", message="hello there")
    assert r.user_id == "s9"


def test_memory_isolation_between_users(engine: RagEngine):
    engine.faq_bot.ask(tenant="default", user_id="alice", session_id="sa",
                       message="my name is Alice")
    r = engine.faq_bot.ask(tenant="default", user_id="bob", session_id="sb",
                           message="How does FAISS work?")
    assert "name is Alice" not in r.memories_used


def test_disabled_memory(tmp_path: Path):
    eng = RagEngine(
        Settings(deployment_mode="single_tenant", openai_api_key=None,
                 data_dir=tmp_path / "d", memory_enabled=False)
    )
    eng.ingest(tenant="default", text=FAISS_DOC, source="f.md", doc_id="faiss")
    eng.faq_bot.ask(tenant="default", user_id="u1", message="my name is Ada")
    assert eng.memory.list("default", "u1") == []
