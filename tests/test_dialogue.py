"""Dialogue manager tests: intent routing, persistence, state, slots."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.core.dialogue import Action
from app.core.intents import Intent
from app.core.rag import RagEngine

FAISS_DOC = "FAISS is a similarity-search library that indexes dense vectors."
POLICY_DOC = "Refund policy: customers may request a refund within thirty days."


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
    eng.ingest(tenant="default", text=POLICY_DOC, source="policy.md", doc_id="policy",
               metadata={"doc_type": "policy"})
    return eng


def test_question_routes_to_rag_answer(engine: RagEngine):
    turn = engine.dialogue.handle(tenant="default", message="What is FAISS?")
    assert turn.intent == Intent.QUESTION
    assert turn.action == Action.ANSWER
    assert turn.citations  # grounded answer with sources


def test_greeting_gets_canned_reply_no_rag(engine: RagEngine):
    turn = engine.dialogue.handle(tenant="default", message="hello")
    assert turn.intent == Intent.GREETING
    assert turn.action == Action.GREET
    assert turn.citations == []
    assert "assistant" in turn.answer.lower()


def test_goodbye_closes(engine: RagEngine):
    turn = engine.dialogue.handle(tenant="default", message="thanks, that's all")
    assert turn.intent == Intent.GOODBYE
    assert turn.action == Action.CLOSE


def test_intents_persisted_in_order(engine: RagEngine):
    sid = engine.dialogue.handle(tenant="default", message="hi").session_id
    engine.dialogue.handle(tenant="default", message="What is FAISS?", session_id=sid)
    engine.dialogue.handle(tenant="default", message="bye", session_id=sid)

    intents = engine.dialogue.get_intents(tenant="default", session_id=sid)
    assert [e.intent for e in intents] == ["greeting", "question", "goodbye"]
    assert [e.turn_index for e in intents] == [0, 1, 2]


def test_dialogue_state_tracks_current_intent_and_count(engine: RagEngine):
    sid = engine.dialogue.handle(tenant="default", message="hi").session_id
    engine.dialogue.handle(tenant="default", message="What is FAISS?", session_id=sid)
    state = engine.dialogue.get_state(tenant="default", session_id=sid)
    assert state.current_intent == "question"
    assert state.turn_count == 2


def test_slots_feed_retrieval_filter(engine: RagEngine):
    # "in the policy docs" -> doc_type=policy slot -> filters retrieval.
    turn = engine.dialogue.handle(
        tenant="default", message="How are refunds handled in the policy docs?"
    )
    assert turn.slots.get("doc_type") == "policy"
    assert turn.citations and all(c["doc_id"] == "policy" for c in turn.citations)


def test_slots_accumulate_in_state(engine: RagEngine):
    sid = engine.dialogue.handle(
        tenant="default", message="refunds in the policy docs?"
    ).session_id
    state = engine.dialogue.get_state(tenant="default", session_id=sid)
    assert state.slots.get("doc_type") == "policy"


def test_conversation_memory_shared(engine: RagEngine):
    # QUESTION turns go through chat, so history is recorded under the session id.
    sid = engine.dialogue.handle(tenant="default", message="What is FAISS?").session_id
    msgs = engine.get_conversation(tenant="default", session_id=sid)
    assert [m.role for m in msgs] == ["user", "assistant"]


def test_delete_dialogue(engine: RagEngine):
    sid = engine.dialogue.handle(tenant="default", message="hi").session_id
    assert engine.dialogue.delete_session(tenant="default", session_id=sid) >= 1
    assert engine.dialogue.get_state(tenant="default", session_id=sid) is None


def test_tenant_isolation(tmp_path: Path):
    eng = RagEngine(
        Settings(deployment_mode="multi_tenant", openai_api_key=None,
                 data_dir=tmp_path / "d", api_keys="")
    )
    sid = eng.dialogue.handle(tenant="a", message="hi").session_id
    assert eng.dialogue.get_state(tenant="b", session_id=sid) is None
