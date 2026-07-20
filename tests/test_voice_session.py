"""Voice session manager tests: FSM + persistence + RAG integration."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.core.rag import RagEngine
from app.core.voice_fsm import Event, InvalidTransition, State
from app.core.voice_session import SessionNotFound

FAISS_DOC = (
    "FAISS is a similarity-search library. Its IVF index lowers latency by "
    "scanning only a few partitions controlled by nprobe."
)


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
    return eng


def _start(engine: RagEngine):
    session = engine.voice.create_session("default")
    engine.voice.send_event(
        tenant="default", session_id=session.session_id, event=Event.START
    )
    return session.session_id


def test_create_starts_idle(engine: RagEngine):
    session = engine.voice.create_session("default")
    assert session.state == State.IDLE


def test_full_turn_cycle(engine: RagEngine):
    sid = _start(engine)
    # Listening -> transcript -> (RAG) -> speaking with a spoken reply.
    result = engine.voice.send_event(
        tenant="default", session_id=sid, event=Event.TRANSCRIPT,
        text="What is FAISS?",
    )
    assert result.state == State.SPEAKING
    assert result.say  # text to speak
    assert result.citations
    assert result.turn_count == 1

    # Finish speaking -> back to listening.
    done = engine.voice.send_event(
        tenant="default", session_id=sid, event=Event.SPEAK_DONE
    )
    assert done.state == State.LISTENING


def test_barge_in_counts_and_returns_to_listening(engine: RagEngine):
    sid = _start(engine)
    engine.voice.send_event(tenant="default", session_id=sid, event=Event.TRANSCRIPT,
                            text="What is FAISS?")
    r = engine.voice.send_event(tenant="default", session_id=sid, event=Event.BARGE_IN)
    assert r.state == State.LISTENING
    assert r.barge_in_count == 1


def test_illegal_event_raises(engine: RagEngine):
    sid = _start(engine)  # LISTENING
    with pytest.raises(InvalidTransition):
        engine.voice.send_event(
            tenant="default", session_id=sid, event=Event.SPEAK_DONE
        )


def test_transcript_requires_text(engine: RagEngine):
    sid = _start(engine)
    with pytest.raises(ValueError):
        engine.voice.send_event(
            tenant="default", session_id=sid, event=Event.TRANSCRIPT, text="  "
        )


def test_state_persists_across_manager_reads(engine: RagEngine):
    sid = _start(engine)
    engine.voice.send_event(tenant="default", session_id=sid, event=Event.TRANSCRIPT,
                            text="What is FAISS?")
    reloaded = engine.voice.get_session("default", sid)
    assert reloaded.state == State.SPEAKING
    assert reloaded.last_transcript == "What is FAISS?"


def test_conversation_memory_shared_with_voice_session(engine: RagEngine):
    sid = _start(engine)
    engine.voice.send_event(tenant="default", session_id=sid, event=Event.TRANSCRIPT,
                            text="What is FAISS?")
    # The voice session id doubles as the chat session id.
    msgs = engine.get_conversation(tenant="default", session_id=sid)
    assert [m.role for m in msgs] == ["user", "assistant"]


def test_multi_turn_follow_up_resolves(engine: RagEngine):
    sid = _start(engine)
    engine.voice.send_event(tenant="default", session_id=sid, event=Event.TRANSCRIPT,
                            text="How does FAISS reduce latency?")
    engine.voice.send_event(tenant="default", session_id=sid, event=Event.SPEAK_DONE)
    r = engine.voice.send_event(tenant="default", session_id=sid, event=Event.TRANSCRIPT,
                                text="what about nprobe?")
    assert r.standalone_question and "FAISS" in r.standalone_question
    assert r.citations and r.citations[0]["doc_id"] == "faiss"


def test_unknown_session_raises(engine: RagEngine):
    with pytest.raises(SessionNotFound):
        engine.voice.get_session("default", "nope")


def test_delete_session(engine: RagEngine):
    sid = _start(engine)
    assert engine.voice.delete_session("default", sid) is True
    with pytest.raises(SessionNotFound):
        engine.voice.get_session("default", sid)


def test_tenant_isolation(tmp_path: Path):
    eng = RagEngine(
        Settings(deployment_mode="multi_tenant", openai_api_key=None,
                 data_dir=tmp_path / "d", api_keys="")
    )
    session = eng.voice.create_session("a")
    # Tenant B cannot see tenant A's voice session.
    with pytest.raises(SessionNotFound):
        eng.voice.get_session("b", session.session_id)
