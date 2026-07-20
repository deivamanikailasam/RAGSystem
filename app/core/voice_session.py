"""Voice session persistence + manager.

Layers state persistence and side effects on top of the pure state machine
(:mod:`app.core.voice_fsm`):

* :class:`VoiceSessionStore` — per-tenant SQLite persistence of each session's
  current FSM state and context, so a voice conversation survives across the
  many HTTP events that make it up.
* :class:`VoiceSessionManager` — applies an event through the FSM, runs the
  side effect for that transition (crucially, on ``TRANSCRIPT`` it runs a RAG
  chat turn and auto-advances ``THINKING → SPEAKING`` with the spoken reply),
  persists the new state, and reports the resulting state + allowed events.

The session id doubles as the chat ``session_id`` (see
:mod:`app.core.conversation`), so the voice session and its dialogue history are
one and the same — the assistant remembers earlier turns within the call.
"""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.voice_fsm import Event, InvalidTransition, State, VoiceStateMachine


@dataclass
class VoiceSession:
    tenant: str
    session_id: str
    state: State
    last_transcript: str | None = None
    last_response: str | None = None
    turn_count: int = 0
    barge_in_count: int = 0
    error: str | None = None
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass
class EventResult:
    session_id: str
    previous_state: State
    event: Event
    state: State
    allowed_events: list[Event]
    turn_count: int
    barge_in_count: int
    # Populated when the turn produced a spoken reply (TRANSCRIPT → SPEAKING).
    say: str | None = None
    citations: list[dict] | None = None
    standalone_question: str | None = None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS voice_sessions (
    tenant         TEXT NOT NULL,
    session_id     TEXT NOT NULL,
    state          TEXT NOT NULL,
    last_transcript TEXT,
    last_response  TEXT,
    turn_count     INTEGER NOT NULL DEFAULT 0,
    barge_in_count INTEGER NOT NULL DEFAULT 0,
    error          TEXT,
    created_at     REAL NOT NULL,
    updated_at     REAL NOT NULL,
    PRIMARY KEY (tenant, session_id)
);
"""


class VoiceSessionStore:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def create(self, tenant: str, session_id: str) -> VoiceSession:
        now = time.time()
        session = VoiceSession(
            tenant=tenant, session_id=session_id, state=State.IDLE,
            created_at=now, updated_at=now,
        )
        with self._lock:
            self._conn.execute(
                """INSERT INTO voice_sessions
                       (tenant, session_id, state, turn_count, barge_in_count,
                        created_at, updated_at)
                   VALUES (?, ?, ?, 0, 0, ?, ?)""",
                (tenant, session_id, session.state.value, now, now),
            )
            self._conn.commit()
        return session

    def get(self, tenant: str, session_id: str) -> VoiceSession | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM voice_sessions WHERE tenant=? AND session_id=?",
                (tenant, session_id),
            ).fetchone()
        return self._row_to_session(row) if row else None

    def save(self, session: VoiceSession) -> None:
        session.updated_at = time.time()
        with self._lock:
            self._conn.execute(
                """UPDATE voice_sessions SET state=?, last_transcript=?,
                       last_response=?, turn_count=?, barge_in_count=?, error=?,
                       updated_at=?
                   WHERE tenant=? AND session_id=?""",
                (session.state.value, session.last_transcript, session.last_response,
                 session.turn_count, session.barge_in_count, session.error,
                 session.updated_at, session.tenant, session.session_id),
            )
            self._conn.commit()

    def delete(self, tenant: str, session_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM voice_sessions WHERE tenant=? AND session_id=?",
                (tenant, session_id),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def delete_tenant(self, tenant: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM voice_sessions WHERE tenant=?", (tenant,))
            self._conn.commit()

    @staticmethod
    def _row_to_session(row: sqlite3.Row) -> VoiceSession:
        return VoiceSession(
            tenant=row["tenant"],
            session_id=row["session_id"],
            state=State(row["state"]),
            last_transcript=row["last_transcript"],
            last_response=row["last_response"],
            turn_count=int(row["turn_count"]),
            barge_in_count=int(row["barge_in_count"]),
            error=row["error"],
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )


class SessionNotFound(Exception):
    pass


class VoiceSessionManager:
    """Drives the FSM, persists sessions, and runs RAG on user transcripts."""

    def __init__(self, store: VoiceSessionStore, engine: Any) -> None:
        self._store = store
        self._engine = engine  # RagEngine (avoids a circular import)
        self._fsm = VoiceStateMachine()

    def create_session(self, tenant: str, session_id: str | None = None) -> VoiceSession:
        return self._store.create(tenant, session_id or uuid.uuid4().hex)

    def get_session(self, tenant: str, session_id: str) -> VoiceSession:
        session = self._store.get(tenant, session_id)
        if session is None:
            raise SessionNotFound(session_id)
        return session

    def delete_session(self, tenant: str, session_id: str) -> bool:
        return self._store.delete(tenant, session_id)

    def allowed_events(self, state: State) -> list[Event]:
        return self._fsm.allowed_events(state)

    def send_event(
        self,
        *,
        tenant: str,
        session_id: str,
        event: Event,
        text: str | None = None,
        message: str | None = None,
        top_k: int | None = None,
        filters: dict[str, str] | None = None,
    ) -> EventResult:
        session = self.get_session(tenant, session_id)
        previous = session.state

        # Validate the transition against the pure FSM (raises InvalidTransition).
        new_state = self._fsm.next_state(previous, event)

        say = citations = standalone = None

        if event is Event.TRANSCRIPT:
            # LISTENING → THINKING → (run RAG) → SPEAKING, all within this call.
            transcript = (text or "").strip()
            if not transcript:
                raise ValueError("TRANSCRIPT event requires non-empty 'text'.")
            chat = self._engine.chat(
                tenant=tenant, message=transcript, session_id=session_id,
                top_k=top_k, filters=filters,
            )
            new_state = self._fsm.next_state(State.THINKING, Event.THINK_DONE)  # SPEAKING
            session.last_transcript = transcript
            session.last_response = chat.answer
            session.turn_count += 1
            say = chat.answer
            citations = [c.model_dump() for c in chat.citations]
            standalone = chat.standalone_question
        elif event is Event.BARGE_IN:
            session.barge_in_count += 1
        elif event is Event.ERROR:
            session.error = message
        elif event is Event.RECOVER:
            session.error = None

        session.state = new_state
        self._store.save(session)

        return EventResult(
            session_id=session_id,
            previous_state=previous,
            event=event,
            state=new_state,
            allowed_events=self._fsm.allowed_events(new_state),
            turn_count=session.turn_count,
            barge_in_count=session.barge_in_count,
            say=say,
            citations=citations,
            standalone_question=standalone,
        )


__all__ = [
    "VoiceSession",
    "VoiceSessionStore",
    "VoiceSessionManager",
    "EventResult",
    "SessionNotFound",
    "InvalidTransition",
]
