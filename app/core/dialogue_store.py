"""Dialogue state + intent persistence (per tenant).

This is the "memory" of the dialogue manager. Two things are persisted, keyed by
``(tenant, session_id)`` so they are tenant-isolated like everything else:

* ``dialogue_state`` — the running state of a conversation: the most recent
  intent, a turn counter, and accumulated slots.
* ``intent_events`` — an append-only log of every classified turn (message,
  intent, confidence, chosen action, slots). This is the **intent persistence**:
  a durable trail you can inspect, audit, or train on.

Stored in its own SQLite file (``{DATA_DIR}/dialogue.db``).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class IntentEvent:
    turn_index: int
    message: str
    intent: str
    confidence: float
    action: str
    slots: dict[str, str] = field(default_factory=dict)
    created_at: float = 0.0


@dataclass
class DialogueState:
    tenant: str
    session_id: str
    current_intent: str | None = None
    turn_count: int = 0
    slots: dict[str, str] = field(default_factory=dict)
    created_at: float = 0.0
    updated_at: float = 0.0


_SCHEMA = """
CREATE TABLE IF NOT EXISTS dialogue_state (
    tenant         TEXT NOT NULL,
    session_id     TEXT NOT NULL,
    current_intent TEXT,
    turn_count     INTEGER NOT NULL DEFAULT 0,
    slots          TEXT NOT NULL DEFAULT '{}',
    created_at     REAL NOT NULL,
    updated_at     REAL NOT NULL,
    PRIMARY KEY (tenant, session_id)
);

CREATE TABLE IF NOT EXISTS intent_events (
    tenant      TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    turn_index  INTEGER NOT NULL,
    message     TEXT NOT NULL,
    intent      TEXT NOT NULL,
    confidence  REAL NOT NULL,
    action      TEXT NOT NULL,
    slots       TEXT NOT NULL,
    created_at  REAL NOT NULL,
    PRIMARY KEY (tenant, session_id, turn_index)
);
"""


class DialogueStore:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def get_state(self, tenant: str, session_id: str) -> DialogueState | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM dialogue_state WHERE tenant=? AND session_id=?",
                (tenant, session_id),
            ).fetchone()
        if row is None:
            return None
        return DialogueState(
            tenant=row["tenant"],
            session_id=row["session_id"],
            current_intent=row["current_intent"],
            turn_count=int(row["turn_count"]),
            slots=json.loads(row["slots"]),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )

    def record_turn(
        self,
        *,
        tenant: str,
        session_id: str,
        message: str,
        intent: str,
        confidence: float,
        action: str,
        slots: dict[str, str],
    ) -> IntentEvent:
        """Persist one classified turn and advance the dialogue state."""
        now = time.time()
        with self._lock:
            state = self._conn.execute(
                "SELECT turn_count, slots, created_at FROM dialogue_state "
                "WHERE tenant=? AND session_id=?",
                (tenant, session_id),
            ).fetchone()
            if state is None:
                turn_index = 0
                merged = dict(slots)
                created = now
            else:
                turn_index = int(state["turn_count"])
                merged = {**json.loads(state["slots"]), **slots}
                created = float(state["created_at"])

            self._conn.execute(
                """INSERT INTO intent_events
                       (tenant, session_id, turn_index, message, intent,
                        confidence, action, slots, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (tenant, session_id, turn_index, message, intent, confidence,
                 action, json.dumps(slots), now),
            )
            self._conn.execute(
                """INSERT INTO dialogue_state
                       (tenant, session_id, current_intent, turn_count, slots,
                        created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(tenant, session_id) DO UPDATE SET
                       current_intent=excluded.current_intent,
                       turn_count=excluded.turn_count,
                       slots=excluded.slots,
                       updated_at=excluded.updated_at""",
                (tenant, session_id, intent, turn_index + 1, json.dumps(merged),
                 created, now),
            )
            self._conn.commit()

        return IntentEvent(
            turn_index=turn_index, message=message, intent=intent,
            confidence=confidence, action=action, slots=slots, created_at=now,
        )

    def get_intents(self, tenant: str, session_id: str) -> list[IntentEvent]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM intent_events WHERE tenant=? AND session_id=? "
                "ORDER BY turn_index",
                (tenant, session_id),
            ).fetchall()
        return [
            IntentEvent(
                turn_index=int(r["turn_index"]),
                message=r["message"],
                intent=r["intent"],
                confidence=float(r["confidence"]),
                action=r["action"],
                slots=json.loads(r["slots"]),
                created_at=float(r["created_at"]),
            )
            for r in rows
        ]

    def delete_session(self, tenant: str, session_id: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM intent_events WHERE tenant=? AND session_id=?",
                (tenant, session_id),
            )
            self._conn.execute(
                "DELETE FROM dialogue_state WHERE tenant=? AND session_id=?",
                (tenant, session_id),
            )
            self._conn.commit()
            return cur.rowcount

    def delete_tenant(self, tenant: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM intent_events WHERE tenant=?", (tenant,))
            self._conn.execute("DELETE FROM dialogue_state WHERE tenant=?", (tenant,))
            self._conn.commit()
