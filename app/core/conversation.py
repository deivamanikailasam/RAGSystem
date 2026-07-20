"""Conversation store — persistent multi-turn chat history (per tenant).

Backs the multi-turn chatbot (see docs/12-multi-turn-chat.md). Every message is
keyed by ``(tenant, session_id)`` so conversations are isolated per tenant, the
same way documents and vectors are. Stored in its own SQLite file
(``{DATA_DIR}/conversations.db``) so chat state is decoupled from the corpus.

Schema
------
``conversations`` — one row per session: created/updated timestamps.
``messages``       — one row per message: monotonically increasing ``turn_index``,
                     ``role`` ("user"|"assistant"), ``content``, and (for
                     assistant turns) the JSON-encoded citations that grounded it.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Message:
    turn_index: int
    role: str  # "user" | "assistant"
    content: str
    citations: list[dict] = field(default_factory=list)
    created_at: float = 0.0


_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    tenant     TEXT NOT NULL,
    session_id TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (tenant, session_id)
);

CREATE TABLE IF NOT EXISTS messages (
    tenant      TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    turn_index  INTEGER NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    citations   TEXT NOT NULL,
    created_at  REAL NOT NULL,
    PRIMARY KEY (tenant, session_id, turn_index)
);
"""


class ConversationStore:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def exists(self, tenant: str, session_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM conversations WHERE tenant=? AND session_id=?",
                (tenant, session_id),
            ).fetchone()
        return row is not None

    def ensure_session(self, tenant: str, session_id: str) -> None:
        now = time.time()
        with self._lock:
            self._conn.execute(
                """INSERT INTO conversations (tenant, session_id, created_at, updated_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(tenant, session_id) DO NOTHING""",
                (tenant, session_id, now, now),
            )
            self._conn.commit()

    def _next_turn_index(self, tenant: str, session_id: str) -> int:
        row = self._conn.execute(
            "SELECT COALESCE(MAX(turn_index), -1) AS m FROM messages "
            "WHERE tenant=? AND session_id=?",
            (tenant, session_id),
        ).fetchone()
        return int(row["m"]) + 1

    def append_message(
        self,
        *,
        tenant: str,
        session_id: str,
        role: str,
        content: str,
        citations: list[dict] | None = None,
    ) -> int:
        now = time.time()
        with self._lock:
            turn_index = self._next_turn_index(tenant, session_id)
            self._conn.execute(
                """INSERT INTO messages
                       (tenant, session_id, turn_index, role, content, citations, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (tenant, session_id, turn_index, role, content,
                 json.dumps(citations or []), now),
            )
            self._conn.execute(
                "UPDATE conversations SET updated_at=? WHERE tenant=? AND session_id=?",
                (now, tenant, session_id),
            )
            self._conn.commit()
        return turn_index

    def get_messages(self, tenant: str, session_id: str) -> list[Message]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM messages WHERE tenant=? AND session_id=? "
                "ORDER BY turn_index",
                (tenant, session_id),
            ).fetchall()
        return [self._row_to_message(r) for r in rows]

    def recent_messages(self, tenant: str, session_id: str, limit: int) -> list[Message]:
        """The last ``limit`` messages, in chronological order."""
        if limit <= 0:
            return []
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM messages WHERE tenant=? AND session_id=? "
                "ORDER BY turn_index DESC LIMIT ?",
                (tenant, session_id, limit),
            ).fetchall()
        return [self._row_to_message(r) for r in reversed(rows)]

    def delete_session(self, tenant: str, session_id: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM messages WHERE tenant=? AND session_id=?",
                (tenant, session_id),
            )
            self._conn.execute(
                "DELETE FROM conversations WHERE tenant=? AND session_id=?",
                (tenant, session_id),
            )
            self._conn.commit()
            return cur.rowcount

    def delete_tenant(self, tenant: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM messages WHERE tenant=?", (tenant,))
            self._conn.execute("DELETE FROM conversations WHERE tenant=?", (tenant,))
            self._conn.commit()

    @staticmethod
    def _row_to_message(row: sqlite3.Row) -> Message:
        return Message(
            turn_index=int(row["turn_index"]),
            role=row["role"],
            content=row["content"],
            citations=json.loads(row["citations"]),
            created_at=float(row["created_at"]),
        )
