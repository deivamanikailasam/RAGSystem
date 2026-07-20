"""Long-term memory for the FAQ bot (per tenant, per user, across sessions).

Short-term memory is the conversation history (`app/core/conversation.py`) and
lives with a session. **Long-term memory** persists salient facts about a *user*
across sessions — their name, plan, tools, and the topics they've asked about —
so the bot can be genuinely context-aware ("you're on the enterprise plan",
"last time you asked about refunds").

Two pieces:

* :class:`MemoryStore` — SQLite persistence keyed by ``(tenant, user_id)`` with
  de-duplication (re-stating a known fact just refreshes ``last_seen``).
* :func:`extract_memories` — a dependency-free heuristic that pulls explicit
  user facts from a message ("my name is Ada", "I'm on the enterprise plan",
  "I use Python"). Swap for an LLM extractor for richer memories.

Recall ranks memories by relevance to the current query (content-word overlap)
and recency, so the most useful few can be injected into the prompt.
"""

from __future__ import annotations

import re
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

_WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass
class Memory:
    memory_id: str
    kind: str          # "fact" | "topic" | "preference"
    content: str
    created_at: float
    last_seen: float


# --- extraction ------------------------------------------------------------ #
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bmy name is ([a-z][\w'-]*)", re.I), "name is {0}"),
    (re.compile(r"\bi'?m (?:on|using) (?:the )?([\w'-]+) plan\b", re.I), "plan: {0}"),
    (re.compile(r"\bi use ([\w'+#.-]+)", re.I), "uses {0}"),
    (re.compile(r"\bi(?:'m| am) a[n]? ([\w'-]+ ?[\w'-]*)\b", re.I), "role: {0}"),
    # Generic "my X is Y" — but not "name" (covered above) to avoid duplicates.
    (re.compile(r"\bmy (?!name\b)([\w'-]+) is ([\w'@.-]+)", re.I), "{0}: {1}"),
    (re.compile(r"\bi prefer ([\w'+#. -]+?)(?:\.|$)", re.I), "prefers {0}"),
]


def _clean(value: str) -> str:
    return value.strip().rstrip(".,;:!?").strip()


def extract_memories(message: str) -> list[tuple[str, str]]:
    """Return ``(kind, content)`` facts found in the message (de-duplicated)."""
    facts: list[tuple[str, str]] = []
    seen: set[str] = set()
    for pattern, template in _PATTERNS:
        m = pattern.search(message)
        if m:
            content = template.format(*[_clean(g) for g in m.groups()])
            if content not in seen:
                seen.add(content)
                facts.append(("fact", content))
    return facts


_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    tenant     TEXT NOT NULL,
    user_id    TEXT NOT NULL,
    memory_id  TEXT NOT NULL,
    kind       TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at REAL NOT NULL,
    last_seen  REAL NOT NULL,
    PRIMARY KEY (tenant, user_id, memory_id)
);
CREATE INDEX IF NOT EXISTS idx_mem_user ON memories (tenant, user_id);
"""


class MemoryStore:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def remember(self, tenant: str, user_id: str, kind: str, content: str) -> Memory:
        now = time.time()
        with self._lock:
            existing = self._conn.execute(
                "SELECT memory_id, created_at FROM memories "
                "WHERE tenant=? AND user_id=? AND kind=? AND content=?",
                (tenant, user_id, kind, content),
            ).fetchone()
            if existing is not None:
                # Known fact — just refresh recency (dedupe).
                self._conn.execute(
                    "UPDATE memories SET last_seen=? WHERE tenant=? AND user_id=? "
                    "AND memory_id=?",
                    (now, tenant, user_id, existing["memory_id"]),
                )
                self._conn.commit()
                return Memory(existing["memory_id"], kind, content,
                              float(existing["created_at"]), now)
            memory_id = uuid.uuid4().hex[:12]
            self._conn.execute(
                """INSERT INTO memories
                       (tenant, user_id, memory_id, kind, content, created_at, last_seen)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (tenant, user_id, memory_id, kind, content, now, now),
            )
            self._conn.commit()
            return Memory(memory_id, kind, content, now, now)

    def list(self, tenant: str, user_id: str) -> list[Memory]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM memories WHERE tenant=? AND user_id=? "
                "ORDER BY last_seen DESC",
                (tenant, user_id),
            ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def recall(self, tenant: str, user_id: str, query: str | None = None,
               limit: int = 5) -> list[Memory]:
        """Most relevant memories for ``query`` (overlap), else most recent."""
        memories = self.list(tenant, user_id)
        if not memories:
            return []
        if not query:
            return memories[:limit]
        q_tokens = set(_WORD_RE.findall(query.lower()))

        def relevance(m: Memory) -> tuple[int, float]:
            overlap = len(q_tokens & set(_WORD_RE.findall(m.content.lower())))
            return (overlap, m.last_seen)

        return sorted(memories, key=relevance, reverse=True)[:limit]

    def forget(self, tenant: str, user_id: str, memory_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM memories WHERE tenant=? AND user_id=? AND memory_id=?",
                (tenant, user_id, memory_id),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def forget_user(self, tenant: str, user_id: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM memories WHERE tenant=? AND user_id=?", (tenant, user_id)
            )
            self._conn.commit()
            return cur.rowcount

    def delete_tenant(self, tenant: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM memories WHERE tenant=?", (tenant,))
            self._conn.commit()

    @staticmethod
    def _row_to_memory(row: sqlite3.Row) -> Memory:
        return Memory(
            memory_id=row["memory_id"], kind=row["kind"], content=row["content"],
            created_at=float(row["created_at"]), last_seen=float(row["last_seen"]),
        )
