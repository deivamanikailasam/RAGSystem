"""SQLite-backed metadata store.

We deliberately separate *vectors* (FAISS) from *metadata* (this docstore) so
the two can evolve independently: you can rebuild or re-tune the FAISS index
structure without migrating document content, and vice versa.

Schema
------
``documents``  — one row per ingested document version.
``chunks``     — one row per chunk; ``vector_id`` is the integer id used inside
                 the FAISS ``IndexIDMap``. ``(tenant, vector_id)`` is unique.

Everything is keyed by ``tenant`` for isolation. A production deployment would
likely use Postgres; the interface here is intentionally small so swapping the
backend is mechanical.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ChunkRecord:
    vector_id: int
    tenant: str
    doc_id: str
    version: int
    chunk_index: int
    source: str
    text: str
    metadata: dict[str, str]


_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    tenant      TEXT NOT NULL,
    doc_id      TEXT NOT NULL,
    version     INTEGER NOT NULL,
    source      TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    created_at  REAL NOT NULL,
    PRIMARY KEY (tenant, doc_id)
);

CREATE TABLE IF NOT EXISTS chunks (
    tenant      TEXT NOT NULL,
    vector_id   INTEGER NOT NULL,
    doc_id      TEXT NOT NULL,
    version     INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    source      TEXT NOT NULL,
    text        TEXT NOT NULL,
    metadata    TEXT NOT NULL,
    PRIMARY KEY (tenant, vector_id)
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks (tenant, doc_id);
"""


class DocStore:
    """Thread-safe SQLite metadata store."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # -- documents --------------------------------------------------------- #
    def get_document(self, tenant: str, doc_id: str) -> sqlite3.Row | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM documents WHERE tenant=? AND doc_id=?",
                (tenant, doc_id),
            )
            return cur.fetchone()

    def next_version(self, tenant: str, doc_id: str) -> int:
        existing = self.get_document(tenant, doc_id)
        return 1 if existing is None else int(existing["version"]) + 1

    def upsert_document(
        self,
        *,
        tenant: str,
        doc_id: str,
        version: int,
        source: str,
        content_hash: str,
        created_at: float,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO documents (tenant, doc_id, version, source, content_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(tenant, doc_id) DO UPDATE SET
                    version=excluded.version,
                    source=excluded.source,
                    content_hash=excluded.content_hash,
                    created_at=excluded.created_at
                """,
                (tenant, doc_id, version, source, content_hash, created_at),
            )
            self._conn.commit()

    # -- chunks ------------------------------------------------------------ #
    def max_vector_id(self, tenant: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "SELECT COALESCE(MAX(vector_id), -1) AS m FROM chunks WHERE tenant=?",
                (tenant,),
            )
            return int(cur.fetchone()["m"])

    def max_vector_id_all(self) -> int:
        """Global max vector id across all tenants.

        Used to allocate globally-unique ids in shared-namespace isolation,
        where every tenant's vectors live in one index and ids must not collide.
        """
        with self._lock:
            cur = self._conn.execute(
                "SELECT COALESCE(MAX(vector_id), -1) AS m FROM chunks"
            )
            return int(cur.fetchone()["m"])

    def vector_ids_for_tenant(self, tenant: str) -> list[int]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT vector_id FROM chunks WHERE tenant=?", (tenant,)
            )
            return [int(row["vector_id"]) for row in cur.fetchall()]

    def add_chunks(self, records: list[ChunkRecord]) -> None:
        if not records:
            return
        with self._lock:
            self._conn.executemany(
                """
                INSERT OR REPLACE INTO chunks
                    (tenant, vector_id, doc_id, version, chunk_index, source, text, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        r.tenant,
                        r.vector_id,
                        r.doc_id,
                        r.version,
                        r.chunk_index,
                        r.source,
                        r.text,
                        json.dumps(r.metadata),
                    )
                    for r in records
                ],
            )
            self._conn.commit()

    def get_chunks(self, tenant: str, vector_ids: list[int]) -> dict[int, ChunkRecord]:
        if not vector_ids:
            return {}
        placeholders = ",".join("?" for _ in vector_ids)
        with self._lock:
            cur = self._conn.execute(
                f"SELECT * FROM chunks WHERE tenant=? AND vector_id IN ({placeholders})",
                (tenant, *vector_ids),
            )
            rows = cur.fetchall()
        return {int(row["vector_id"]): self._row_to_record(row) for row in rows}

    def vector_ids_for_doc(self, tenant: str, doc_id: str) -> list[int]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT vector_id FROM chunks WHERE tenant=? AND doc_id=?",
                (tenant, doc_id),
            )
            return [int(row["vector_id"]) for row in cur.fetchall()]

    def all_chunks(self, tenant: str) -> list[ChunkRecord]:
        """Return every chunk for a tenant (used to build the BM25 index)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM chunks WHERE tenant=? ORDER BY vector_id", (tenant,)
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def count_documents(self, tenant: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "SELECT COUNT(*) AS c FROM documents WHERE tenant=?", (tenant,)
            )
            return int(cur.fetchone()["c"])

    def delete_tenant(self, tenant: str) -> int:
        """Remove all documents and chunks for a tenant (used when purging)."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM chunks WHERE tenant=?", (tenant,)
            )
            self._conn.execute("DELETE FROM documents WHERE tenant=?", (tenant,))
            self._conn.commit()
            return cur.rowcount

    def delete_doc_chunks(self, tenant: str, doc_id: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM chunks WHERE tenant=? AND doc_id=?", (tenant, doc_id)
            )
            self._conn.execute(
                "DELETE FROM documents WHERE tenant=? AND doc_id=?", (tenant, doc_id)
            )
            self._conn.commit()
            return cur.rowcount

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> ChunkRecord:
        return ChunkRecord(
            vector_id=int(row["vector_id"]),
            tenant=row["tenant"],
            doc_id=row["doc_id"],
            version=int(row["version"]),
            chunk_index=int(row["chunk_index"]),
            source=row["source"],
            text=row["text"],
            metadata=json.loads(row["metadata"]),
        )
