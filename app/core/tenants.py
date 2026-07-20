"""Tenant registry — the multi-tenant control plane.

In single-tenant mode this module is unused: there is exactly one implicit
tenant and no registry. In multi-tenant mode the registry is the source of
truth for *which tenants exist* and *their per-tenant policy*:

* identity     — an optional issued API key (stored hashed, never in plaintext);
* policy       — a custom system-prompt template and FAISS index type;
* quotas       — max documents and max queries/day (0 == unlimited);
* lifecycle    — enabled/disabled, created timestamp;
* usage        — per-day query counters for quota enforcement.

Backed by its own SQLite file (``{DATA_DIR}/tenants.db``) so it is decoupled
from the document/vector storage. Swap for Postgres in production — the method
surface is small and stable.
"""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path


# --------------------------------------------------------------------------- #
# Errors (translated to HTTP status codes by app/main.py exception handlers)
# --------------------------------------------------------------------------- #
class TenantError(Exception):
    """Base class for tenant-related failures."""


class TenantNotFound(TenantError):
    pass


class TenantDisabled(TenantError):
    pass


class TenantExists(TenantError):
    pass


class QuotaExceeded(TenantError):
    def __init__(self, quota: str, limit: int) -> None:
        super().__init__(f"Quota '{quota}' exceeded (limit={limit}).")
        self.quota = quota
        self.limit = limit


@dataclass
class Tenant:
    tenant_id: str
    name: str
    prompt_template: str | None
    index_type: str
    max_documents: int
    max_queries_per_day: int
    disabled: bool
    created_at: float

    def public_dict(self) -> dict[str, object]:
        return {
            "tenant_id": self.tenant_id,
            "name": self.name,
            "prompt_template": self.prompt_template,
            "index_type": self.index_type,
            "max_documents": self.max_documents,
            "max_queries_per_day": self.max_queries_per_day,
            "disabled": self.disabled,
            "created_at": self.created_at,
        }


def hash_api_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def generate_api_key() -> str:
    """A URL-safe secret key. Shown to the admin once, only its hash is stored."""
    return "qas_" + secrets.token_urlsafe(32)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS tenants (
    tenant_id           TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    api_key_hash        TEXT,
    prompt_template     TEXT,
    index_type          TEXT NOT NULL,
    max_documents       INTEGER NOT NULL DEFAULT 0,
    max_queries_per_day INTEGER NOT NULL DEFAULT 0,
    disabled            INTEGER NOT NULL DEFAULT 0,
    created_at          REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tenants_key ON tenants (api_key_hash);

CREATE TABLE IF NOT EXISTS tenant_usage (
    tenant_id TEXT NOT NULL,
    day       TEXT NOT NULL,
    queries   INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (tenant_id, day)
);
"""


class TenantRegistry:
    def __init__(self, db_path: Path, default_index_type: str) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._default_index_type = default_index_type
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # -- lifecycle --------------------------------------------------------- #
    def create(
        self,
        *,
        tenant_id: str,
        name: str | None = None,
        api_key: str | None = None,
        prompt_template: str | None = None,
        index_type: str | None = None,
        max_documents: int = 0,
        max_queries_per_day: int = 0,
    ) -> tuple[Tenant, str | None]:
        """Create a tenant. Returns (tenant, api_key_plaintext_or_None).

        The plaintext key is returned exactly once (either the one supplied or a
        freshly generated one) and never persisted — only its hash is stored.
        """
        if self.get(tenant_id) is not None:
            raise TenantExists(f"Tenant '{tenant_id}' already exists.")

        issued_key: str | None = None
        key_hash: str | None = None
        if api_key is not None:
            key_hash = hash_api_key(api_key)
            issued_key = api_key
        else:
            issued_key = generate_api_key()
            key_hash = hash_api_key(issued_key)

        tenant = Tenant(
            tenant_id=tenant_id,
            name=name or tenant_id,
            prompt_template=prompt_template,
            index_type=index_type or self._default_index_type,
            max_documents=max_documents,
            max_queries_per_day=max_queries_per_day,
            disabled=False,
            created_at=time.time(),
        )
        with self._lock:
            self._conn.execute(
                """INSERT INTO tenants (tenant_id, name, api_key_hash, prompt_template,
                       index_type, max_documents, max_queries_per_day, disabled, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)""",
                (
                    tenant.tenant_id, tenant.name, key_hash, tenant.prompt_template,
                    tenant.index_type, tenant.max_documents,
                    tenant.max_queries_per_day, tenant.created_at,
                ),
            )
            self._conn.commit()
        return tenant, issued_key

    def ensure(self, tenant_id: str, name: str | None = None) -> Tenant:
        """Idempotently ensure a tenant row exists (used to seed static keys)."""
        existing = self.get(tenant_id)
        if existing is not None:
            return existing
        tenant, _ = self.create(tenant_id=tenant_id, name=name, api_key=None)
        return tenant

    def get(self, tenant_id: str) -> Tenant | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM tenants WHERE tenant_id=?", (tenant_id,)
            ).fetchone()
        return self._row_to_tenant(row) if row else None

    def get_by_api_key(self, api_key: str) -> Tenant | None:
        key_hash = hash_api_key(api_key)
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM tenants WHERE api_key_hash=?", (key_hash,)
            ).fetchone()
        return self._row_to_tenant(row) if row else None

    def list(self) -> list[Tenant]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM tenants ORDER BY created_at"
            ).fetchall()
        return [self._row_to_tenant(r) for r in rows]

    def update(self, tenant_id: str, **fields: object) -> Tenant:
        allowed = {
            "name", "prompt_template", "index_type",
            "max_documents", "max_queries_per_day", "disabled",
        }
        updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
        if not self.get(tenant_id):
            raise TenantNotFound(tenant_id)
        if updates:
            if "disabled" in updates:
                updates["disabled"] = 1 if updates["disabled"] else 0
            assignments = ", ".join(f"{k}=?" for k in updates)
            with self._lock:
                self._conn.execute(
                    f"UPDATE tenants SET {assignments} WHERE tenant_id=?",
                    (*updates.values(), tenant_id),
                )
                self._conn.commit()
        result = self.get(tenant_id)
        assert result is not None
        return result

    def delete(self, tenant_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM tenants WHERE tenant_id=?", (tenant_id,))
            self._conn.execute(
                "DELETE FROM tenant_usage WHERE tenant_id=?", (tenant_id,)
            )
            self._conn.commit()

    # -- usage / quotas ---------------------------------------------------- #
    def queries_today(self, tenant_id: str) -> int:
        day = date.today().isoformat()
        with self._lock:
            row = self._conn.execute(
                "SELECT queries FROM tenant_usage WHERE tenant_id=? AND day=?",
                (tenant_id, day),
            ).fetchone()
        return int(row["queries"]) if row else 0

    def record_query(self, tenant_id: str) -> int:
        day = date.today().isoformat()
        with self._lock:
            self._conn.execute(
                """INSERT INTO tenant_usage (tenant_id, day, queries)
                   VALUES (?, ?, 1)
                   ON CONFLICT(tenant_id, day)
                   DO UPDATE SET queries = queries + 1""",
                (tenant_id, day),
            )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT queries FROM tenant_usage WHERE tenant_id=? AND day=?",
                (tenant_id, day),
            ).fetchone()
        return int(row["queries"])

    @staticmethod
    def _row_to_tenant(row: sqlite3.Row) -> Tenant:
        return Tenant(
            tenant_id=row["tenant_id"],
            name=row["name"],
            prompt_template=row["prompt_template"],
            index_type=row["index_type"],
            max_documents=int(row["max_documents"]),
            max_queries_per_day=int(row["max_queries_per_day"]),
            disabled=bool(row["disabled"]),
            created_at=float(row["created_at"]),
        )
