"""Multi-tenant platform tests: control plane, isolation, quotas."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.core.rag import RagEngine
from app.deps import get_engine
from app.main import create_app

ADMIN = {"Authorization": "Bearer admin-secret"}


def _client(tmp_path: Path):
    settings = Settings(
        deployment_mode="multi_tenant",
        admin_api_key="admin-secret",
        openai_api_key=None,
        data_dir=tmp_path / "data",
        chunk_tokens=40,
        chunk_overlap=8,
        api_keys="",  # no static keys; tenants are created via the admin API
    )
    app = create_app()
    engine = RagEngine(settings)
    app.dependency_overrides[get_engine] = lambda: engine
    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app), engine


def _create_tenant(client: TestClient, tenant_id: str, **kwargs) -> str:
    body = {"tenant_id": tenant_id, "name": tenant_id, **kwargs}
    r = client.post("/admin/tenants", headers=ADMIN, json=body)
    assert r.status_code == 200, r.text
    return r.json()["api_key"]


def test_admin_requires_admin_key(tmp_path: Path):
    client, _ = _client(tmp_path)
    assert client.get("/admin/tenants").status_code == 401  # no token
    r = client.get("/admin/tenants", headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 403


def test_create_tenant_returns_key_once(tmp_path: Path):
    client, _ = _client(tmp_path)
    key = _create_tenant(client, "acme")
    assert key and key.startswith("qas_")
    # Duplicate create -> 409.
    dup = client.post("/admin/tenants", headers=ADMIN, json={"tenant_id": "acme"})
    assert dup.status_code == 409


def test_issued_key_authorizes_tenant(tmp_path: Path):
    client, _ = _client(tmp_path)
    key = _create_tenant(client, "acme")
    hdr = {"Authorization": f"Bearer {key}"}
    ing = client.post(
        "/v1/ingest",
        headers=hdr,
        json={"documents": [{"doc_id": "d", "source": "s", "text": "FAISS stores dense vectors."}]},
    )
    assert ing.status_code == 200
    me = client.get("/v1/me", headers=hdr).json()
    assert me["tenant"] == "acme"
    assert me["documents"] == 1


def test_tenant_isolation(tmp_path: Path):
    client, _ = _client(tmp_path)
    key_a = _create_tenant(client, "tenant-a")
    key_b = _create_tenant(client, "tenant-b")
    client.post(
        "/v1/ingest",
        headers={"Authorization": f"Bearer {key_a}"},
        json={"documents": [{"doc_id": "d", "source": "s", "text": "secret alpha content about vectors"}]},
    )
    # Tenant B cannot see tenant A's documents.
    r = client.post(
        "/v1/query",
        headers={"Authorization": f"Bearer {key_b}"},
        json={"question": "what is the secret content?"},
    )
    assert r.status_code == 200
    assert r.json()["citations"] == []


def test_document_quota_enforced(tmp_path: Path):
    client, _ = _client(tmp_path)
    key = _create_tenant(client, "small", max_documents=1)
    hdr = {"Authorization": f"Bearer {key}"}
    ok = client.post(
        "/v1/ingest",
        headers=hdr,
        json={"documents": [{"doc_id": "d1", "source": "s", "text": "first document content"}]},
    )
    assert ok.status_code == 200
    blocked = client.post(
        "/v1/ingest",
        headers=hdr,
        json={"documents": [{"doc_id": "d2", "source": "s", "text": "second document content"}]},
    )
    assert blocked.status_code == 429
    assert blocked.json()["quota"] == "max_documents"


def test_query_quota_enforced(tmp_path: Path):
    client, _ = _client(tmp_path)
    key = _create_tenant(client, "rate", max_queries_per_day=2)
    hdr = {"Authorization": f"Bearer {key}"}
    for _ in range(2):
        assert client.post("/v1/query", headers=hdr, json={"question": "x"}).status_code == 200
    third = client.post("/v1/query", headers=hdr, json={"question": "x"})
    assert third.status_code == 429
    assert third.json()["quota"] == "max_queries_per_day"


def test_disabled_tenant_blocked(tmp_path: Path):
    client, _ = _client(tmp_path)
    key = _create_tenant(client, "acme")
    client.patch("/admin/tenants/acme", headers=ADMIN, json={"disabled": True})
    r = client.post(
        "/v1/query",
        headers={"Authorization": f"Bearer {key}"},
        json={"question": "hi"},
    )
    assert r.status_code == 403


def test_delete_tenant_purges_data(tmp_path: Path):
    client, engine = _client(tmp_path)
    key = _create_tenant(client, "acme")
    client.post(
        "/v1/ingest",
        headers={"Authorization": f"Bearer {key}"},
        json={"documents": [{"doc_id": "d", "source": "s", "text": "content to be purged"}]},
    )
    assert engine.vector_store.for_tenant("acme").ntotal >= 1
    r = client.delete("/admin/tenants/acme", headers=ADMIN)
    assert r.status_code == 200 and r.json()["purged"] is True
    assert engine.vector_store.for_tenant("acme").ntotal == 0
    assert engine.tenants.get("acme") is None


def test_per_tenant_prompt_template_stored(tmp_path: Path):
    client, engine = _client(tmp_path)
    _create_tenant(client, "acme", prompt_template="Answer like a pirate.")
    assert engine.tenants.get("acme").prompt_template == "Answer like a pirate."


def test_list_tenants(tmp_path: Path):
    client, _ = _client(tmp_path)
    _create_tenant(client, "a")
    _create_tenant(client, "b")
    r = client.get("/admin/tenants", headers=ADMIN)
    ids = {t["tenant_id"] for t in r.json()["tenants"]}
    assert {"a", "b"} <= ids
