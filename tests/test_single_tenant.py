"""Single-tenant deployment mode tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.core.rag import RagEngine
from app.deps import get_engine
from app.main import create_app


def _client(tmp_path: Path, require_auth: bool = False) -> TestClient:
    settings = Settings(
        deployment_mode="single_tenant",
        single_tenant_id="internal",
        single_tenant_require_auth=require_auth,
        openai_api_key=None,
        data_dir=tmp_path / "data",
        chunk_tokens=40,
        chunk_overlap=8,
        api_keys="the-key:internal",
    )
    app = create_app()
    app.dependency_overrides[get_engine] = lambda: RagEngine(settings)
    # get_settings is used by deps too; override it for consistency.
    from app.config import get_settings

    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app)


def test_no_auth_required_by_default(tmp_path: Path):
    client = _client(tmp_path, require_auth=False)
    # No Authorization header at all.
    r = client.post(
        "/v1/ingest",
        json={"documents": [{"doc_id": "d", "source": "s", "text": "FAISS indexes vectors for search."}]},
    )
    assert r.status_code == 200
    q = client.post("/v1/query", json={"question": "What does FAISS do?"})
    assert q.status_code == 200
    assert q.json()["citations"]


def test_health_reports_single_tenant(tmp_path: Path):
    client = _client(tmp_path)
    body = client.get("/health").json()
    assert body["deployment_mode"] == "single_tenant"


def test_me_reports_fixed_tenant(tmp_path: Path):
    client = _client(tmp_path)
    body = client.get("/v1/me").json()
    assert body["tenant"] == "internal"
    assert body["mode"] == "single_tenant"
    # No quotas in single-tenant mode.
    assert body["quotas"]["max_documents"] == 0


def test_auth_enforced_when_configured(tmp_path: Path):
    client = _client(tmp_path, require_auth=True)
    # Missing token -> 401.
    assert client.post("/v1/query", json={"question": "hi"}).status_code == 401
    # Wrong token -> 401.
    r = client.post(
        "/v1/query",
        headers={"Authorization": "Bearer nope"},
        json={"question": "hi"},
    )
    assert r.status_code == 401
    # Correct token -> ok.
    ok = client.post(
        "/v1/query",
        headers={"Authorization": "Bearer the-key"},
        json={"question": "hi"},
    )
    assert ok.status_code == 200


def test_admin_api_unavailable_in_single_tenant(tmp_path: Path):
    client = _client(tmp_path)
    r = client.get("/admin/tenants", headers={"Authorization": "Bearer whatever"})
    assert r.status_code == 404
