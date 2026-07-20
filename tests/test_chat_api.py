"""HTTP tests for the /v1/chat endpoints."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.core.rag import RagEngine
from app.deps import get_engine
from app.main import create_app

AUTH = {"Authorization": "Bearer demo-key"}


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    settings = Settings(
        deployment_mode="multi_tenant",
        openai_api_key=None,
        data_dir=tmp_path / "data",
        api_keys="demo-key:demo",
    )
    engine = RagEngine(settings)
    engine.ingest(
        tenant="demo",
        text="FAISS indexes dense vectors; its IVF index cuts latency via nprobe.",
        source="faiss.md",
        doc_id="faiss",
    )
    app = create_app()
    app.dependency_overrides[get_engine] = lambda: engine
    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app)


def test_chat_starts_and_continues_session(client: TestClient):
    r1 = client.post("/v1/chat", headers=AUTH, json={"message": "What is FAISS?"})
    assert r1.status_code == 200
    sid = r1.json()["session_id"]
    assert r1.json()["citations"]

    r2 = client.post(
        "/v1/chat", headers=AUTH,
        json={"message": "what about its latency?", "session_id": sid},
    )
    assert r2.status_code == 200
    assert r2.json()["session_id"] == sid
    assert "standalone_question" in r2.json()


def test_get_conversation(client: TestClient):
    sid = client.post("/v1/chat", headers=AUTH,
                      json={"message": "What is FAISS?"}).json()["session_id"]
    r = client.get(f"/v1/chat/{sid}", headers=AUTH)
    assert r.status_code == 200
    roles = [m["role"] for m in r.json()["messages"]]
    assert roles == ["user", "assistant"]


def test_get_unknown_conversation_404(client: TestClient):
    r = client.get("/v1/chat/does-not-exist", headers=AUTH)
    assert r.status_code == 404


def test_delete_conversation(client: TestClient):
    sid = client.post("/v1/chat", headers=AUTH,
                      json={"message": "What is FAISS?"}).json()["session_id"]
    d = client.delete(f"/v1/chat/{sid}", headers=AUTH)
    assert d.status_code == 200 and d.json()["deleted_messages"] >= 2
    assert client.get(f"/v1/chat/{sid}", headers=AUTH).status_code == 404


def test_chat_requires_auth(client: TestClient):
    assert client.post("/v1/chat", json={"message": "hi"}).status_code == 401
