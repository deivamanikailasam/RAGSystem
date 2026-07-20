"""HTTP tests for the FAQ bot + memory endpoints."""

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
    engine.ingest(tenant="demo", text="FAISS indexes dense vectors for similarity search.",
                  source="faiss.md", doc_id="faiss")
    app = create_app()
    app.dependency_overrides[get_engine] = lambda: engine
    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app)


def test_faq_crud(client: TestClient):
    add = client.post("/v1/faqs", headers=AUTH, json={
        "question": "How do I reset my password?",
        "answer": "Go to Settings > Security.", "tags": ["auth"]})
    assert add.status_code == 200
    fid = add.json()["faq_id"]
    assert any(f["faq_id"] == fid for f in client.get("/v1/faqs", headers=AUTH).json()["faqs"])
    assert client.delete(f"/v1/faqs/{fid}", headers=AUTH).json()["deleted"] is True


def test_ask_faq_hit(client: TestClient):
    client.post("/v1/faqs", headers=AUTH, json={
        "question": "How do I reset my password?", "answer": "Go to Settings."})
    r = client.post("/v1/faq/ask", headers=AUTH,
                    json={"message": "how can I reset my password?"})
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "faq"
    assert body["answer"] == "Go to Settings."


def test_ask_rag_fallback(client: TestClient):
    r = client.post("/v1/faq/ask", headers=AUTH,
                    json={"message": "How does FAISS index vectors?"})
    assert r.json()["source"] == "rag"
    assert r.json()["citations"][0]["type"] == "doc"


def test_memory_endpoints(client: TestClient):
    client.post("/v1/faq/ask", headers=AUTH,
                json={"message": "my name is Ada", "user_id": "u1"})
    mem = client.get("/v1/memory/u1", headers=AUTH)
    assert mem.status_code == 200
    assert any(m["content"] == "name is Ada" for m in mem.json()["memories"])
    # Forget clears it.
    assert client.delete("/v1/memory/u1", headers=AUTH).json()["forgotten"] >= 1
    assert client.get("/v1/memory/u1", headers=AUTH).json()["memories"] == []


def test_memory_recalled_across_sessions(client: TestClient):
    client.post("/v1/faq/ask", headers=AUTH,
                json={"message": "my name is Ada", "user_id": "u1", "session_id": "s1"})
    r = client.post("/v1/faq/ask", headers=AUTH, json={
        "message": "How does FAISS work?", "user_id": "u1", "session_id": "s2"})
    assert "name is Ada" in r.json()["memories_used"]


def test_faq_requires_auth(client: TestClient):
    assert client.post("/v1/faq/ask", json={"message": "hi"}).status_code == 401
