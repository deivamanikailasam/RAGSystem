"""HTTP tests for the /v1/dialogue endpoints."""

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
    engine.ingest(tenant="demo", text="FAISS indexes dense vectors for search.",
                  source="faiss.md", doc_id="faiss")
    app = create_app()
    app.dependency_overrides[get_engine] = lambda: engine
    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app)


def _say(client, text, sid=None):
    body = {"message": text}
    if sid:
        body["session_id"] = sid
    return client.post("/v1/dialogue", headers=AUTH, json=body)


def test_question_and_greeting_routing(client: TestClient):
    q = _say(client, "What is FAISS?").json()
    assert q["intent"] == "question" and q["action"] == "answer"
    assert q["citations"]

    g = _say(client, "hello", sid=q["session_id"]).json()
    assert g["intent"] == "greeting" and g["action"] == "greet"
    assert g["citations"] == []


def test_dialogue_state_endpoint_returns_intent_history(client: TestClient):
    sid = _say(client, "hi").json()["session_id"]
    _say(client, "What is FAISS?", sid=sid)
    _say(client, "bye", sid=sid)

    r = client.get(f"/v1/dialogue/{sid}", headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["turn_count"] == 3
    assert body["current_intent"] == "goodbye"
    assert [e["intent"] for e in body["intents"]] == ["greeting", "question", "goodbye"]


def test_unknown_dialogue_404(client: TestClient):
    assert client.get("/v1/dialogue/nope", headers=AUTH).status_code == 404


def test_delete_dialogue(client: TestClient):
    sid = _say(client, "hi").json()["session_id"]
    d = client.delete(f"/v1/dialogue/{sid}", headers=AUTH)
    assert d.status_code == 200 and d.json()["deleted_events"] >= 1
    assert client.get(f"/v1/dialogue/{sid}", headers=AUTH).status_code == 404


def test_dialogue_requires_auth(client: TestClient):
    assert client.post("/v1/dialogue", json={"message": "hi"}).status_code == 401
