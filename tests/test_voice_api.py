"""HTTP tests for the /v1/voice session state-machine endpoints."""

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
    engine.ingest(tenant="demo", text="FAISS indexes dense vectors; nprobe tunes IVF latency.",
                  source="faiss.md", doc_id="faiss")
    app = create_app()
    app.dependency_overrides[get_engine] = lambda: engine
    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app)


def _create(client: TestClient) -> str:
    r = client.post("/v1/voice/sessions", headers=AUTH, json={})
    assert r.status_code == 200
    assert r.json()["state"] == "idle"
    return r.json()["session_id"]


def _event(client, sid, event, **kw):
    return client.post(f"/v1/voice/sessions/{sid}/events", headers=AUTH,
                       json={"event": event, **kw})


def test_full_voice_flow(client: TestClient):
    sid = _create(client)
    assert _event(client, sid, "start").json()["state"] == "listening"

    spoke = _event(client, sid, "transcript", text="What is FAISS?")
    body = spoke.json()
    assert body["state"] == "speaking"
    assert body["say"]
    assert body["citations"]
    assert body["standalone_question"]

    assert _event(client, sid, "speak_done").json()["state"] == "listening"


def test_illegal_transition_returns_409_with_allowed(client: TestClient):
    sid = _create(client)  # idle
    r = _event(client, sid, "transcript", text="hi")  # can't transcribe from idle
    assert r.status_code == 409
    body = r.json()
    assert body["state"] == "idle"
    assert "start" in body["allowed_events"]


def test_unknown_event_422(client: TestClient):
    sid = _create(client)
    assert _event(client, sid, "levitate").status_code == 422


def test_get_session_reports_state_and_allowed(client: TestClient):
    sid = _create(client)
    _event(client, sid, "start")
    r = client.get(f"/v1/voice/sessions/{sid}", headers=AUTH)
    assert r.status_code == 200
    assert r.json()["state"] == "listening"
    assert "transcript" in r.json()["allowed_events"]


def test_unknown_session_404(client: TestClient):
    assert client.get("/v1/voice/sessions/nope", headers=AUTH).status_code == 404


def test_delete_session(client: TestClient):
    sid = _create(client)
    assert client.delete(f"/v1/voice/sessions/{sid}", headers=AUTH).json()["deleted"] is True
    assert client.get(f"/v1/voice/sessions/{sid}", headers=AUTH).status_code == 404


def test_barge_in_over_http(client: TestClient):
    sid = _create(client)
    _event(client, sid, "start")
    _event(client, sid, "transcript", text="What is FAISS?")  # -> speaking
    r = _event(client, sid, "barge_in")
    assert r.json()["state"] == "listening"
    assert r.json()["barge_in_count"] == 1


def test_voice_requires_auth(client: TestClient):
    assert client.post("/v1/voice/sessions", json={}).status_code == 401
