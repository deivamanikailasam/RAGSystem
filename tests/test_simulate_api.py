"""HTTP tests for the /v1/simulate endpoint."""

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
    settings = Settings(deployment_mode="multi_tenant", openai_api_key=None,
                        data_dir=tmp_path / "data", api_keys="demo-key:demo")
    engine = RagEngine(settings)
    engine.ingest(tenant="demo", doc_id="faiss", source="faiss.md",
                  text="FAISS reduces latency using IVF and nprobe over dense vectors.")
    app = create_app()
    app.dependency_overrides[get_engine] = lambda: engine
    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app)


def test_simulate_dialogue_flow(client: TestClient):
    r = client.post("/v1/simulate", headers=AUTH, json={
        "channel": "dialogue", "name": "d1",
        "turns": ["hi", "How does FAISS reduce latency?", "bye"]})
    assert r.status_code == 200
    body = r.json()
    assert body["states"] == ["greeting", "question", "goodbye"]
    assert body["diagrams"]["sequence"].startswith("sequenceDiagram")
    assert body["diagrams"]["path"].startswith("flowchart")


def test_simulate_voice_flow(client: TestClient):
    r = client.post("/v1/simulate", headers=AUTH, json={
        "channel": "voice", "name": "v1", "turns": [
            {"event": "start"},
            {"event": "transcript", "text": "How does FAISS reduce latency?"},
            {"event": "speak_done"},
            {"event": "end"},
        ]})
    assert r.status_code == 200
    body = r.json()
    assert body["states"] == ["idle", "listening", "speaking", "listening", "ended"]
    assert body["diagrams"]["path"].startswith("stateDiagram-v2")


def test_simulate_unknown_channel_422(client: TestClient):
    r = client.post("/v1/simulate", headers=AUTH,
                    json={"channel": "telepathy", "turns": []})
    assert r.status_code == 422


def test_simulate_requires_auth(client: TestClient):
    assert client.post("/v1/simulate",
                       json={"channel": "dialogue", "turns": ["hi"]}).status_code == 401
