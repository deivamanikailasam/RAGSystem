"""HTTP API tests using FastAPI's TestClient with an isolated engine."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.core.rag import RagEngine
from app.deps import get_engine
from app.main import create_app

AUTH = {"Authorization": "Bearer demo-key"}


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    settings = Settings(
        openai_api_key=None,
        data_dir=tmp_path / "data",
        chunk_tokens=40,
        chunk_overlap=8,
        api_keys="demo-key:demo",
    )
    app = create_app()
    # Inject an isolated engine so tests don't touch the default ./data dir.
    app.dependency_overrides[get_engine] = lambda: RagEngine(settings)
    return TestClient(app)


def test_health(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ingest_and_query_flow(client: TestClient):
    ingest = client.post(
        "/v1/ingest",
        headers=AUTH,
        json={
            "documents": [
                {
                    "doc_id": "faiss",
                    "source": "faiss.md",
                    "text": "FAISS performs similarity search over dense embedding "
                            "vectors. Documents are chunked and embedded first.",
                }
            ]
        },
    )
    assert ingest.status_code == 200
    assert ingest.json()["total_chunks"] >= 1

    query = client.post(
        "/v1/query",
        headers=AUTH,
        json={"question": "What does FAISS do?"},
    )
    assert query.status_code == 200
    body = query.json()
    assert body["citations"]
    assert body["citations"][0]["doc_id"] == "faiss"
    assert "request_id" in body


def test_query_requires_auth(client: TestClient):
    r = client.post("/v1/query", json={"question": "hi"})
    assert r.status_code == 401


def test_invalid_key_rejected(client: TestClient):
    r = client.post(
        "/v1/query",
        headers={"Authorization": "Bearer nope"},
        json={"question": "hi"},
    )
    assert r.status_code == 401


def test_delete_endpoint(client: TestClient):
    client.post(
        "/v1/ingest",
        headers=AUTH,
        json={"documents": [{"doc_id": "d1", "source": "s", "text": "some text here about topics"}]},
    )
    r = client.delete("/v1/documents/d1", headers=AUTH)
    assert r.status_code == 200
    assert r.json()["removed_vectors"] >= 1


def test_metrics_endpoint(client: TestClient):
    client.post(
        "/v1/ingest",
        headers=AUTH,
        json={"documents": [{"doc_id": "d", "source": "s", "text": "hello world content"}]},
    )
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "counters" in r.json()
