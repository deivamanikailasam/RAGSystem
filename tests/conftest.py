"""Shared pytest fixtures.

All tests run against the offline local-fallback provider, so no OpenAI key or
network access is required. Each test gets an isolated temp ``DATA_DIR`` so
FAISS indices and the SQLite docstore never leak between tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.core.rag import RagEngine


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        openai_api_key=None,
        data_dir=tmp_path / "data",
        chunk_tokens=40,
        chunk_overlap=8,
        retrieval_top_k=4,
        api_keys="demo-key:demo",
    )


@pytest.fixture
def engine(settings: Settings) -> RagEngine:
    return RagEngine(settings)
