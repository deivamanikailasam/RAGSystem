"""Typed, environment-driven configuration (12-factor).

All runtime knobs live here so nothing is hard-coded deep in the pipeline.
Values are read from environment variables / a local ``.env`` file. See
``.env.example`` for the full annotated list.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- OpenAI -----------------------------------------------------------
    openai_api_key: str | None = Field(default=None)
    embedding_model: str = "text-embedding-3-small"
    generation_model: str = "gpt-4.1-mini"
    openai_timeout_seconds: float = 30.0
    openai_max_retries: int = 5

    # --- Chunking ---------------------------------------------------------
    chunk_tokens: int = 400
    chunk_overlap: int = 60

    # --- Retrieval --------------------------------------------------------
    retrieval_top_k: int = 6
    rerank_enabled: bool = False
    min_score: float = 0.0

    # --- FAISS ------------------------------------------------------------
    faiss_index_type: str = "flat"  # "flat" | "ivf"
    ivf_nlist: int = 100
    ivf_nprobe: int = 8

    # --- Storage ----------------------------------------------------------
    data_dir: Path = Path("./data")

    # --- Auth -------------------------------------------------------------
    # Static "key:tenant" pairs for local/dev. Swap for a real IdP + secrets
    # manager in production (see docs/04-deployment-and-ops.md).
    api_keys: str = "demo-key:demo"

    # --- Generation guardrails -------------------------------------------
    max_context_chunks: int = 6
    generation_temperature: float = 0.1

    # --- Server -----------------------------------------------------------
    log_level: str = "INFO"

    @property
    def use_openai(self) -> bool:
        """Whether a real OpenAI key is configured.

        When ``False`` the system uses the deterministic local embedding
        provider and the extractive answerer, so it runs fully offline.
        """
        return bool(self.openai_api_key)

    @property
    def api_key_map(self) -> dict[str, str]:
        """Parse ``API_KEYS`` into ``{api_key: tenant_id}``."""
        mapping: dict[str, str] = {}
        for pair in self.api_keys.split(","):
            pair = pair.strip()
            if not pair:
                continue
            key, _, tenant = pair.partition(":")
            mapping[key.strip()] = (tenant or "default").strip()
        return mapping


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide cached Settings instance."""
    return Settings()
