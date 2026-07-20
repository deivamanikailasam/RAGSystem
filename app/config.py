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
    min_score: float = 0.0

    # Retrieval mode (stage 1; see docs/09-hybrid-retrieval.md):
    #   "vector" — dense FAISS only
    #   "bm25"   — sparse BM25 only
    #   "hybrid" — run both and fuse (default)
    retrieval_mode: str = "hybrid"
    # Fusion for hybrid mode: "rrf" (rank-based, scale-free) or "weighted".
    hybrid_fusion: str = "rrf"
    rrf_k: int = 60          # RRF damping constant
    hybrid_alpha: float = 0.5  # dense weight in weighted fusion (1-alpha = sparse)

    # --- Reranking (second stage; see docs/08-reranking.md) ---------------
    # Strategy: "none" | "lexical" | "cross_encoder" | "llm".
    rerank_strategy: str = "lexical"
    # Candidates pulled from vector search before reranking down to top_k.
    rerank_candidates: int = 20
    # Cross-encoder model (requires requirements-rerank.txt).
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # LLM reranker model (defaults to GENERATION_MODEL when unset).
    rerank_llm_model: str | None = None
    # Deprecated alias kept for backward compatibility: if the strategy is left
    # at "none" but this is true, the lexical reranker is used.
    rerank_enabled: bool = False

    # --- FAISS ------------------------------------------------------------
    faiss_index_type: str = "flat"  # "flat" | "ivf"
    ivf_nlist: int = 100
    ivf_nprobe: int = 8

    # --- Storage ----------------------------------------------------------
    data_dir: Path = Path("./data")

    # --- Deployment mode --------------------------------------------------
    # "single_tenant" — one implicit corpus, minimal/optional auth. Ideal for
    #                   an internal doc bot on a limited corpus.
    # "multi_tenant"  — a platform: per-tenant FAISS indices, a tenant registry,
    #                   an admin control plane, per-tenant config + quotas.
    # See docs/07-deployment-modes.md.
    deployment_mode: str = "multi_tenant"

    # Single-tenant knobs (ignored in multi-tenant mode).
    single_tenant_id: str = "default"
    single_tenant_require_auth: bool = False

    # Multi-tenant control plane: the admin key guards /admin/* tenant
    # management. Leave unset to disable the admin API entirely.
    admin_api_key: str | None = None

    # Tenant isolation strategy (multi-tenant; see docs/10-tenant-isolation.md):
    #   "index_per_tenant" — one FAISS index file per tenant (physical, default)
    #   "shared_namespace" — one shared FAISS index partitioned by tenant id,
    #                        queried with an exact per-namespace id selector
    tenant_isolation: str = "index_per_tenant"

    # --- Auth -------------------------------------------------------------
    # Static "key:tenant" pairs for local/dev. Swap for a real IdP + secrets
    # manager in production (see docs/04-deployment-and-ops.md).
    api_keys: str = "demo-key:demo"

    # --- Generation guardrails -------------------------------------------
    max_context_chunks: int = 6
    generation_temperature: float = 0.1

    # --- Multi-turn chat (see docs/12-multi-turn-chat.md) -----------------
    # How many prior messages to feed the generator as conversation context.
    chat_history_turns: int = 8
    # Rewrite follow-up questions into standalone queries before retrieval.
    chat_condense_question: bool = True

    # --- Server -----------------------------------------------------------
    log_level: str = "INFO"

    @property
    def is_single_tenant(self) -> bool:
        return self.deployment_mode == "single_tenant"

    @property
    def effective_rerank_strategy(self) -> str:
        """Resolve the rerank strategy, honoring the legacy ``rerank_enabled``."""
        if self.rerank_strategy == "none" and self.rerank_enabled:
            return "lexical"
        return self.rerank_strategy

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
