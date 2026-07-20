"""Pydantic request/response schemas for the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #
class IngestDocument(BaseModel):
    """A single document supplied inline as text."""

    doc_id: str | None = Field(
        default=None,
        description="Stable identifier. If omitted, derived from the content hash.",
    )
    source: str = Field(default="inline", description="Where the doc came from.")
    text: str = Field(..., description="Raw document text.")
    metadata: dict[str, str] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    documents: list[IngestDocument]


class IngestedDoc(BaseModel):
    doc_id: str
    chunks: int
    version: int


class IngestResponse(BaseModel):
    tenant: str
    documents: list[IngestedDoc]
    total_chunks: int


# --------------------------------------------------------------------------- #
# Query
# --------------------------------------------------------------------------- #
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = Field(
        default=None, description="Override the default retrieval depth."
    )
    filters: dict[str, str] = Field(
        default_factory=dict,
        description="Metadata equality filters, e.g. {'doc_type': 'policy'}.",
    )
    stream: bool = False


class Citation(BaseModel):
    doc_id: str
    source: str
    chunk_index: int
    score: float = Field(..., description="Raw vector similarity (stage 1).")
    rerank_score: float | None = Field(
        default=None, description="Reranker relevance score (stage 2), if reranked."
    )
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    model: str
    reranker: str = Field(default="none", description="Reranking strategy used.")
    retrieval_ms: float
    rerank_ms: float = 0.0
    generation_ms: float
    tokens: dict[str, int] = Field(default_factory=dict)
    request_id: str


# --------------------------------------------------------------------------- #
# Misc
# --------------------------------------------------------------------------- #
class DeleteResponse(BaseModel):
    doc_id: str
    removed_vectors: int


class HealthResponse(BaseModel):
    status: str
    version: str
    openai_enabled: bool
    deployment_mode: str


# --------------------------------------------------------------------------- #
# Tenant control plane (multi-tenant mode)
# --------------------------------------------------------------------------- #
class TenantCreate(BaseModel):
    tenant_id: str = Field(..., min_length=1, pattern=r"^[a-zA-Z0-9_-]+$")
    name: str | None = None
    api_key: str | None = Field(
        default=None,
        description="Optional key to assign. If omitted, one is generated and "
        "returned exactly once.",
    )
    prompt_template: str | None = None
    index_type: str | None = Field(default=None, description="'flat' or 'ivf'.")
    max_documents: int = Field(default=0, ge=0, description="0 = unlimited.")
    max_queries_per_day: int = Field(default=0, ge=0, description="0 = unlimited.")


class TenantUpdate(BaseModel):
    name: str | None = None
    prompt_template: str | None = None
    index_type: str | None = None
    max_documents: int | None = Field(default=None, ge=0)
    max_queries_per_day: int | None = Field(default=None, ge=0)
    disabled: bool | None = None


class TenantInfo(BaseModel):
    tenant_id: str
    name: str
    prompt_template: str | None
    index_type: str
    max_documents: int
    max_queries_per_day: int
    disabled: bool
    created_at: float


class TenantCreateResponse(BaseModel):
    tenant: TenantInfo
    api_key: str | None = Field(
        default=None,
        description="Shown ONCE. Store it now — only its hash is persisted.",
    )


class TenantListResponse(BaseModel):
    tenants: list[TenantInfo]


class TenantStats(BaseModel):
    tenant: str
    mode: str
    documents: int
    vectors: int
    queries_today: int
    quotas: dict[str, int]
