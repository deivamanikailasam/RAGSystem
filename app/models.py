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
    score: float = Field(..., description="Stage-1 primary score (fused in hybrid).")
    vector_score: float | None = Field(
        default=None, description="Dense cosine similarity, if vector search ran."
    )
    bm25_score: float | None = Field(
        default=None, description="Sparse BM25 score, if BM25 search ran."
    )
    rerank_score: float | None = Field(
        default=None, description="Reranker relevance score (stage 2), if reranked."
    )
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    model: str
    retrieval_mode: str = Field(
        default="vector", description="Stage-1 mode: vector | bm25 | hybrid."
    )
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


# --------------------------------------------------------------------------- #
# Multi-turn chat
# --------------------------------------------------------------------------- #
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str | None = Field(
        default=None,
        description="Conversation to continue. Omit to start a new session; "
        "the created id is returned in the response.",
    )
    top_k: int | None = None
    filters: dict[str, str] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    session_id: str
    turn_index: int
    answer: str
    citations: list[Citation]
    standalone_question: str = Field(
        ..., description="The follow-up rewritten for retrieval (may equal the input)."
    )
    model: str
    retrieval_mode: str
    reranker: str
    condenser: str
    retrieval_ms: float
    generation_ms: float
    tokens: dict[str, int] = Field(default_factory=dict)
    request_id: str


class ChatMessage(BaseModel):
    turn_index: int
    role: str
    content: str
    citations: list[dict] = Field(default_factory=list)
    created_at: float


class ConversationResponse(BaseModel):
    session_id: str
    tenant: str
    messages: list[ChatMessage]


class DeleteConversationResponse(BaseModel):
    session_id: str
    deleted_messages: int


# --------------------------------------------------------------------------- #
# Voice assistant session state machine
# --------------------------------------------------------------------------- #
class VoiceSessionCreate(BaseModel):
    session_id: str | None = Field(
        default=None, description="Optional client-supplied id; generated if omitted."
    )


class VoiceEventRequest(BaseModel):
    event: str = Field(
        ...,
        description="FSM event: start | transcript | speak_done | barge_in | "
        "silence_timeout | end | error | recover | wake.",
    )
    text: str | None = Field(
        default=None, description="User utterance transcript (for 'transcript')."
    )
    message: str | None = Field(default=None, description="Detail for 'error'.")
    top_k: int | None = None
    filters: dict[str, str] = Field(default_factory=dict)


class VoiceEventResponse(BaseModel):
    session_id: str
    event: str
    previous_state: str
    state: str
    allowed_events: list[str]
    turn_count: int
    barge_in_count: int
    say: str | None = None
    citations: list[dict] | None = None
    standalone_question: str | None = None


class VoiceSessionResponse(BaseModel):
    session_id: str
    tenant: str
    state: str
    allowed_events: list[str]
    last_transcript: str | None
    last_response: str | None
    turn_count: int
    barge_in_count: int
    error: str | None
    created_at: float
    updated_at: float


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
