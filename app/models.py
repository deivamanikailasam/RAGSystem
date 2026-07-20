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
    score: float
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    model: str
    retrieval_ms: float
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
