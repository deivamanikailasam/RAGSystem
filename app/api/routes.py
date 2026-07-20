"""HTTP endpoints: /health, /metrics, /v1/ingest, /v1/query, /v1/documents/{id}.

Document parsing for uploaded files lives here (``_extract_text``) so the core
pipeline stays format-agnostic — it only ever sees normalized text + metadata.
"""

from __future__ import annotations

import io

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app import __version__
from app.core.rag import RagEngine
from app.deps import get_engine, require_tenant
from app.models import (
    DeleteResponse,
    HealthResponse,
    IngestedDoc,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from app.observability.metrics import METRICS

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health(engine: RagEngine = Depends(get_engine)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=__version__,
        openai_enabled=engine._settings.use_openai,  # noqa: SLF001 (read-only)
    )


@router.get("/metrics", tags=["ops"])
def metrics() -> dict[str, object]:
    return METRICS.snapshot()


@router.post("/v1/ingest", response_model=IngestResponse, tags=["ingest"])
def ingest(
    body: IngestRequest,
    tenant: str = Depends(require_tenant),
    engine: RagEngine = Depends(get_engine),
) -> IngestResponse:
    results: list[IngestedDoc] = []
    total = 0
    for doc in body.documents:
        res = engine.ingest(
            tenant=tenant,
            text=doc.text,
            source=doc.source,
            doc_id=doc.doc_id,
            metadata=doc.metadata,
        )
        results.append(
            IngestedDoc(doc_id=res.doc_id, chunks=res.chunks, version=res.version)
        )
        total += res.chunks
    return IngestResponse(tenant=tenant, documents=results, total_chunks=total)


@router.post("/v1/ingest/file", response_model=IngestResponse, tags=["ingest"])
async def ingest_file(
    file: UploadFile = File(...),
    source: str | None = Form(default=None),
    tenant: str = Depends(require_tenant),
    engine: RagEngine = Depends(get_engine),
) -> IngestResponse:
    raw = await file.read()
    text = _extract_text(file.filename or "upload", raw)
    if not text.strip():
        raise HTTPException(status_code=422, detail="No extractable text in file.")
    res = engine.ingest(
        tenant=tenant, text=text, source=source or (file.filename or "upload")
    )
    return IngestResponse(
        tenant=tenant,
        documents=[IngestedDoc(doc_id=res.doc_id, chunks=res.chunks, version=res.version)],
        total_chunks=res.chunks,
    )


@router.post("/v1/query", response_model=QueryResponse, tags=["query"])
def query(
    body: QueryRequest,
    tenant: str = Depends(require_tenant),
    engine: RagEngine = Depends(get_engine),
) -> QueryResponse:
    return engine.query(
        tenant=tenant,
        question=body.question,
        top_k=body.top_k,
        filters=body.filters,
    )


@router.delete(
    "/v1/documents/{doc_id}", response_model=DeleteResponse, tags=["ingest"]
)
def delete_document(
    doc_id: str,
    tenant: str = Depends(require_tenant),
    engine: RagEngine = Depends(get_engine),
) -> DeleteResponse:
    removed = engine.delete_document(tenant=tenant, doc_id=doc_id)
    return DeleteResponse(doc_id=doc_id, removed_vectors=removed)


# --------------------------------------------------------------------------- #
# File text extraction
# --------------------------------------------------------------------------- #
def _extract_text(filename: str, raw: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        return _extract_pdf(raw)
    # txt / md / html / csv / json → decode as text. HTML tags are left in; a
    # production system would strip them (e.g. selectolax / BeautifulSoup).
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="ignore")


def _extract_pdf(raw: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise HTTPException(500, "pypdf not installed for PDF ingestion.") from exc
    reader = PdfReader(io.BytesIO(raw))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)
