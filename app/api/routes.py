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
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ConversationResponse,
    DeleteConversationResponse,
    DeleteResponse,
    DialogueRequest,
    DialogueResponse,
    DialogueStateResponse,
    IntentEventModel,
    HealthResponse,
    IngestedDoc,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    TenantStats,
    VoiceEventRequest,
    VoiceEventResponse,
    VoiceSessionCreate,
    VoiceSessionResponse,
)
from app.core.voice_fsm import Event, InvalidTransition
from app.core.voice_session import SessionNotFound
from app.observability.metrics import METRICS

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health(engine: RagEngine = Depends(get_engine)) -> HealthResponse:
    settings = engine._settings  # noqa: SLF001 (read-only)
    return HealthResponse(
        status="ok",
        version=__version__,
        openai_enabled=settings.use_openai,
        deployment_mode=settings.deployment_mode,
    )


@router.get("/v1/me", response_model=TenantStats, tags=["ops"])
def whoami(
    tenant: str = Depends(require_tenant),
    engine: RagEngine = Depends(get_engine),
) -> TenantStats:
    """Return the calling tenant's id, mode, usage, and quotas."""
    return TenantStats(**engine.tenant_stats(tenant))


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


@router.post("/v1/chat", response_model=ChatResponse, tags=["chat"])
def chat(
    body: ChatRequest,
    tenant: str = Depends(require_tenant),
    engine: RagEngine = Depends(get_engine),
) -> ChatResponse:
    """Send one conversational turn; continues a session or starts a new one."""
    return engine.chat(
        tenant=tenant,
        message=body.message,
        session_id=body.session_id,
        top_k=body.top_k,
        filters=body.filters,
    )


@router.get(
    "/v1/chat/{session_id}", response_model=ConversationResponse, tags=["chat"]
)
def get_conversation(
    session_id: str,
    tenant: str = Depends(require_tenant),
    engine: RagEngine = Depends(get_engine),
) -> ConversationResponse:
    messages = engine.get_conversation(tenant=tenant, session_id=session_id)
    if not messages:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return ConversationResponse(
        session_id=session_id,
        tenant=tenant,
        messages=[
            ChatMessage(
                turn_index=m.turn_index,
                role=m.role,
                content=m.content,
                citations=m.citations,
                created_at=m.created_at,
            )
            for m in messages
        ],
    )


@router.delete(
    "/v1/chat/{session_id}",
    response_model=DeleteConversationResponse,
    tags=["chat"],
)
def delete_conversation(
    session_id: str,
    tenant: str = Depends(require_tenant),
    engine: RagEngine = Depends(get_engine),
) -> DeleteConversationResponse:
    deleted = engine.delete_conversation(tenant=tenant, session_id=session_id)
    return DeleteConversationResponse(session_id=session_id, deleted_messages=deleted)


@router.post("/v1/dialogue", response_model=DialogueResponse, tags=["dialogue"])
def dialogue(
    body: DialogueRequest,
    tenant: str = Depends(require_tenant),
    engine: RagEngine = Depends(get_engine),
) -> DialogueResponse:
    """Classify the turn's intent, act on it (RAG answer or canned reply),
    and persist the intent + dialogue state."""
    turn = engine.dialogue.handle(
        tenant=tenant, message=body.message, session_id=body.session_id,
        top_k=body.top_k, filters=body.filters,
    )
    return DialogueResponse(
        session_id=turn.session_id,
        turn_index=turn.turn_index,
        intent=turn.intent.value,
        confidence=turn.confidence,
        action=turn.action.value,
        slots=turn.slots,
        answer=turn.answer,
        classifier=turn.classifier,
        citations=turn.citations,
        standalone_question=turn.standalone_question,
        model=turn.model,
    )


@router.get(
    "/v1/dialogue/{session_id}",
    response_model=DialogueStateResponse,
    tags=["dialogue"],
)
def get_dialogue(
    session_id: str,
    tenant: str = Depends(require_tenant),
    engine: RagEngine = Depends(get_engine),
) -> DialogueStateResponse:
    """Return the dialogue state and the persisted intent history."""
    state = engine.dialogue.get_state(tenant, session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Dialogue not found.")
    intents = engine.dialogue.get_intents(tenant, session_id)
    return DialogueStateResponse(
        session_id=session_id,
        tenant=tenant,
        current_intent=state.current_intent,
        turn_count=state.turn_count,
        slots=state.slots,
        intents=[
            IntentEventModel(
                turn_index=e.turn_index, message=e.message, intent=e.intent,
                confidence=e.confidence, action=e.action, slots=e.slots,
                created_at=e.created_at,
            )
            for e in intents
        ],
        created_at=state.created_at,
        updated_at=state.updated_at,
    )


@router.delete("/v1/dialogue/{session_id}", tags=["dialogue"])
def delete_dialogue(
    session_id: str,
    tenant: str = Depends(require_tenant),
    engine: RagEngine = Depends(get_engine),
) -> dict[str, object]:
    deleted = engine.dialogue.delete_session(tenant, session_id)
    return {"session_id": session_id, "deleted_events": deleted}


@router.post("/v1/voice/sessions", response_model=VoiceSessionResponse, tags=["voice"])
def create_voice_session(
    body: VoiceSessionCreate | None = None,
    tenant: str = Depends(require_tenant),
    engine: RagEngine = Depends(get_engine),
) -> VoiceSessionResponse:
    """Create a voice session (starts in state 'idle')."""
    session_id = body.session_id if body else None
    session = engine.voice.create_session(tenant, session_id)
    return _voice_session_response(session, engine)


@router.post(
    "/v1/voice/sessions/{session_id}/events",
    response_model=VoiceEventResponse,
    tags=["voice"],
)
def voice_event(
    session_id: str,
    body: VoiceEventRequest,
    tenant: str = Depends(require_tenant),
    engine: RagEngine = Depends(get_engine),
) -> VoiceEventResponse:
    """Drive the session state machine with an event.

    A 'transcript' event runs a grounded RAG turn and returns the reply to
    speak. Illegal transitions return 409 with the events allowed here.
    """
    try:
        event = Event(body.event)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Unknown event '{body.event}'.") from exc

    result = engine.voice.send_event(
        tenant=tenant, session_id=session_id, event=event,
        text=body.text, message=body.message, top_k=body.top_k, filters=body.filters,
    )
    return VoiceEventResponse(
        session_id=result.session_id,
        event=result.event.value,
        previous_state=result.previous_state.value,
        state=result.state.value,
        allowed_events=[e.value for e in result.allowed_events],
        turn_count=result.turn_count,
        barge_in_count=result.barge_in_count,
        say=result.say,
        citations=result.citations,
        standalone_question=result.standalone_question,
    )


@router.get(
    "/v1/voice/sessions/{session_id}",
    response_model=VoiceSessionResponse,
    tags=["voice"],
)
def get_voice_session(
    session_id: str,
    tenant: str = Depends(require_tenant),
    engine: RagEngine = Depends(get_engine),
) -> VoiceSessionResponse:
    session = engine.voice.get_session(tenant, session_id)
    return _voice_session_response(session, engine)


@router.delete("/v1/voice/sessions/{session_id}", tags=["voice"])
def delete_voice_session(
    session_id: str,
    tenant: str = Depends(require_tenant),
    engine: RagEngine = Depends(get_engine),
) -> dict[str, object]:
    deleted = engine.voice.delete_session(tenant, session_id)
    return {"session_id": session_id, "deleted": deleted}


def _voice_session_response(session, engine: RagEngine) -> VoiceSessionResponse:
    return VoiceSessionResponse(
        session_id=session.session_id,
        tenant=session.tenant,
        state=session.state.value,
        allowed_events=[e.value for e in engine.voice.allowed_events(session.state)],
        last_transcript=session.last_transcript,
        last_response=session.last_response,
        turn_count=session.turn_count,
        barge_in_count=session.barge_in_count,
        error=session.error,
        created_at=session.created_at,
        updated_at=session.updated_at,
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
