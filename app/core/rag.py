"""RagEngine — composes every component into one process-wide service.

This is the object the API layer talks to. It owns the embedding provider,
FAISS vector store, SQLite docstore, ingestion pipeline, retriever, and
generator, and exposes three high-level operations: ``ingest``, ``query``,
and ``delete_document``.
"""

from __future__ import annotations

import time
import uuid

from app.config import Settings
from app.core.docstore import DocStore
from app.core.embeddings import build_embedding_provider
from app.core.generator import build_generator
from app.core.ingest import IngestedDocResult, IngestionPipeline
from app.core.retriever import Retriever
from app.core.vector_store import VectorStore
from app.models import Citation, QueryResponse
from app.observability.metrics import METRICS


class RagEngine:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.embeddings = build_embedding_provider(settings)
        self.vector_store = VectorStore(
            data_dir=settings.data_dir,
            dimension=self.embeddings.dimension,
            index_type=settings.faiss_index_type,
            nlist=settings.ivf_nlist,
            nprobe=settings.ivf_nprobe,
        )
        self.docstore = DocStore(settings.data_dir / "docstore.db")
        self.ingestion = IngestionPipeline(
            settings, self.embeddings, self.vector_store, self.docstore
        )
        self.retriever = Retriever(
            settings, self.embeddings, self.vector_store, self.docstore
        )
        self.generator = build_generator(settings)

    # -- operations -------------------------------------------------------- #
    def ingest(
        self,
        *,
        tenant: str,
        text: str,
        source: str,
        doc_id: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> IngestedDocResult:
        with METRICS.timer("ingest_ms"):
            result = self.ingestion.ingest_document(
                tenant=tenant, text=text, source=source,
                doc_id=doc_id, metadata=metadata,
            )
        METRICS.increment("documents_ingested")
        METRICS.increment("chunks_ingested", result.chunks)
        return result

    def query(
        self,
        *,
        tenant: str,
        question: str,
        top_k: int | None = None,
        filters: dict[str, str] | None = None,
    ) -> QueryResponse:
        request_id = uuid.uuid4().hex
        METRICS.increment("queries")

        t0 = time.perf_counter()
        chunks = self.retriever.retrieve(
            tenant=tenant, question=question, top_k=top_k, filters=filters
        )
        retrieval_ms = (time.perf_counter() - t0) * 1000
        METRICS.observe("retrieval_ms", retrieval_ms)

        # Cap context passed to the model regardless of retrieval depth.
        chunks = chunks[: self._settings.max_context_chunks]

        t1 = time.perf_counter()
        generation = self.generator.generate(question, chunks)
        generation_ms = (time.perf_counter() - t1) * 1000
        METRICS.observe("generation_ms", generation_ms)

        if generation.tokens.get("total"):
            METRICS.increment("tokens_total", generation.tokens["total"])

        citations = [
            Citation(
                doc_id=ch.record.doc_id,
                source=ch.record.source,
                chunk_index=ch.record.chunk_index,
                score=round(ch.score, 4),
                snippet=_snippet(ch.record.text),
            )
            for ch in chunks
        ]

        return QueryResponse(
            answer=generation.answer,
            citations=citations,
            model=generation.model,
            retrieval_ms=round(retrieval_ms, 2),
            generation_ms=round(generation_ms, 2),
            tokens=generation.tokens,
            request_id=request_id,
        )

    def delete_document(self, *, tenant: str, doc_id: str) -> int:
        removed = self.ingestion.delete_document(tenant=tenant, doc_id=doc_id)
        METRICS.increment("documents_deleted")
        return removed


def _snippet(text: str, limit: int = 240) -> str:
    text = text.strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"
