"""Ingestion pipeline: parse → chunk → embed → index (idempotent, versioned).

The pipeline is deliberately idempotent per ``(tenant, doc_id)``: re-ingesting a
document removes the previous version's vectors before adding the new ones, so
updates never leave stale chunks behind. Documents are content-addressed — if
the text is unchanged, ingestion is a no-op.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import numpy as np

from app.config import Settings
from app.core.chunking import chunk_text
from app.core.docstore import ChunkRecord, DocStore
from app.core.embeddings import EmbeddingProvider
from app.core.vector_store import VectorStore


@dataclass
class IngestedDocResult:
    doc_id: str
    version: int
    chunks: int
    skipped: bool = False


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def derive_doc_id(source: str, text: str) -> str:
    return hashlib.sha1(f"{source}:{content_hash(text)}".encode()).hexdigest()[:16]


class IngestionPipeline:
    def __init__(
        self,
        settings: Settings,
        embeddings: EmbeddingProvider,
        vector_store: VectorStore,
        docstore: DocStore,
    ) -> None:
        self._settings = settings
        self._embeddings = embeddings
        self._store = vector_store
        self._docs = docstore

    def ingest_document(
        self,
        *,
        tenant: str,
        text: str,
        source: str,
        doc_id: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> IngestedDocResult:
        metadata = metadata or {}
        doc_id = doc_id or derive_doc_id(source, text)
        chash = content_hash(text)

        existing = self._docs.get_document(tenant, doc_id)
        if existing is not None and existing["content_hash"] == chash:
            # Unchanged content — nothing to do.
            return IngestedDocResult(
                doc_id=doc_id, version=int(existing["version"]),
                chunks=0, skipped=True,
            )

        # If a prior version exists, drop its vectors first (idempotent update).
        if existing is not None:
            old_ids = self._docs.vector_ids_for_doc(tenant, doc_id)
            if old_ids:
                self._store.for_tenant(tenant).remove(old_ids)
                self._docs.delete_doc_chunks(tenant, doc_id)

        version = self._docs.next_version(tenant, doc_id)

        chunks = chunk_text(
            text,
            chunk_tokens=self._settings.chunk_tokens,
            overlap=self._settings.chunk_overlap,
        )
        if not chunks:
            self._docs.upsert_document(
                tenant=tenant, doc_id=doc_id, version=version, source=source,
                content_hash=chash, created_at=time.time(),
            )
            return IngestedDocResult(doc_id=doc_id, version=version, chunks=0)

        vectors = self._embeddings.embed([c.text for c in chunks])

        base_id = self._docs.max_vector_id(tenant) + 1
        vector_ids = np.arange(base_id, base_id + len(chunks), dtype=np.int64)

        records = [
            ChunkRecord(
                vector_id=int(vector_ids[i]),
                tenant=tenant,
                doc_id=doc_id,
                version=version,
                chunk_index=chunks[i].index,
                source=source,
                text=chunks[i].text,
                metadata=metadata,
            )
            for i in range(len(chunks))
        ]

        tenant_index = self._store.for_tenant(tenant)
        tenant_index.add(vectors, vector_ids)
        tenant_index.persist()
        self._docs.add_chunks(records)
        self._docs.upsert_document(
            tenant=tenant, doc_id=doc_id, version=version, source=source,
            content_hash=chash, created_at=time.time(),
        )
        return IngestedDocResult(doc_id=doc_id, version=version, chunks=len(chunks))

    def delete_document(self, *, tenant: str, doc_id: str) -> int:
        vector_ids = self._docs.vector_ids_for_doc(tenant, doc_id)
        removed = 0
        if vector_ids:
            tenant_index = self._store.for_tenant(tenant)
            removed = tenant_index.remove(vector_ids)
            tenant_index.persist()
        self._docs.delete_doc_chunks(tenant, doc_id)
        return removed
