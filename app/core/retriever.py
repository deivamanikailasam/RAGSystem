"""Retrieval stage: embed query → FAISS search → metadata filter.

This is the *first* stage of two-stage retrieval. It is intentionally
recall-oriented: it returns a candidate pool (larger than the final ``top_k``)
ordered by raw vector similarity. The precision-oriented reordering happens in
the separate reranking stage (:mod:`app.core.reranker`), wired together in
:meth:`app.core.rag.RagEngine.query`.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.core.docstore import ChunkRecord, DocStore
from app.core.embeddings import EmbeddingProvider
from app.core.vector_store import VectorStore


@dataclass
class RetrievedChunk:
    record: ChunkRecord
    score: float                      # raw vector similarity (stage 1)
    rerank_score: float | None = None  # set by the reranker (stage 2)


class Retriever:
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

    def retrieve(
        self,
        *,
        tenant: str,
        question: str,
        limit: int,
        filters: dict[str, str] | None = None,
    ) -> list[RetrievedChunk]:
        """Return up to ``limit`` candidate chunks, ordered by vector score.

        ``limit`` is the *candidate pool* size — typically larger than the final
        answer's ``top_k`` so the reranker has room to promote a better passage
        that vector search ranked lower.
        """
        filters = filters or {}

        # Over-fetch further when filtering, so filtered-out hits don't starve
        # the candidate pool.
        fetch_k = limit * 4 if filters else limit
        query_vec = self._embeddings.embed([question])[0]

        hits = self._store.for_tenant(tenant).search(query_vec, fetch_k)
        if not hits:
            return []

        records = self._docs.get_chunks(tenant, [h.vector_id for h in hits])

        results: list[RetrievedChunk] = []
        for hit in hits:
            rec = records.get(hit.vector_id)
            if rec is None:
                continue  # vector removed between search and lookup
            if hit.score < self._settings.min_score:
                continue
            if not self._matches(rec, filters):
                continue
            results.append(RetrievedChunk(record=rec, score=hit.score))
            if len(results) >= limit:
                break

        return results

    @staticmethod
    def _matches(rec: ChunkRecord, filters: dict[str, str]) -> bool:
        for key, value in filters.items():
            if rec.metadata.get(key) != value:
                return False
        return True
