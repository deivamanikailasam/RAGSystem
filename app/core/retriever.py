"""Retrieval: embed query → FAISS search → metadata filter → optional rerank."""

from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.core.docstore import ChunkRecord, DocStore
from app.core.embeddings import EmbeddingProvider
from app.core.vector_store import VectorStore


@dataclass
class RetrievedChunk:
    record: ChunkRecord
    score: float


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
        top_k: int | None = None,
        filters: dict[str, str] | None = None,
    ) -> list[RetrievedChunk]:
        filters = filters or {}
        k = top_k or self._settings.retrieval_top_k

        # Over-fetch so that metadata filtering still leaves enough candidates.
        fetch_k = k * 4 if filters else k
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

        if self._settings.rerank_enabled:
            results = self._rerank(question, results)

        return results[:k]

    @staticmethod
    def _matches(rec: ChunkRecord, filters: dict[str, str]) -> bool:
        for key, value in filters.items():
            if rec.metadata.get(key) != value:
                return False
        return True

    def _rerank(
        self, question: str, results: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """Lightweight lexical rerank layered on top of vector scores.

        A production system would call a cross-encoder / LLM reranker here. We
        provide a cheap, dependency-free heuristic — term-overlap boosting —
        so the hook is exercised end-to-end offline. Swap this method to plug
        in a real reranker (see docs/03-implementation.md §3.2).
        """
        q_terms = {t.lower() for t in question.split()}

        def boosted(item: RetrievedChunk) -> float:
            terms = {t.lower() for t in item.record.text.split()}
            overlap = len(q_terms & terms) / (len(q_terms) or 1)
            return item.score + 0.1 * overlap

        return sorted(results, key=boosted, reverse=True)
