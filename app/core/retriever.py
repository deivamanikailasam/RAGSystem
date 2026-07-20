"""Retrieval stage (stage 1): dense, sparse, or hybrid candidate retrieval.

Modes (``RETRIEVAL_MODE``):

* ``vector`` — dense FAISS search only (semantic similarity).
* ``bm25``   — sparse BM25 search only (lexical / keyword match).
* ``hybrid`` — run both over the full corpus and **fuse** the two ranked lists
  (RRF or weighted), so a passage found by *either* signal can make the
  candidate pool. This is the default and the point of hybrid retrieval: dense
  catches paraphrases, sparse catches exact terms/IDs neither would find alone.

The result is a candidate pool ordered by the stage-1 score, handed to the
reranking stage (:mod:`app.core.reranker`) by
:meth:`app.core.rag.RagEngine.query`.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.core.bm25 import BM25Store
from app.core.docstore import ChunkRecord, DocStore
from app.core.embeddings import EmbeddingProvider
from app.core.fusion import reciprocal_rank_fusion, weighted_fusion
from app.core.vector_store import VectorStore


@dataclass
class RetrievedChunk:
    record: ChunkRecord
    score: float                       # stage-1 primary score (fused in hybrid)
    vector_score: float | None = None  # dense cosine, if vector search ran
    bm25_score: float | None = None    # sparse BM25, if BM25 search ran
    rerank_score: float | None = None  # set by the reranker (stage 2)


class Retriever:
    def __init__(
        self,
        settings: Settings,
        embeddings: EmbeddingProvider,
        vector_store: VectorStore,
        docstore: DocStore,
        bm25_store: BM25Store,
    ) -> None:
        self._settings = settings
        self._embeddings = embeddings
        self._store = vector_store
        self._docs = docstore
        self._bm25 = bm25_store

    def retrieve(
        self,
        *,
        tenant: str,
        question: str,
        limit: int,
        filters: dict[str, str] | None = None,
    ) -> list[RetrievedChunk]:
        filters = filters or {}
        mode = self._settings.retrieval_mode
        # Over-fetch when filtering so filtered-out hits don't starve the pool.
        fetch_n = limit * 4 if filters else limit

        dense: dict[int, float] = {}
        sparse: dict[int, float] = {}
        if mode in ("vector", "hybrid"):
            dense = self._vector_search(tenant, question, fetch_n)
        if mode in ("bm25", "hybrid"):
            sparse = self._bm25_search(tenant, question, fetch_n)

        fused = self._fuse(mode, dense, sparse)
        if not fused:
            return []

        # Hydrate metadata, apply filters, attach per-signal scores.
        records = self._docs.get_chunks(tenant, list(fused.keys()))
        results: list[RetrievedChunk] = []
        for vector_id, primary in fused.items():
            rec = records.get(vector_id)
            if rec is None:
                continue  # vector removed between search and lookup
            if not self._matches(rec, filters):
                continue
            results.append(
                RetrievedChunk(
                    record=rec,
                    score=primary,
                    vector_score=dense.get(vector_id),
                    bm25_score=sparse.get(vector_id),
                )
            )

        results.sort(key=lambda c: c.score, reverse=True)
        return results[:limit]

    # -- individual signals ------------------------------------------------ #
    def _vector_search(self, tenant: str, question: str, n: int) -> dict[int, float]:
        query_vec = self._embeddings.embed([question])[0]
        hits = self._store.for_tenant(tenant).search(query_vec, n)
        return {
            h.vector_id: h.score
            for h in hits
            if h.score >= self._settings.min_score
        }

    def _bm25_search(self, tenant: str, question: str, n: int) -> dict[int, float]:
        return {vid: score for vid, score in self._bm25.for_tenant(tenant).search(question, n)}

    # -- fusion ------------------------------------------------------------ #
    def _fuse(
        self, mode: str, dense: dict[int, float], sparse: dict[int, float]
    ) -> dict[int, float]:
        if mode == "vector":
            return dense
        if mode == "bm25":
            return sparse
        # hybrid
        if self._settings.hybrid_fusion == "weighted":
            return weighted_fusion(dense, sparse, alpha=self._settings.hybrid_alpha)
        dense_ranked = _rank(dense)
        sparse_ranked = _rank(sparse)
        return reciprocal_rank_fusion(
            [dense_ranked, sparse_ranked], k=self._settings.rrf_k
        )

    @staticmethod
    def _matches(rec: ChunkRecord, filters: dict[str, str]) -> bool:
        for key, value in filters.items():
            if rec.metadata.get(key) != value:
                return False
        return True


def _rank(scores: dict[int, float]) -> list[int]:
    """Vector ids ordered best-first by score (input to RRF)."""
    return [vid for vid, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]
