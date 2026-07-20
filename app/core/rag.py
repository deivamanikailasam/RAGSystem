"""RagEngine — composes every component into one process-wide service.

This is the object the API layer talks to. It owns the embedding provider,
FAISS vector store, SQLite docstore, tenant registry, ingestion pipeline,
retriever, and generator.

Deployment modes (see docs/07-deployment-modes.md):

* **single_tenant** — one implicit corpus (``settings.single_tenant_id``). The
  registry is bypassed; per-tenant config falls back to global defaults and no
  quotas apply. Simplest possible internal doc bot.
* **multi_tenant** — the registry is the source of truth. Every operation
  resolves the tenant's per-tenant policy (prompt template, index type) and
  enforces its quotas (max documents, max queries/day). Tenants referenced by
  static ``API_KEYS`` are auto-seeded at startup so they work out of the box.
"""

from __future__ import annotations

import time
import uuid

from app.config import Settings
from app.core.bm25 import BM25Store
from app.core.condenser import build_condenser
from app.core.conversation import ConversationStore
from app.core.dialogue import DialogueManager
from app.core.dialogue_store import DialogueStore
from app.core.docstore import DocStore
from app.core.faq import FAQMatcher, FAQStore
from app.core.faqbot import FAQBot
from app.core.memory import MemoryStore
from app.core.embeddings import build_embedding_provider
from app.core.generator import build_generator
from app.core.ingest import IngestedDocResult, IngestionPipeline
from app.core.intents import build_intent_classifier
from app.core.reranker import build_reranker
from app.core.retriever import Retriever
from app.core.tenants import QuotaExceeded, Tenant, TenantRegistry
from app.core.vector_store import VectorStore
from app.core.voice_session import VoiceSessionManager, VoiceSessionStore
from app.models import ChatResponse, Citation, QueryResponse
from app.observability.metrics import METRICS


class RagEngine:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.embeddings = build_embedding_provider(settings)
        # Docstore first: shared-namespace vector isolation needs it to resolve
        # each tenant's ids.
        self.docstore = DocStore(settings.data_dir / "docstore.db")
        self.vector_store = VectorStore(
            data_dir=settings.data_dir,
            dimension=self.embeddings.dimension,
            index_type=settings.faiss_index_type,
            nlist=settings.ivf_nlist,
            nprobe=settings.ivf_nprobe,
            isolation=settings.tenant_isolation,
            docstore=self.docstore,
        )
        self.bm25 = BM25Store(self.docstore)
        self.tenants = TenantRegistry(
            settings.data_dir / "tenants.db",
            default_index_type=settings.faiss_index_type,
        )
        self.ingestion = IngestionPipeline(
            settings, self.embeddings, self.vector_store, self.docstore
        )
        self.retriever = Retriever(
            settings, self.embeddings, self.vector_store, self.docstore, self.bm25
        )
        self.reranker = build_reranker(settings)
        self.generator = build_generator(settings)
        self.conversations = ConversationStore(settings.data_dir / "conversations.db")
        self.condenser = build_condenser(settings)
        self.voice = VoiceSessionManager(
            VoiceSessionStore(settings.data_dir / "voice_sessions.db"), self
        )
        self.dialogue = DialogueManager(
            DialogueStore(settings.data_dir / "dialogue.db"),
            self,
            build_intent_classifier(settings),
            settings,
        )
        self.faqs = FAQStore(settings.data_dir / "faqs.db")
        self.faq_matcher = FAQMatcher(self.embeddings, self.faqs)
        self.memory = MemoryStore(settings.data_dir / "memory.db")
        self.faq_bot = FAQBot(self, self.faq_matcher, self.memory, settings)

        if not settings.is_single_tenant:
            self._seed_static_tenants()

    def _seed_static_tenants(self) -> None:
        """Ensure tenants referenced by static API_KEYS exist in the registry."""
        for tenant_id in set(self._settings.api_key_map.values()):
            self.tenants.ensure(tenant_id)

    # -- per-tenant config resolution -------------------------------------- #
    def _tenant_config(self, tenant: str) -> Tenant | None:
        """Registry record for a tenant, or None in single-tenant mode."""
        if self._settings.is_single_tenant:
            return None
        return self.tenants.get(tenant)

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
        cfg = self._tenant_config(tenant)

        # Warm the index with the tenant's configured type before the pipeline
        # touches it (first-touch fixes the type for new indices).
        index_type = cfg.index_type if cfg else self._settings.faiss_index_type
        self.vector_store.for_tenant(tenant, index_type=index_type)

        # Enforce the document quota (only counts genuinely new documents).
        if cfg and cfg.max_documents > 0:
            existing = self.docstore.get_document(tenant, doc_id) if doc_id else None
            if existing is None and self.docstore.count_documents(tenant) >= cfg.max_documents:
                raise QuotaExceeded("max_documents", cfg.max_documents)

        with METRICS.timer("ingest_ms"):
            result = self.ingestion.ingest_document(
                tenant=tenant, text=text, source=source,
                doc_id=doc_id, metadata=metadata,
            )
        # Corpus changed → rebuild this tenant's BM25 index on next query.
        if not result.skipped:
            self.bm25.invalidate(tenant)
        METRICS.increment("documents_ingested")
        METRICS.increment("chunks_ingested", result.chunks)
        return result

    def _candidate_pool(self, final_k: int) -> int:
        """Stage-1 pool size: >= final_k so the reranker can promote a passage.

        ``none`` reranking skips the over-fetch (there is nothing to reorder).
        """
        if self.reranker.name == "none":
            return final_k
        return max(self._settings.rerank_candidates, final_k)

    def retrieve(
        self,
        *,
        tenant: str,
        question: str,
        top_k: int | None = None,
        filters: dict[str, str] | None = None,
    ):
        """Run stage-1 retrieval + stage-2 reranking and return the ranked chunks.

        This is the generation-free path used by the retrieval evaluation harness
        (``eval/retrieval_eval.py``); ``query`` performs the same two stages and
        then calls the generator.
        """
        final_k = top_k or self._settings.retrieval_top_k
        candidates = self.retriever.retrieve(
            tenant=tenant, question=question,
            limit=self._candidate_pool(final_k), filters=filters,
        )
        return self.reranker.rerank(question, candidates, top_n=final_k)

    def query(
        self,
        *,
        tenant: str,
        question: str,
        top_k: int | None = None,
        filters: dict[str, str] | None = None,
    ) -> QueryResponse:
        request_id = uuid.uuid4().hex
        cfg = self._tenant_config(tenant)

        # Enforce and record the per-day query quota.
        if cfg and cfg.max_queries_per_day > 0:
            if self.tenants.queries_today(tenant) >= cfg.max_queries_per_day:
                raise QuotaExceeded("max_queries_per_day", cfg.max_queries_per_day)
        if cfg:
            self.tenants.record_query(tenant)

        METRICS.increment("queries")

        final_k = top_k or self._settings.retrieval_top_k

        t0 = time.perf_counter()
        candidates = self.retriever.retrieve(
            tenant=tenant, question=question,
            limit=self._candidate_pool(final_k), filters=filters,
        )
        retrieval_ms = (time.perf_counter() - t0) * 1000
        METRICS.observe("retrieval_ms", retrieval_ms)

        # Stage 2 — rerank the candidates and keep the best final_k.
        t_r = time.perf_counter()
        chunks = self.reranker.rerank(question, candidates, top_n=final_k)
        rerank_ms = (time.perf_counter() - t_r) * 1000
        METRICS.observe("rerank_ms", rerank_ms)

        # Cap context passed to the model regardless of retrieval depth.
        chunks = chunks[: self._settings.max_context_chunks]

        system_prompt = cfg.prompt_template if cfg else None

        t1 = time.perf_counter()
        generation = self.generator.generate(question, chunks, system_prompt)
        generation_ms = (time.perf_counter() - t1) * 1000
        METRICS.observe("generation_ms", generation_ms)

        if generation.tokens.get("total"):
            METRICS.increment("tokens_total", generation.tokens["total"])

        citations = _to_citations(chunks)

        return QueryResponse(
            answer=generation.answer,
            citations=citations,
            model=generation.model,
            retrieval_mode=self._settings.retrieval_mode,
            reranker=self.reranker.name,
            retrieval_ms=round(retrieval_ms, 2),
            rerank_ms=round(rerank_ms, 2),
            generation_ms=round(generation_ms, 2),
            tokens=generation.tokens,
            request_id=request_id,
        )

    # -- multi-turn chat --------------------------------------------------- #
    def chat(
        self,
        *,
        tenant: str,
        message: str,
        session_id: str | None = None,
        top_k: int | None = None,
        filters: dict[str, str] | None = None,
    ) -> ChatResponse:
        """One conversational turn: condense → retrieve → generate → persist.

        Maintains per-session history so follow-ups resolve against earlier
        turns. A new ``session_id`` is minted when none is supplied.
        """
        request_id = uuid.uuid4().hex
        session_id = session_id or uuid.uuid4().hex
        cfg = self._tenant_config(tenant)

        # Same daily quota as /query (a chat turn is a retrieval + generation).
        if cfg and cfg.max_queries_per_day > 0:
            if self.tenants.queries_today(tenant) >= cfg.max_queries_per_day:
                raise QuotaExceeded("max_queries_per_day", cfg.max_queries_per_day)
        if cfg:
            self.tenants.record_query(tenant)
        METRICS.increment("chat_turns")

        self.conversations.ensure_session(tenant, session_id)
        history = self.conversations.recent_messages(
            tenant, session_id, self._settings.chat_history_turns
        )

        # Rewrite the follow-up into a standalone query for retrieval.
        standalone = self.condenser.condense(history, message)

        final_k = top_k or self._settings.retrieval_top_k
        t0 = time.perf_counter()
        candidates = self.retriever.retrieve(
            tenant=tenant, question=standalone,
            limit=self._candidate_pool(final_k), filters=filters,
        )
        chunks = self.reranker.rerank(standalone, candidates, top_n=final_k)
        chunks = chunks[: self._settings.max_context_chunks]
        retrieval_ms = (time.perf_counter() - t0) * 1000
        METRICS.observe("retrieval_ms", retrieval_ms)

        # Generate, giving the model the recent turns for reference resolution.
        history_messages = [
            {"role": m.role, "content": m.content} for m in history
        ]
        system_prompt = cfg.prompt_template if cfg else None
        t1 = time.perf_counter()
        generation = self.generator.generate(
            message, chunks, system_prompt, history_messages
        )
        generation_ms = (time.perf_counter() - t1) * 1000
        METRICS.observe("generation_ms", generation_ms)
        if generation.tokens.get("total"):
            METRICS.increment("tokens_total", generation.tokens["total"])

        citations = _to_citations(chunks)

        # Persist the turn (user message, then assistant reply + its citations).
        self.conversations.append_message(
            tenant=tenant, session_id=session_id, role="user", content=message
        )
        turn_index = self.conversations.append_message(
            tenant=tenant, session_id=session_id, role="assistant",
            content=generation.answer,
            citations=[c.model_dump() for c in citations],
        )

        return ChatResponse(
            session_id=session_id,
            turn_index=turn_index,
            answer=generation.answer,
            citations=citations,
            standalone_question=standalone,
            model=generation.model,
            retrieval_mode=self._settings.retrieval_mode,
            reranker=self.reranker.name,
            condenser=self.condenser.name,
            retrieval_ms=round(retrieval_ms, 2),
            generation_ms=round(generation_ms, 2),
            tokens=generation.tokens,
            request_id=request_id,
        )

    # -- FAQ management (keeps the matcher cache in sync) ------------------- #
    def add_faq(self, *, tenant: str, question: str, answer: str,
                tags: list[str] | None = None, faq_id: str | None = None):
        faq = self.faqs.add(tenant, question, answer, tags, faq_id)
        self.faq_matcher.invalidate(tenant)
        return faq

    def list_faqs(self, *, tenant: str):
        return self.faqs.list(tenant)

    def delete_faq(self, *, tenant: str, faq_id: str) -> bool:
        removed = self.faqs.delete(tenant, faq_id)
        self.faq_matcher.invalidate(tenant)
        return removed

    def get_conversation(self, *, tenant: str, session_id: str):
        """Return the full message history for a session (empty if unknown)."""
        return self.conversations.get_messages(tenant, session_id)

    def delete_conversation(self, *, tenant: str, session_id: str) -> int:
        return self.conversations.delete_session(tenant, session_id)

    def delete_document(self, *, tenant: str, doc_id: str) -> int:
        removed = self.ingestion.delete_document(tenant=tenant, doc_id=doc_id)
        self.bm25.invalidate(tenant)
        METRICS.increment("documents_deleted")
        return removed

    def purge_tenant(self, tenant: str) -> None:
        """Delete all of a tenant's data: vectors + metadata (not the registry row)."""
        # Drop vectors first: shared-namespace mode reads the tenant's ids from
        # the docstore to know which vectors to remove from the shared index.
        self.vector_store.drop_tenant(tenant)
        self.docstore.delete_tenant(tenant)
        self.bm25.invalidate(tenant)
        self.conversations.delete_tenant(tenant)
        self.voice._store.delete_tenant(tenant)  # noqa: SLF001
        self.dialogue._store.delete_tenant(tenant)  # noqa: SLF001
        self.faqs.delete_tenant(tenant)
        self.faq_matcher.invalidate(tenant)
        self.memory.delete_tenant(tenant)

    def tenant_stats(self, tenant: str) -> dict[str, object]:
        """Runtime stats for the current tenant (used by GET /v1/me)."""
        cfg = self._tenant_config(tenant)
        return {
            "tenant": tenant,
            "mode": self._settings.deployment_mode,
            "documents": self.docstore.count_documents(tenant),
            "vectors": self.vector_store.for_tenant(tenant).ntotal,
            "queries_today": self.tenants.queries_today(tenant) if cfg else 0,
            "quotas": {
                "max_documents": cfg.max_documents if cfg else 0,
                "max_queries_per_day": cfg.max_queries_per_day if cfg else 0,
            },
        }


def _to_citations(chunks) -> list[Citation]:
    return [
        Citation(
            doc_id=ch.record.doc_id,
            source=ch.record.source,
            chunk_index=ch.record.chunk_index,
            score=round(ch.score, 4),
            vector_score=(
                round(ch.vector_score, 4) if ch.vector_score is not None else None
            ),
            bm25_score=(
                round(ch.bm25_score, 4) if ch.bm25_score is not None else None
            ),
            rerank_score=(
                round(ch.rerank_score, 4) if ch.rerank_score is not None else None
            ),
            snippet=_snippet(ch.record.text),
        )
        for ch in chunks
    ]


def _snippet(text: str, limit: int = 240) -> str:
    text = text.strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"
