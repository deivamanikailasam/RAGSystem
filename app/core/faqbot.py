"""Context-aware FAQ bot with memory.

Orchestrates a turn:

1. **Condense** the follow-up against conversation history (short-term memory).
2. **Extract & store** any new long-term memories from the user's message.
3. **Recall** relevant long-term memories for this user.
4. **FAQ-first:** match the (condensed) question against the curated FAQ base;
   if the score clears the threshold, return the curated answer — exact and
   LLM-free.
5. **RAG fallback:** otherwise retrieve + generate over documents, injecting the
   recalled memories into the system prompt so the answer is user-aware.
6. **Persist** the turn to the conversation and remember the topic.

The result reports its `source` (`faq` / `rag` / `fallback`), the memories used,
and citations, so the behavior is fully transparent.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings
from app.core.faq import FAQMatcher
from app.core.generator import SYSTEM_PROMPT
from app.core.memory import MemoryStore, extract_memories
from app.core.tenants import QuotaExceeded


@dataclass
class FAQBotAnswer:
    session_id: str
    user_id: str
    source: str                       # "faq" | "rag" | "fallback"
    answer: str
    standalone_question: str
    citations: list[dict] = field(default_factory=list)
    memories_used: list[str] = field(default_factory=list)
    faq_id: str | None = None
    faq_question: str | None = None
    score: float | None = None
    model: str | None = None


def _memory_prompt(memories: list[str]) -> str:
    if not memories:
        return ""
    lines = "\n".join(f"- {m}" for m in memories)
    return (
        "\n\nKnown about the user (use only if relevant, do not invent):\n" + lines
    )


class FAQBot:
    def __init__(self, engine: Any, faq_matcher: FAQMatcher,
                 memory: MemoryStore, settings: Settings) -> None:
        self._engine = engine  # RagEngine
        self._faq = faq_matcher
        self._memory = memory
        self._settings = settings

    def ask(
        self,
        *,
        tenant: str,
        message: str,
        user_id: str | None = None,
        session_id: str | None = None,
        top_k: int | None = None,
        filters: dict[str, str] | None = None,
    ) -> FAQBotAnswer:
        session_id = session_id or uuid.uuid4().hex
        # Default the memory scope to the session; pass a stable user_id for
        # cross-session memory.
        user_id = user_id or session_id
        engine = self._engine

        # Daily query quota (same as chat).
        cfg = engine._tenant_config(tenant)  # noqa: SLF001
        if cfg and cfg.max_queries_per_day > 0:
            if engine.tenants.queries_today(tenant) >= cfg.max_queries_per_day:
                raise QuotaExceeded("max_queries_per_day", cfg.max_queries_per_day)
        if cfg:
            engine.tenants.record_query(tenant)

        # 1) Condense against conversation history (short-term memory).
        history = engine.conversations.recent_messages(
            tenant, session_id, self._settings.chat_history_turns
        )
        standalone = engine.condenser.condense(history, message)

        # 2) Extract + store new long-term memories from the user message.
        if self._settings.memory_enabled:
            for kind, content in extract_memories(message):
                self._memory.remember(tenant, user_id, kind, content)

        # 3) Recall relevant memories.
        recalled = (
            self._memory.recall(
                tenant, user_id, query=standalone,
                limit=self._settings.memory_recall_limit,
            )
            if self._settings.memory_enabled
            else []
        )
        memories_used = [m.content for m in recalled]

        # 4) FAQ-first.
        match = self._faq.match(tenant, standalone)
        if match is not None and match.score >= self._settings.faq_match_threshold:
            answer = match.faq.answer
            citations = [{
                "type": "faq", "faq_id": match.faq.faq_id,
                "question": match.faq.question, "score": round(match.score, 4),
            }]
            result = FAQBotAnswer(
                session_id=session_id, user_id=user_id, source="faq",
                answer=answer, standalone_question=standalone, citations=citations,
                memories_used=memories_used, faq_id=match.faq.faq_id,
                faq_question=match.faq.question, score=round(match.score, 4),
            )
            topic = match.faq.question
            model = None
        else:
            # 5) RAG fallback, memory-augmented.
            chunks = engine.retrieve(
                tenant=tenant, question=standalone, top_k=top_k, filters=filters
            )
            base = (cfg.prompt_template if cfg and cfg.prompt_template else SYSTEM_PROMPT)
            system_prompt = base + _memory_prompt(memories_used)
            history_messages = [{"role": m.role, "content": m.content} for m in history]
            generation = engine.generator.generate(
                message, chunks, system_prompt, history_messages
            )
            citations = [
                {"type": "doc", "doc_id": ch.record.doc_id, "source": ch.record.source,
                 "score": round(ch.score, 4)}
                for ch in chunks
            ]
            result = FAQBotAnswer(
                session_id=session_id, user_id=user_id,
                source="rag" if chunks else "fallback",
                answer=generation.answer, standalone_question=standalone,
                citations=citations, memories_used=memories_used, model=generation.model,
            )
            topic = citations[0]["doc_id"] if citations else None
            model = generation.model

        # 6) Persist the turn + remember the topic.
        engine.conversations.ensure_session(tenant, session_id)
        engine.conversations.append_message(
            tenant=tenant, session_id=session_id, role="user", content=message
        )
        engine.conversations.append_message(
            tenant=tenant, session_id=session_id, role="assistant",
            content=result.answer, citations=result.citations,
        )
        if self._settings.memory_enabled and topic:
            self._memory.remember(tenant, user_id, "topic", f"asked about {topic}")

        result.model = model
        return result
