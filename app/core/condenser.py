"""Question condensing — turn a follow-up + history into a standalone query.

The core problem of multi-turn RAG: a follow-up like *"what about its latency?"*
is meaningless to the retriever on its own — "it" refers to something from an
earlier turn. Before retrieval we **condense** the conversation + follow-up into
a self-contained question, then retrieve with that.

Two strategies, one interface (`condense(history, question) -> str`):

* :class:`LLMCondenser` — asks the model to rewrite the follow-up as a
  standalone question given the history (best quality; needs OpenAI).
* :class:`HeuristicCondenser` — offline fallback: if there's history, prepend
  the most recent user question so its topic terms are present for retrieval.
  Crude but genuinely improves follow-up recall with zero dependencies.

`build_condenser` picks LLM when a key is configured, else the heuristic; if
condensing is disabled the follow-up is used as-is.
"""

from __future__ import annotations

import logging
import re

from app.config import Settings
from app.core.conversation import Message

logger = logging.getLogger("ragsystem.condenser")

# Pronouns/determiners that, when they appear near the *start* of a question,
# usually refer back to something in the conversation.
_REFERENTS = {
    "it", "its", "they", "them", "their", "that", "this", "those", "these",
    "he", "she", "him", "his", "her", "hers",
}
# Phrases/conjunctions that, when a question *begins* with them, signal a follow-up.
_LEADING_PHRASES = ("what about", "how about", "what if")
_LEADING_WORDS = {"and", "also", "then", "why", "but"}


def _looks_like_followup(question: str) -> bool:
    q = question.strip().lower()
    words = re.findall(r"[a-z']+", q)
    if not words:
        return False
    if len(words) <= 4:                       # terse queries lean on context
        return True
    if _REFERENTS & set(words[:6]):           # an early referent ("its", "that")
        return True
    if any(q.startswith(p) for p in _LEADING_PHRASES):
        return True
    return words[0] in _LEADING_WORDS         # leading conjunction only


class HeuristicCondenser:
    name = "heuristic"

    def condense(self, history: list[Message], question: str) -> str:
        if not history:
            return question
        # Most recent user question in the history (skip the current one).
        prev_user = next(
            (m.content for m in reversed(history) if m.role == "user"), None
        )
        if prev_user is None:
            return question
        if not _looks_like_followup(question):
            return question
        # Prepend prior-topic terms so the retriever has the referent in scope.
        return f"{prev_user} {question}"


class LLMCondenser:
    name = "llm"

    def __init__(self, settings: Settings) -> None:
        from openai import OpenAI

        self._client = OpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.openai_timeout_seconds,
            max_retries=settings.openai_max_retries,
        )
        self._model = settings.generation_model

    def condense(self, history: list[Message], question: str) -> str:
        if not history:
            return question
        convo = "\n".join(f"{m.role}: {m.content}" for m in history)
        prompt = (
            "Given the conversation below and a follow-up question, rewrite the "
            "follow-up as a STANDALONE question that can be understood without "
            "the conversation. If it is already standalone, return it unchanged. "
            "Return only the rewritten question.\n\n"
            f"Conversation:\n{convo}\n\nFollow-up: {question}\n\nStandalone question:"
        )
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            rewritten = (resp.choices[0].message.content or "").strip()
            return rewritten or question
        except Exception as exc:  # noqa: BLE001 — never fail a chat on condensing
            logger.warning("LLM condense failed (%s); using raw question", exc)
            return question


def build_condenser(settings: Settings):
    if not settings.chat_condense_question:
        return _NoOpCondenser()
    if settings.use_openai:
        return LLMCondenser(settings)
    return HeuristicCondenser()


class _NoOpCondenser:
    name = "none"

    def condense(self, history: list[Message], question: str) -> str:
        return question
