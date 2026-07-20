"""Dialogue manager — classify intent, decide an action, respond, persist.

Sits above the RAG chat: for each user turn it

1. **classifies the intent** (`app/core/intents.py`),
2. applies a **policy** mapping intent → action,
3. **acts** — a QUESTION is answered by the grounded RAG chat (with any slots as
   metadata filters); social intents get canned responses without spending a RAG
   call,
4. **persists** the intent + action and advances the dialogue state
   (`app/core/dialogue_store.py`).

The session id doubles as the chat/voice session id, so RAG answers keep full
conversation memory and follow-up condensing across the dialogue.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.config import Settings
from app.core.dialogue_store import DialogueState, DialogueStore, IntentEvent
from app.core.intents import Intent


class Action(str, Enum):
    ANSWER = "answer"          # run the grounded RAG chat
    GREET = "greet"
    CLOSE = "close"
    HELP = "help"
    SMALLTALK = "smalltalk"
    ACKNOWLEDGE = "acknowledge"


# Dialogue policy: which action each intent triggers.
_POLICY: dict[Intent, Action] = {
    Intent.QUESTION: Action.ANSWER,
    Intent.GREETING: Action.GREET,
    Intent.GOODBYE: Action.CLOSE,
    Intent.HELP: Action.HELP,
    Intent.SMALLTALK: Action.SMALLTALK,
    Intent.AFFIRM: Action.ACKNOWLEDGE,
    Intent.DENY: Action.ACKNOWLEDGE,
}

# Canned replies for non-RAG actions (ANSWER is produced by the chat engine).
_CANNED: dict[Action, str] = {
    Action.GREET: "Hello! I'm your documentation assistant. Ask me anything about "
                  "the docs, or say 'help' to see what I can do.",
    Action.CLOSE: "You're welcome — happy to help. Goodbye!",
    Action.HELP: "I answer questions grounded in your documents and cite the "
                 "sources I used. Just ask a question; say 'bye' to end.",
    Action.SMALLTALK: "I'm a documentation assistant, so I'll stick to answering "
                      "questions about your docs. What would you like to know?",
    Action.ACKNOWLEDGE: "Got it. What would you like to know?",
}


@dataclass
class DialogueTurn:
    session_id: str
    turn_index: int
    intent: Intent
    confidence: float
    action: Action
    slots: dict[str, str]
    answer: str
    classifier: str
    citations: list[dict] = field(default_factory=list)
    standalone_question: str | None = None
    model: str | None = None


class DialogueManager:
    def __init__(
        self, store: DialogueStore, engine: Any, classifier: Any, settings: Settings
    ) -> None:
        self._store = store
        self._engine = engine  # RagEngine (avoids a circular import)
        self._classifier = classifier
        self._settings = settings

    def handle(
        self,
        *,
        tenant: str,
        message: str,
        session_id: str | None = None,
        top_k: int | None = None,
        filters: dict[str, str] | None = None,
    ) -> DialogueTurn:
        session_id = session_id or uuid.uuid4().hex

        result = self._classifier.classify(message)
        intent = result.intent
        # Low-confidence non-question → treat as a question (answer by default).
        if (
            intent is not Intent.QUESTION
            and result.confidence < self._settings.intent_confidence_threshold
        ):
            intent = Intent.QUESTION

        action = _POLICY[intent]

        answer = ""
        citations: list[dict] = []
        standalone = None
        model = None

        if action is Action.ANSWER:
            # Slots (e.g. doc_type) become retrieval metadata filters.
            merged_filters = {**(filters or {}), **result.slots}
            chat = self._engine.chat(
                tenant=tenant, message=message, session_id=session_id,
                top_k=top_k, filters=merged_filters,
            )
            answer = chat.answer
            citations = [c.model_dump() for c in chat.citations]
            standalone = chat.standalone_question
            model = chat.model
        else:
            answer = _CANNED[action]

        event = self._store.record_turn(
            tenant=tenant, session_id=session_id, message=message,
            intent=intent.value, confidence=result.confidence,
            action=action.value, slots=result.slots,
        )

        return DialogueTurn(
            session_id=session_id,
            turn_index=event.turn_index,
            intent=intent,
            confidence=result.confidence,
            action=action,
            slots=result.slots,
            answer=answer,
            classifier=self._classifier.name,
            citations=citations,
            standalone_question=standalone,
            model=model,
        )

    def get_state(self, tenant: str, session_id: str) -> DialogueState | None:
        return self._store.get_state(tenant, session_id)

    def get_intents(self, tenant: str, session_id: str) -> list[IntentEvent]:
        return self._store.get_intents(tenant, session_id)

    def delete_session(self, tenant: str, session_id: str) -> int:
        return self._store.delete_session(tenant, session_id)
