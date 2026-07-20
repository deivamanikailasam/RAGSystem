"""Intent classification for the dialogue manager.

The dialogue manager needs to know *what the user is trying to do* before it can
decide how to respond — answering a documentation question, greeting, saying
goodbye, asking for help, or making small talk. This module classifies a user
message into one of a small taxonomy and extracts any slots (parameters) it can.

Two strategies behind one interface (`classify(message) -> IntentResult`):

* :class:`RuleIntentClassifier` — fast, deterministic keyword/pattern rules;
  offline, the default, and what the tests pin behavior against.
* :class:`LLMIntentClassifier` — an LLM classifies into the same taxonomy
  (better on paraphrases); needs OpenAI, falls back to rules on any error.

Design bias: this is a **documentation assistant**, so anything that isn't
clearly social is treated as a QUESTION and answered via RAG — the safe default.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum

from app.config import Settings

logger = logging.getLogger("ragsystem.intents")


class Intent(str, Enum):
    QUESTION = "question"     # a documentation question -> answer via RAG
    GREETING = "greeting"     # hi / hello
    GOODBYE = "goodbye"       # bye / thanks, that's all
    HELP = "help"             # what can you do?
    SMALLTALK = "smalltalk"   # how are you / who are you
    AFFIRM = "affirm"         # yes / correct
    DENY = "deny"             # no / nope


@dataclass
class IntentResult:
    intent: Intent
    confidence: float
    slots: dict[str, str] = field(default_factory=dict)
    matched_by: str = "rule"


_GREETING = {"hi", "hello", "hey", "yo", "howdy", "greetings", "hiya", "hullo"}
_AFFIRM = {"yes", "yeah", "yep", "yup", "sure", "correct", "right", "ok",
           "okay", "affirmative", "exactly", "please"}
_DENY = {"no", "nope", "nah", "negative", "incorrect"}
_QUESTION_WORDS = {
    "what", "why", "how", "when", "where", "which", "who", "whom", "whose",
    "does", "do", "is", "are", "can", "could", "should", "would", "will",
    "did", "explain", "describe", "list", "define", "tell", "give", "show",
}
_HELP_PATTERNS = (
    "what can you do", "how do you work", "what do you know", "can you help",
    "what are your capabilities", "how can you help", "what can i ask",
)
_SMALLTALK_PATTERNS = (
    "how are you", "who are you", "what is your name", "what's your name",
    "whats your name", "tell me a joke", "are you a robot", "are you human",
    "how's it going", "hows it going",
)
_GOODBYE_PATTERNS = (
    "bye", "goodbye", "see you", "see ya", "that's all", "thats all",
    "that is all", "no more questions", "nothing else", "we're done",
    "were done", "that will be all", "i'm done", "im done", "farewell",
)
_THANKS_PATTERNS = ("thank you", "thanks", "thx", "appreciate it", "cheers")

_DOC_TYPE_RE = re.compile(
    r"\b(?:in|from|within|under)\s+(?:the\s+)?([a-z][a-z0-9-]+)\s+"
    r"(?:docs|documentation|manual|guide|section|policy|policies)\b"
)


def _extract_slots(text: str) -> dict[str, str]:
    """Pull light slots from the message (currently a doc_type filter hint)."""
    slots: dict[str, str] = {}
    m = _DOC_TYPE_RE.search(text)
    if m:
        slots["doc_type"] = m.group(1)
    return slots


class RuleIntentClassifier:
    name = "rule"

    def classify(self, message: str) -> IntentResult:
        text = message.strip().lower()
        words = re.findall(r"[a-z']+", text)
        n = len(words)
        slots = _extract_slots(text)
        has_question = text.endswith("?") or (bool(words) and words[0] in _QUESTION_WORDS)

        # 1) Terse social one-liners. Affirm/deny only when the message is
        #    essentially just that word (<=2), so "no more questions" (a
        #    farewell) is not mistaken for "no".
        if words:
            if n <= 3 and words[0] in _GREETING:
                return IntentResult(Intent.GREETING, 0.95, slots, "greeting")
            if n <= 2 and words[0] in _AFFIRM:
                return IntentResult(Intent.AFFIRM, 0.9, slots, "affirm")
            if n <= 2 and words[0] in _DENY:
                return IntentResult(Intent.DENY, 0.9, slots, "deny")

        # 2) Explicit capability / small-talk phrases.
        if any(p in text for p in _HELP_PATTERNS):
            return IntentResult(Intent.HELP, 0.9, slots, "help")
        if any(p in text for p in _SMALLTALK_PATTERNS):
            return IntentResult(Intent.SMALLTALK, 0.85, slots, "smalltalk")

        # 3) Farewell / thanks — but a real question takes precedence.
        if not has_question and any(
            p in text for p in (*_GOODBYE_PATTERNS, *_THANKS_PATTERNS)
        ):
            return IntentResult(Intent.GOODBYE, 0.85, slots, "goodbye")

        # 4) A greeting that leads into more (short, no question) -> greeting.
        if words and words[0] in _GREETING and not has_question and n <= 6:
            return IntentResult(Intent.GREETING, 0.8, slots, "greeting")

        # 5) Question, or default (a documentation assistant answers by default).
        if has_question:
            return IntentResult(Intent.QUESTION, 0.9, slots, "question")
        return IntentResult(Intent.QUESTION, 0.5, slots, "default")


class LLMIntentClassifier:
    name = "llm"

    def __init__(self, settings: Settings) -> None:
        from openai import OpenAI

        self._client = OpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.openai_timeout_seconds,
            max_retries=settings.openai_max_retries,
        )
        self._model = settings.generation_model
        self._fallback = RuleIntentClassifier()

    def classify(self, message: str) -> IntentResult:
        import json

        labels = ", ".join(i.value for i in Intent)
        prompt = (
            "Classify the user's message into exactly one intent from this list: "
            f"{labels}. Reply with ONLY a JSON object "
            '{"intent": "<label>", "confidence": <0..1>}.\n\n'
            f"Message: {message}"
        )
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content or "{}")
            intent = Intent(data.get("intent", "question"))
            confidence = float(data.get("confidence", 0.7))
            return IntentResult(intent, confidence, _extract_slots(message.lower()), "llm")
        except Exception as exc:  # noqa: BLE001 — never fail a turn on classification
            logger.warning("LLM intent classify failed (%s); using rules", exc)
            return self._fallback.classify(message)


def build_intent_classifier(settings: Settings):
    if settings.intent_strategy == "llm" and settings.use_openai:
        return LLMIntentClassifier(settings)
    return RuleIntentClassifier()
