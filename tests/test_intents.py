"""Intent classifier tests (rule-based, offline)."""

from __future__ import annotations

import pytest

from app.config import Settings
from app.core.intents import Intent, RuleIntentClassifier, build_intent_classifier

clf = RuleIntentClassifier()


@pytest.mark.parametrize(
    "message,expected",
    [
        ("hi", Intent.GREETING),
        ("hello there", Intent.GREETING),
        ("bye", Intent.GOODBYE),
        ("thanks, that's all", Intent.GOODBYE),
        ("no more questions", Intent.GOODBYE),
        ("what can you do?", Intent.HELP),
        ("how are you?", Intent.SMALLTALK),
        ("yes", Intent.AFFIRM),
        ("no", Intent.DENY),
        ("How does FAISS index vectors?", Intent.QUESTION),
        ("explain the reranking stage", Intent.QUESTION),
    ],
)
def test_intent_classification(message, expected):
    assert clf.classify(message).intent == expected


def test_question_takes_precedence_over_thanks():
    # A "thanks" that is really a question should classify as QUESTION.
    r = clf.classify("thanks — how does hybrid retrieval work?")
    assert r.intent == Intent.QUESTION


def test_default_is_question_low_confidence():
    r = clf.classify("the ingestion pipeline chunks documents")  # a statement
    assert r.intent == Intent.QUESTION
    assert r.confidence < 0.6


def test_confidence_high_for_explicit_question():
    assert clf.classify("What is BM25?").confidence >= 0.9


def test_slot_extraction_doc_type():
    r = clf.classify("How are refunds handled in the policy docs?")
    assert r.slots.get("doc_type") == "policy"


def test_no_slots_when_absent():
    assert clf.classify("What is FAISS?").slots == {}


def test_factory_offline_is_rule():
    c = build_intent_classifier(Settings(intent_strategy="rule", openai_api_key=None))
    assert c.name == "rule"
    # llm requested but no key -> falls back to rule.
    c2 = build_intent_classifier(Settings(intent_strategy="llm", openai_api_key=None))
    assert c2.name == "rule"
