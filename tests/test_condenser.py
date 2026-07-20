"""Question-condenser tests (follow-up -> standalone rewriting)."""

from __future__ import annotations

from app.config import Settings
from app.core.condenser import (
    HeuristicCondenser,
    _looks_like_followup,
    build_condenser,
)
from app.core.conversation import Message


def _history() -> list[Message]:
    return [
        Message(0, "user", "How does FAISS index vectors?"),
        Message(1, "assistant", "FAISS builds an index for similarity search. [1]"),
    ]


def test_followup_detection():
    assert _looks_like_followup("what about its latency?")
    assert _looks_like_followup("and the memory usage?")
    assert _looks_like_followup("why?")
    assert not _looks_like_followup(
        "Explain how the ingestion pipeline chunks documents in detail"
    )


def test_heuristic_no_history_returns_question():
    c = HeuristicCondenser()
    assert c.condense([], "what about latency?") == "what about latency?"


def test_heuristic_prepends_prior_question_for_followup():
    c = HeuristicCondenser()
    standalone = c.condense(_history(), "what about its latency?")
    # The prior topic (FAISS/index) is now present for the retriever.
    assert "FAISS" in standalone
    assert "latency" in standalone


def test_heuristic_leaves_standalone_question_untouched():
    c = HeuristicCondenser()
    q = "Explain how BM25 scoring works with term frequency and IDF"
    assert c.condense(_history(), q) == q


def test_factory_offline_is_heuristic():
    assert build_condenser(Settings(openai_api_key=None)).name == "heuristic"


def test_factory_disabled_is_noop():
    c = build_condenser(Settings(chat_condense_question=False))
    assert c.name == "none"
    assert c.condense(_history(), "why?") == "why?"
