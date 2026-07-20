"""FAQ store + matcher tests."""

from __future__ import annotations

from pathlib import Path

from app.core.embeddings import LocalEmbeddingProvider
from app.core.faq import FAQMatcher, FAQStore


def _store(tmp_path: Path) -> FAQStore:
    return FAQStore(tmp_path / "faqs.db")


def test_add_list_delete(tmp_path: Path):
    store = _store(tmp_path)
    faq = store.add("t", "How do I reset my password?", "Go to settings.", ["auth"])
    assert store.list("t")[0].faq_id == faq.faq_id
    assert store.delete("t", faq.faq_id) is True
    assert store.list("t") == []


def test_tenant_isolation(tmp_path: Path):
    store = _store(tmp_path)
    store.add("a", "Q?", "A")
    assert store.list("a") and store.list("b") == []


def test_matcher_exact_and_paraphrase(tmp_path: Path):
    store = _store(tmp_path)
    emb = LocalEmbeddingProvider()
    store.add("t", "How do I reset my password?", "Reset via settings.")
    store.add("t", "What are your business hours?", "9 to 5.")
    matcher = FAQMatcher(emb, store)

    exact = matcher.match("t", "How do I reset my password?")
    assert exact.faq.question == "How do I reset my password?"
    assert exact.score > 0.8

    para = matcher.match("t", "how can I reset my password")
    assert para.faq.question == "How do I reset my password?"
    assert para.score >= 0.45


def test_matcher_unrelated_scores_low(tmp_path: Path):
    store = _store(tmp_path)
    emb = LocalEmbeddingProvider()
    store.add("t", "How do I reset my password?", "Reset via settings.")
    matcher = FAQMatcher(emb, store)
    m = matcher.match("t", "explain quantum chromodynamics")
    assert m.score < 0.45


def test_matcher_empty_returns_none(tmp_path: Path):
    matcher = FAQMatcher(LocalEmbeddingProvider(), _store(tmp_path))
    assert matcher.match("t", "anything") is None


def test_matcher_cache_invalidation(tmp_path: Path):
    store = _store(tmp_path)
    emb = LocalEmbeddingProvider()
    matcher = FAQMatcher(emb, store)
    assert matcher.match("t", "hello") is None  # builds empty cache
    store.add("t", "How do I reset my password?", "Reset via settings.")
    matcher.invalidate("t")  # required after a write
    assert matcher.match("t", "reset my password") is not None
