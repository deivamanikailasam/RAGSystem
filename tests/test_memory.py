"""Long-term memory store + extraction tests."""

from __future__ import annotations

from pathlib import Path

from app.core.memory import MemoryStore, extract_memories


def test_extract_name():
    assert ("fact", "name is Ada") in extract_memories("Hello, my name is Ada.")


def test_extract_plan_and_tool():
    facts = dict(extract_memories("I'm on the enterprise plan and I use Python."))
    vals = set(facts.values()) if isinstance(facts, dict) else set()
    contents = {c for _, c in extract_memories("I'm on the enterprise plan and I use Python.")}
    assert "plan: enterprise" in contents
    assert "uses Python" in contents


def test_extract_my_x_is_y():
    contents = {c for _, c in extract_memories("my email is ada@example.com")}
    assert "email: ada@example.com" in contents


def test_no_facts_from_plain_question():
    assert extract_memories("How do I reset my password?") == []


def _store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "memory.db")


def test_remember_and_list(tmp_path: Path):
    store = _store(tmp_path)
    store.remember("t", "u1", "fact", "name is Ada")
    assert [m.content for m in store.list("t", "u1")] == ["name is Ada"]


def test_remember_dedupes(tmp_path: Path):
    store = _store(tmp_path)
    store.remember("t", "u1", "fact", "plan: enterprise")
    store.remember("t", "u1", "fact", "plan: enterprise")
    assert len(store.list("t", "u1")) == 1  # deduped, last_seen refreshed


def test_recall_ranks_by_relevance(tmp_path: Path):
    store = _store(tmp_path)
    store.remember("t", "u1", "topic", "asked about refunds")
    store.remember("t", "u1", "topic", "asked about password reset")
    top = store.recall("t", "u1", query="how do I reset my password", limit=1)
    assert top[0].content == "asked about password reset"


def test_recall_recency_without_query(tmp_path: Path):
    store = _store(tmp_path)
    store.remember("t", "u1", "fact", "first")
    store.remember("t", "u1", "fact", "second")
    recent = store.recall("t", "u1", limit=1)
    assert recent[0].content == "second"  # most recent last_seen


def test_forget_user_and_isolation(tmp_path: Path):
    store = _store(tmp_path)
    store.remember("t", "u1", "fact", "x")
    store.remember("t", "u2", "fact", "y")
    assert store.forget_user("t", "u1") == 1
    assert store.list("t", "u1") == []
    assert store.list("t", "u2")  # other user untouched


def test_cross_user_isolation(tmp_path: Path):
    store = _store(tmp_path)
    store.remember("t", "u1", "fact", "secret")
    assert store.list("t", "u2") == []
