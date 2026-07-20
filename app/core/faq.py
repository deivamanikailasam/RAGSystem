"""FAQ knowledge base + matcher.

An FAQ bot answers from a curated set of question→answer pairs *before* falling
back to open RAG over documents: curated answers are exact, fast (no LLM), and
authoritative. This module stores the FAQs (per tenant) and matches an incoming
question against them.

Matching combines two signals so it is robust offline and strong with real
embeddings:

* **semantic** — cosine similarity between the query embedding and each FAQ
  question embedding (catches paraphrases: "reset password" ≈ "change my
  password");
* **lexical** — Jaccard overlap of content words (catches exact keyword hits and
  keeps the offline fallback embedder honest).

The per-tenant FAQ embedding index is cached and rebuilt when FAQs change.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an the of to in on for and or is are was were be do does did how what "
    "why when where which who i you my your me it its can could should would "
    "with as by at from this that".split()
)


def _content_tokens(text: str) -> set[str]:
    return {t for t in _WORD_RE.findall(text.lower()) if t not in _STOPWORDS}


@dataclass
class FAQ:
    faq_id: str
    question: str
    answer: str
    tags: list[str] = field(default_factory=list)
    created_at: float = 0.0


@dataclass
class FAQMatch:
    faq: FAQ
    score: float
    cosine: float
    lexical: float


_SCHEMA = """
CREATE TABLE IF NOT EXISTS faqs (
    tenant     TEXT NOT NULL,
    faq_id     TEXT NOT NULL,
    question   TEXT NOT NULL,
    answer     TEXT NOT NULL,
    tags       TEXT NOT NULL DEFAULT '[]',
    created_at REAL NOT NULL,
    PRIMARY KEY (tenant, faq_id)
);
"""


class FAQStore:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def add(self, tenant: str, question: str, answer: str,
            tags: list[str] | None = None, faq_id: str | None = None) -> FAQ:
        faq = FAQ(
            faq_id=faq_id or uuid.uuid4().hex[:12],
            question=question, answer=answer, tags=tags or [],
            created_at=time.time(),
        )
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO faqs
                       (tenant, faq_id, question, answer, tags, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (tenant, faq.faq_id, faq.question, faq.answer,
                 json.dumps(faq.tags), faq.created_at),
            )
            self._conn.commit()
        return faq

    def list(self, tenant: str) -> list[FAQ]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM faqs WHERE tenant=? ORDER BY created_at", (tenant,)
            ).fetchall()
        return [self._row_to_faq(r) for r in rows]

    def delete(self, tenant: str, faq_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM faqs WHERE tenant=? AND faq_id=?", (tenant, faq_id)
            )
            self._conn.commit()
            return cur.rowcount > 0

    def delete_tenant(self, tenant: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM faqs WHERE tenant=?", (tenant,))
            self._conn.commit()

    @staticmethod
    def _row_to_faq(row: sqlite3.Row) -> FAQ:
        return FAQ(
            faq_id=row["faq_id"], question=row["question"], answer=row["answer"],
            tags=json.loads(row["tags"]), created_at=float(row["created_at"]),
        )


class FAQMatcher:
    """Matches a query against a tenant's FAQs (semantic + lexical)."""

    def __init__(self, embeddings, store: FAQStore,
                 semantic_weight: float = 0.5) -> None:
        self._embeddings = embeddings
        self._store = store
        self._w = semantic_weight
        self._cache: dict[str, tuple[list[FAQ], np.ndarray, list[set[str]]]] = {}
        self._lock = threading.Lock()

    def invalidate(self, tenant: str) -> None:
        with self._lock:
            self._cache.pop(tenant, None)

    def _index(self, tenant: str):
        with self._lock:
            cached = self._cache.get(tenant)
            if cached is None:
                faqs = self._store.list(tenant)
                if faqs:
                    matrix = self._embeddings.embed([f.question for f in faqs])
                    tokens = [_content_tokens(f.question) for f in faqs]
                else:
                    matrix = np.zeros((0, self._embeddings.dimension), dtype=np.float32)
                    tokens = []
                cached = (faqs, matrix, tokens)
                self._cache[tenant] = cached
            return cached

    def match(self, tenant: str, query: str) -> FAQMatch | None:
        faqs, matrix, tokens = self._index(tenant)
        if not faqs:
            return None
        qv = self._embeddings.embed([query])[0]
        cosines = matrix @ qv  # normalized vectors -> cosine
        q_tokens = _content_tokens(query)

        best: FAQMatch | None = None
        for i, faq in enumerate(faqs):
            cos = float(cosines[i])
            union = q_tokens | tokens[i]
            lex = len(q_tokens & tokens[i]) / len(union) if union else 0.0
            score = self._w * cos + (1 - self._w) * lex
            if best is None or score > best.score:
                best = FAQMatch(faq=faq, score=score, cosine=cos, lexical=lex)
        return best
