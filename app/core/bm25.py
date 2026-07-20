"""BM25 sparse retrieval over the full corpus (per tenant).

This is the *sparse* half of hybrid retrieval. Unlike the lexical **reranker**
(which BM25-scores only the candidates vector search already found), this BM25
index covers the **entire tenant corpus**, so it can surface documents that
dense vector search missed — exact keyword matches, rare terms, IDs, product
names, code symbols. Fusing the two (see :mod:`app.core.fusion`) gives both
semantic recall (dense) and lexical precision (sparse).

Implementation notes
--------------------
* An **inverted index** (``term -> [(doc_idx, term_freq)]``) so a search only
  touches documents that contain a query term — not the whole corpus.
* Standard **Okapi BM25** scoring with length normalization.
* Built lazily from the SQLite docstore (the source of truth) and cached per
  tenant; the cache is invalidated on ingest/delete so it never goes stale.
  For very large corpora you would persist the index and update it
  incrementally — see docs/09-hybrid-retrieval.md §Scaling.
"""

from __future__ import annotations

import math
import re
import threading
from collections import Counter, defaultdict

from app.core.docstore import DocStore

_WORD_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


class BM25Index:
    """An immutable Okapi BM25 index over a fixed set of documents."""

    def __init__(
        self,
        vector_ids: list[int],
        docs_tokens: list[list[str]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._k1 = k1
        self._b = b
        self._vector_ids = vector_ids
        self.n = len(vector_ids)
        self._doc_len = [len(t) for t in docs_tokens]
        self._avgdl = (sum(self._doc_len) / self.n) if self.n else 0.0

        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._df: Counter[str] = Counter()
        for idx, tokens in enumerate(docs_tokens):
            counts = Counter(tokens)
            for term, freq in counts.items():
                self._postings[term].append((idx, freq))
                self._df[term] += 1

    def _idf(self, term: str) -> float:
        n_t = self._df.get(term, 0)
        # BM25 idf with +1 smoothing so it is always non-negative.
        return math.log(1 + (self.n - n_t + 0.5) / (n_t + 0.5))

    def search(self, query: str, limit: int) -> list[tuple[int, float]]:
        """Return up to ``limit`` ``(vector_id, bm25_score)`` pairs, best first."""
        if self.n == 0:
            return []
        scores: dict[int, float] = defaultdict(float)
        for term in tokenize(query):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf(term)
            for idx, freq in postings:
                dl = self._doc_len[idx]
                denom = freq + self._k1 * (
                    1 - self._b + self._b * dl / (self._avgdl or 1.0)
                )
                scores[idx] += idf * (freq * (self._k1 + 1)) / denom
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:limit]
        return [(self._vector_ids[idx], score) for idx, score in ranked]


class BM25Store:
    """Per-tenant lazy cache of BM25 indices built from the docstore."""

    def __init__(self, docstore: DocStore, k1: float = 1.5, b: float = 0.75) -> None:
        self._docs = docstore
        self._k1 = k1
        self._b = b
        self._cache: dict[str, BM25Index] = {}
        self._lock = threading.Lock()

    def for_tenant(self, tenant: str) -> BM25Index:
        with self._lock:
            index = self._cache.get(tenant)
            if index is None:
                records = self._docs.all_chunks(tenant)
                index = BM25Index(
                    vector_ids=[r.vector_id for r in records],
                    docs_tokens=[tokenize(r.text) for r in records],
                    k1=self._k1,
                    b=self._b,
                )
                self._cache[tenant] = index
            return index

    def invalidate(self, tenant: str) -> None:
        """Drop the cached index so it rebuilds on next use (after a write)."""
        with self._lock:
            self._cache.pop(tenant, None)
