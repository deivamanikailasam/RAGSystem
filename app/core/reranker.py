"""Reranking stage — the second stage of two-stage retrieval.

Vector (bi-encoder) search is fast but coarse: it compares a single query
embedding against single chunk embeddings, so it can rank a loosely-related
passage above the one that actually answers the question. A **reranker** takes
the top-N candidates from vector search and re-scores each one *jointly with the
query*, producing a sharper ordering. We then keep the best ``top_n``.

    query ─┬─▶ vector search (fast, recall-oriented) ─▶ N candidates
           └─▶ reranker (slow, precision-oriented)   ─▶ top_n

Four interchangeable strategies, selected by ``RERANK_STRATEGY``:

* ``none``          — identity; keep vector order (baseline).
* ``lexical``       — dependency-free term-overlap (BM25-ish) reranker. Offline,
                      the default, and what the test suite / eval exercise.
* ``cross_encoder`` — a real cross-encoder model via ``sentence-transformers``
                      (optional dependency; see requirements-rerank.txt).
* ``llm``           — an LLM scores each candidate's relevance (needs OpenAI).

``build_reranker`` degrades gracefully: if a strategy's dependency or API key is
missing, it logs a warning and falls back to ``lexical`` so the pipeline always
runs. All rerankers share one interface, so swapping strategies is config-only.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Protocol

from app.config import Settings
from app.core.retriever import RetrievedChunk

logger = logging.getLogger("ragsystem.reranker")

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


class Reranker(Protocol):
    name: str

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        """Return the ``top_n`` candidates, reordered, with ``rerank_score`` set."""
        ...


# --------------------------------------------------------------------------- #
# none — keep vector order
# --------------------------------------------------------------------------- #
class NoOpReranker:
    name = "none"

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        for c in candidates:
            c.rerank_score = c.score
        return candidates[:top_n]


# --------------------------------------------------------------------------- #
# lexical — BM25-style term overlap (offline, no dependencies)
# --------------------------------------------------------------------------- #
class LexicalReranker:
    """BM25-scored reranking over the candidate set.

    BM25 rewards passages that contain the query terms, dampens very frequent
    terms via IDF computed *within the candidate set*, and normalizes for
    passage length. It is a genuinely useful precision booster and needs no
    model or network — ideal as the default and for deterministic tests.
    """

    name = "lexical"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []

        docs = [_tokens(c.record.text) for c in candidates]
        lengths = [len(d) for d in docs]
        avgdl = (sum(lengths) / len(lengths)) or 1.0
        n = len(docs)

        # Document frequency of each query term within the candidate set.
        q_terms = _tokens(query)
        df: dict[str, int] = {}
        for term in set(q_terms):
            df[term] = sum(1 for d in docs if term in d)

        def idf(term: str) -> float:
            # BM25 idf with +1 smoothing (always non-negative).
            return math.log(1 + (n - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5))

        for i, chunk in enumerate(candidates):
            doc = docs[i]
            score = 0.0
            for term in q_terms:
                tf = doc.count(term)
                if tf == 0:
                    continue
                denom = tf + self._k1 * (
                    1 - self._b + self._b * lengths[i] / avgdl
                )
                score += idf(term) * (tf * (self._k1 + 1)) / denom
            chunk.rerank_score = score

        return sorted(candidates, key=lambda c: c.rerank_score or 0.0, reverse=True)[
            :top_n
        ]


# --------------------------------------------------------------------------- #
# cross_encoder — real reranker model (optional dependency)
# --------------------------------------------------------------------------- #
class CrossEncoderReranker:
    """Cross-encoder reranking via sentence-transformers.

    A cross-encoder feeds ``[query, passage]`` through a transformer together
    and outputs a single relevance logit — far more accurate than comparing
    independent embeddings. Requires ``pip install -r requirements-rerank.txt``
    and downloads the model on first use.
    """

    name = "cross_encoder"

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder  # lazy, optional

        self._model = CrossEncoder(model_name)
        self.name = f"cross_encoder:{model_name}"

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        pairs = [(query, c.record.text) for c in candidates]
        scores = self._model.predict(pairs)
        for chunk, score in zip(candidates, scores):
            chunk.rerank_score = float(score)
        return sorted(candidates, key=lambda c: c.rerank_score or 0.0, reverse=True)[
            :top_n
        ]


# --------------------------------------------------------------------------- #
# llm — LLM-as-reranker (needs OpenAI)
# --------------------------------------------------------------------------- #
class LLMReranker:
    """Pointwise LLM reranking: the model scores each passage's relevance 0–10.

    One batched call scores all candidates. On any parse/API failure it falls
    back to the incoming vector order so a reranker hiccup never breaks a query.
    """

    name = "llm"

    def __init__(self, settings: Settings) -> None:
        from openai import OpenAI

        self._client = OpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.openai_timeout_seconds,
            max_retries=settings.openai_max_retries,
        )
        self._model = settings.rerank_llm_model or settings.generation_model
        self.name = f"llm:{self._model}"

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        import json

        listing = "\n".join(
            f"[{i}] {c.record.text[:500]}" for i, c in enumerate(candidates)
        )
        prompt = (
            "Score how well each passage answers the question, 0 (irrelevant) to "
            "10 (fully answers). Return ONLY a JSON object mapping the passage "
            f'index to its score, e.g. {{"0": 8, "1": 2}}.\n\n'
            f"Question: {query}\n\nPassages:\n{listing}"
        )
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            scores = json.loads(resp.choices[0].message.content or "{}")
            for i, chunk in enumerate(candidates):
                chunk.rerank_score = float(scores.get(str(i), 0.0))
        except Exception as exc:  # noqa: BLE001 — never fail the query on rerank
            logger.warning("LLM rerank failed (%s); falling back to vector order", exc)
            for c in candidates:
                c.rerank_score = c.score
            return candidates[:top_n]
        return sorted(candidates, key=lambda c: c.rerank_score or 0.0, reverse=True)[
            :top_n
        ]


# --------------------------------------------------------------------------- #
# factory
# --------------------------------------------------------------------------- #
def build_reranker(settings: Settings) -> Reranker:
    """Construct the configured reranker, falling back to lexical on any gap."""
    strategy = settings.effective_rerank_strategy

    if strategy == "none":
        return NoOpReranker()
    if strategy == "lexical":
        return LexicalReranker()
    if strategy == "cross_encoder":
        try:
            return CrossEncoderReranker(settings.cross_encoder_model)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "cross_encoder reranker unavailable (%s); install "
                "requirements-rerank.txt. Falling back to lexical.",
                exc,
            )
            return LexicalReranker()
    if strategy == "llm":
        if not settings.use_openai:
            logger.warning(
                "llm reranker requires OPENAI_API_KEY; falling back to lexical."
            )
            return LexicalReranker()
        try:
            return LLMReranker(settings)
        except Exception as exc:  # noqa: BLE001
            logger.warning("llm reranker init failed (%s); falling back to lexical.", exc)
            return LexicalReranker()

    logger.warning("Unknown RERANK_STRATEGY '%s'; falling back to lexical.", strategy)
    return LexicalReranker()
