"""Embedding provider abstraction.

Two implementations share one interface:

* :class:`OpenAIEmbeddingProvider` — calls the OpenAI embeddings API in
  batches with exponential-backoff retries. Used when ``OPENAI_API_KEY`` is set.
* :class:`LocalEmbeddingProvider` — a deterministic, dependency-free hashing
  embedder used as a fallback so the whole system runs (and is testable)
  offline. It is **not** semantically strong, but it is stable and fast, which
  is exactly what unit tests and local demos need.

All vectors are L2-normalized so that inner product == cosine similarity, which
is what the FAISS ``IndexFlatIP`` / ``IndexIVFFlat`` layers below assume.
"""

from __future__ import annotations

import hashlib
import time
from typing import Protocol

import numpy as np

from app.config import Settings


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization; zero rows are left as zeros."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class EmbeddingProvider(Protocol):
    dimension: int
    model_name: str

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an ``(len(texts), dimension)`` float32 matrix of unit vectors."""
        ...


class LocalEmbeddingProvider:
    """Deterministic offline embedder.

    Maps text to a fixed-dimension vector by hashing both whole words and their
    character trigrams into a signed bag-of-features. The subword features let
    morphological variants (``chunking`` ↔ ``chunks``) land near each other, so
    two texts sharing vocabulary or word stems are close in vector space. This
    is enough for the pipeline to function and for tests to assert retrieval
    behavior deterministically. It is a *fallback*, not a semantic model — set
    ``OPENAI_API_KEY`` for real embeddings.
    """

    _STOPWORDS = frozenset(
        "a an the of to in on for and or is are was were be been it its this that "
        "with as by at from how does do you your we our".split()
    )

    def __init__(self, dimension: int = 512) -> None:
        self.dimension = dimension
        self.model_name = f"local-hash-{dimension}"

    def _add_feature(self, vec: np.ndarray, token: str, weight: float) -> None:
        h = hashlib.sha1(token.encode("utf-8")).digest()
        idx = int.from_bytes(h[:4], "little") % self.dimension
        sign = 1.0 if h[4] % 2 == 0 else -1.0
        vec[idx] += sign * weight

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        for raw in text.lower().split():
            token = raw.strip(".,!?;:()[]{}\"'`")
            if not token or token in self._STOPWORDS:
                continue
            # Whole-word feature (weighted higher than subword features).
            self._add_feature(vec, f"w:{token}", 2.0)
            # Character-trigram features for morphological / typo robustness.
            padded = f"^{token}$"
            for i in range(len(padded) - 2):
                self._add_feature(vec, f"g:{padded[i:i + 3]}", 1.0)
        return vec

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        matrix = np.vstack([self._embed_one(t) for t in texts]).astype(np.float32)
        return l2_normalize(matrix)


class OpenAIEmbeddingProvider:
    """OpenAI embeddings with batching + exponential-backoff retries."""

    # Native dimensions of the common OpenAI embedding models.
    _MODEL_DIMS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, settings: Settings) -> None:
        # Imported lazily so the package need not be present offline.
        from openai import OpenAI

        self._client = OpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.openai_timeout_seconds,
            max_retries=0,  # we implement our own backoff for full control
        )
        self.model_name = settings.embedding_model
        self.dimension = self._MODEL_DIMS.get(self.model_name, 1536)
        self._max_retries = settings.openai_max_retries
        self._batch_size = 128

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            vectors.extend(self._embed_batch_with_retry(batch))

        matrix = np.asarray(vectors, dtype=np.float32)
        return l2_normalize(matrix)

    def _embed_batch_with_retry(self, batch: list[str]) -> list[list[float]]:
        from openai import APIError, APITimeoutError, RateLimitError

        attempt = 0
        while True:
            try:
                resp = self._client.embeddings.create(
                    model=self.model_name, input=batch
                )
                # Preserve input order.
                return [item.embedding for item in resp.data]
            except (RateLimitError, APITimeoutError, APIError) as exc:
                attempt += 1
                if attempt > self._max_retries:
                    raise
                sleep_s = min(2**attempt, 30)
                time.sleep(sleep_s)
                _ = exc  # retried


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Factory: real OpenAI when configured, deterministic local otherwise."""
    if settings.use_openai:
        return OpenAIEmbeddingProvider(settings)
    return LocalEmbeddingProvider()
