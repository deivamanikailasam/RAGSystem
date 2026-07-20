"""FAISS-backed vector store with per-tenant isolation and persistence.

Design decisions
----------------
* **One index per tenant.** Keeps per-index size manageable and gives hard data
  isolation between tenants (see docs/06-scaling-and-evolution.md). Indices are
  loaded lazily and cached in memory.
* **Inner product on normalized vectors == cosine similarity.** Embedding
  providers return unit vectors, so ``IndexFlatIP`` / ``IndexIVFFlat`` give
  cosine scores directly.
* **``IndexIDMap2`` wrapper** lets us assign our own stable integer ids (shared
  with the SQLite docstore) and, crucially, supports ``remove_ids`` so document
  deletion / re-ingestion works.
* **Flat vs IVF.** ``flat`` is exact and ideal up to ~100k vectors/tenant.
  ``ivf`` is approximate (trained on the data) for larger corpora; tune
  ``nlist`` / ``nprobe`` for the recall/latency trade-off.

Persistence layout (per tenant ``t``)::

    {DATA_DIR}/tenants/{t}/index.faiss     # the FAISS index
    {DATA_DIR}/docstore.db                 # shared SQLite metadata (all tenants)
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np


@dataclass
class SearchHit:
    vector_id: int
    score: float


class TenantIndex:
    """Wraps a single tenant's FAISS index plus its on-disk path."""

    def __init__(self, path: Path, dimension: int, index_type: str,
                 nlist: int, nprobe: int) -> None:
        self.path = path
        self.dimension = dimension
        self.index_type = index_type
        self._nlist = nlist
        self._nprobe = nprobe
        self._lock = threading.Lock()
        self._index = self._load_or_create()

    # -- lifecycle --------------------------------------------------------- #
    def _new_index(self) -> faiss.Index:
        if self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self.dimension)
            base = faiss.IndexIVFFlat(
                quantizer, self.dimension, self._nlist, faiss.METRIC_INNER_PRODUCT
            )
            base.nprobe = self._nprobe
        else:
            base = faiss.IndexFlatIP(self.dimension)
        return faiss.IndexIDMap2(base)

    def _load_or_create(self) -> faiss.Index:
        if self.path.exists():
            return faiss.read_index(str(self.path))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return self._new_index()

    def persist(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, str(self.path))

    # -- write ------------------------------------------------------------- #
    def add(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        if vectors.shape[0] == 0:
            return
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        ids = np.ascontiguousarray(ids, dtype=np.int64)
        with self._lock:
            self._maybe_train(vectors)
            self._index.add_with_ids(vectors, ids)

    def _maybe_train(self, vectors: np.ndarray) -> None:
        """IVF indices must be trained before the first add."""
        if self.index_type != "ivf":
            return
        # IndexIDMap2 -> underlying IVF index.
        inner = faiss.downcast_index(self._index.index)
        if not inner.is_trained:
            # Need at least nlist points to train meaningfully; if fewer, clamp.
            if vectors.shape[0] < self._nlist:
                inner.nlist = max(1, vectors.shape[0])
            inner.train(vectors)

    def remove(self, ids: list[int]) -> int:
        if not ids:
            return 0
        selector = np.asarray(ids, dtype=np.int64)
        with self._lock:
            return int(self._index.remove_ids(selector))

    # -- read -------------------------------------------------------------- #
    def search(self, query: np.ndarray, top_k: int) -> list[SearchHit]:
        with self._lock:
            if self._index.ntotal == 0:
                return []
            query = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
            k = min(top_k, self._index.ntotal)
            scores, ids = self._index.search(query, k)
        hits: list[SearchHit] = []
        for score, vid in zip(scores[0], ids[0]):
            if vid == -1:
                continue
            hits.append(SearchHit(vector_id=int(vid), score=float(score)))
        return hits

    @property
    def ntotal(self) -> int:
        with self._lock:
            return int(self._index.ntotal)


class VectorStore:
    """Manages a cache of per-tenant FAISS indices."""

    def __init__(self, data_dir: Path, dimension: int, index_type: str,
                 nlist: int, nprobe: int) -> None:
        self._data_dir = data_dir
        self._dimension = dimension
        self._index_type = index_type
        self._nlist = nlist
        self._nprobe = nprobe
        self._cache: dict[str, TenantIndex] = {}
        self._cache_lock = threading.Lock()

    def _tenant_path(self, tenant: str) -> Path:
        return self._data_dir / "tenants" / tenant / "index.faiss"

    def for_tenant(self, tenant: str) -> TenantIndex:
        with self._cache_lock:
            idx = self._cache.get(tenant)
            if idx is None:
                idx = TenantIndex(
                    self._tenant_path(tenant),
                    self._dimension,
                    self._index_type,
                    self._nlist,
                    self._nprobe,
                )
                self._cache[tenant] = idx
            return idx
