"""FAISS-backed vector store with two tenant-isolation strategies.

Isolation strategies (``TENANT_ISOLATION``; see docs/10-tenant-isolation.md):

* ``index_per_tenant`` (default) — **physical** isolation: each tenant gets its
  own FAISS index file. Hardest data boundary, per-tenant index type, cheap
  whole-tenant drop. Cost: one index/file per tenant.
* ``shared_namespace`` — one shared FAISS index holds every tenant's vectors,
  partitioned by a namespace = the tenant id. A query is restricted to the
  caller's ids with a FAISS ``IDSelectorBatch`` so results are **exact** for
  that namespace (no post-filtering, no small-namespace starvation) and never
  leak across tenants. Best when you have many small tenants and per-file
  overhead dominates.

Common design points:
* **Inner product on normalized vectors == cosine similarity.**
* **``IndexIDMap2``** gives stable integer ids (shared with the SQLite docstore)
  and supports ``remove_ids`` for deletion / re-ingestion.
* **Flat vs IVF** trades exactness for scale.

Persistence layout::

    {DATA_DIR}/tenants/{t}/index.faiss     # index_per_tenant: one file per tenant
    {DATA_DIR}/shared/index.faiss          # shared_namespace: one shared file
    {DATA_DIR}/docstore.db                 # SQLite metadata (all tenants, row-scoped)
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import faiss
import numpy as np

if TYPE_CHECKING:
    from app.core.docstore import DocStore


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
    def search(
        self, query: np.ndarray, top_k: int, selector: "faiss.IDSelector | None" = None
    ) -> list[SearchHit]:
        """Search, optionally restricted to a subset of ids via ``selector``.

        The selector (used by shared-namespace isolation) makes FAISS score
        *only* the given ids, so results are exact for that subset — no
        post-filtering, no small-namespace starvation.
        """
        with self._lock:
            if self._index.ntotal == 0:
                return []
            query = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
            k = min(top_k, self._index.ntotal)
            if selector is not None:
                scores, ids = self._index.search(
                    query, k, params=self._search_params(selector)
                )
            else:
                scores, ids = self._index.search(query, k)
        hits: list[SearchHit] = []
        for score, vid in zip(scores[0], ids[0]):
            if vid == -1:
                continue
            hits.append(SearchHit(vector_id=int(vid), score=float(score)))
        return hits

    def _search_params(self, selector: "faiss.IDSelector") -> faiss.SearchParameters:
        if self.index_type == "ivf":
            params = faiss.SearchParametersIVF()
            params.nprobe = self._nprobe
        else:
            params = faiss.SearchParameters()
        params.sel = selector
        return params

    @property
    def ntotal(self) -> int:
        with self._lock:
            return int(self._index.ntotal)


class SharedNamespaceView:
    """A per-tenant *view* over the single shared index.

    Presents the same ``add / search / remove / persist / ntotal`` surface as
    :class:`TenantIndex`, but scopes every operation to one namespace (tenant):
    searches are restricted to the tenant's ids via an ``IDSelectorBatch``, and
    ``ntotal`` counts only that tenant's vectors.
    """

    def __init__(self, store: "VectorStore", tenant: str) -> None:
        self._store = store
        self._tenant = tenant
        self._shared = store._shared  # noqa: SLF001

    def add(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        self._shared.add(vectors, ids)
        self._store._invalidate_ids(self._tenant)  # noqa: SLF001

    def remove(self, ids: list[int]) -> int:
        removed = self._shared.remove(ids)
        self._store._invalidate_ids(self._tenant)  # noqa: SLF001
        return removed

    def search(self, query: np.ndarray, top_k: int) -> list[SearchHit]:
        ids = self._store._tenant_ids(self._tenant)  # noqa: SLF001
        if ids.size == 0:
            return []
        selector = faiss.IDSelectorBatch(ids)
        return self._shared.search(query, top_k, selector=selector)

    def persist(self) -> None:
        self._shared.persist()

    @property
    def ntotal(self) -> int:
        return int(self._store._tenant_ids(self._tenant).size)  # noqa: SLF001


class VectorStore:
    """Routes tenant vector operations to the configured isolation strategy."""

    def __init__(self, data_dir: Path, dimension: int, index_type: str,
                 nlist: int, nprobe: int, isolation: str = "index_per_tenant",
                 docstore: "DocStore | None" = None) -> None:
        self._data_dir = data_dir
        self._dimension = dimension
        self._index_type = index_type
        self._nlist = nlist
        self._nprobe = nprobe
        self._isolation = isolation
        self._docs = docstore
        self._cache: dict[str, TenantIndex] = {}
        self._cache_lock = threading.Lock()

        # shared_namespace: one index for everyone + a per-tenant id cache used
        # to build search selectors.
        self._shared: TenantIndex | None = None
        self._id_cache: dict[str, np.ndarray] = {}
        if isolation == "shared_namespace":
            if docstore is None:
                raise ValueError("shared_namespace isolation requires a docstore")
            self._shared = TenantIndex(
                data_dir / "shared" / "index.faiss",
                dimension, index_type, nlist, nprobe,
            )

    def _tenant_path(self, tenant: str) -> Path:
        return self._data_dir / "tenants" / tenant / "index.faiss"

    # -- shared-namespace id bookkeeping ----------------------------------- #
    def _tenant_ids(self, tenant: str) -> np.ndarray:
        """Cached int64 array of a tenant's vector ids (from the docstore)."""
        with self._cache_lock:
            arr = self._id_cache.get(tenant)
            if arr is None:
                assert self._docs is not None
                ids = self._docs.vector_ids_for_tenant(tenant)
                arr = np.asarray(ids, dtype=np.int64)
                self._id_cache[tenant] = arr
            return arr

    def _invalidate_ids(self, tenant: str) -> None:
        with self._cache_lock:
            self._id_cache.pop(tenant, None)

    # -- public API -------------------------------------------------------- #
    def for_tenant(self, tenant: str, index_type: str | None = None):
        """Return a per-tenant index or a namespace view over the shared index.

        ``index_type`` overrides the global default the first time a tenant's
        own index is created (``index_per_tenant`` only; ignored when sharing).
        """
        if self._isolation == "shared_namespace":
            return SharedNamespaceView(self, tenant)

        with self._cache_lock:
            idx = self._cache.get(tenant)
            if idx is None:
                idx = TenantIndex(
                    self._tenant_path(tenant),
                    self._dimension,
                    index_type or self._index_type,
                    self._nlist,
                    self._nprobe,
                )
                self._cache[tenant] = idx
            return idx

    def drop_tenant(self, tenant: str) -> None:
        """Remove a tenant's vectors entirely.

        Must be called *before* the tenant's docstore rows are deleted, since
        shared-namespace mode reads the tenant's ids from the docstore to know
        which vectors to remove from the shared index.
        """
        if self._isolation == "shared_namespace":
            assert self._docs is not None and self._shared is not None
            ids = self._docs.vector_ids_for_tenant(tenant)
            if ids:
                self._shared.remove(ids)
                self._shared.persist()
            self._invalidate_ids(tenant)
            return

        with self._cache_lock:
            self._cache.pop(tenant, None)
        path = self._tenant_path(tenant)
        if path.exists():
            path.unlink()
        # Best-effort cleanup of the now-empty tenant directory.
        parent = path.parent
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
