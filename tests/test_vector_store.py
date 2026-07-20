import numpy as np

from app.core.embeddings import LocalEmbeddingProvider
from app.core.vector_store import VectorStore


def _store(tmp_path, dim=256):
    return VectorStore(
        data_dir=tmp_path, dimension=dim, index_type="flat", nlist=10, nprobe=4
    )


def test_add_search_returns_nearest(tmp_path):
    emb = LocalEmbeddingProvider()
    store = _store(tmp_path, emb.dimension)
    idx = store.for_tenant("t1")

    texts = ["faiss vector search", "banana bread recipe", "python type hints"]
    vecs = emb.embed(texts)
    ids = np.arange(len(texts), dtype=np.int64)
    idx.add(vecs, ids)

    q = emb.embed(["vector search with faiss"])[0]
    hits = idx.search(q, top_k=3)
    assert hits
    assert hits[0].vector_id == 0  # closest to the faiss text


def test_remove_ids(tmp_path):
    emb = LocalEmbeddingProvider()
    store = _store(tmp_path, emb.dimension)
    idx = store.for_tenant("t1")
    vecs = emb.embed(["a b c", "d e f"])
    idx.add(vecs, np.array([10, 11], dtype=np.int64))
    assert idx.ntotal == 2
    removed = idx.remove([10])
    assert removed == 1
    assert idx.ntotal == 1


def test_persistence_roundtrip(tmp_path):
    emb = LocalEmbeddingProvider()
    store = _store(tmp_path, emb.dimension)
    idx = store.for_tenant("t1")
    idx.add(emb.embed(["persisted text"]), np.array([7], dtype=np.int64))
    idx.persist()

    # Fresh store instance loads from disk.
    store2 = _store(tmp_path, emb.dimension)
    idx2 = store2.for_tenant("t1")
    assert idx2.ntotal == 1


def test_tenant_isolation(tmp_path):
    emb = LocalEmbeddingProvider()
    store = _store(tmp_path, emb.dimension)
    store.for_tenant("a").add(emb.embed(["alpha"]), np.array([1], dtype=np.int64))
    assert store.for_tenant("a").ntotal == 1
    assert store.for_tenant("b").ntotal == 0
