"""Shared-namespace tenant isolation tests.

All tenants share ONE FAISS index; a query is restricted to the caller's ids via
an IDSelectorBatch. These tests verify: exactly one shared index file exists,
searches never leak across tenants, ids are globally unique, and delete/purge
remove only the target tenant's vectors from the shared index.
"""

from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.core.rag import RagEngine

FAISS_DOC = "FAISS performs similarity search over dense embedding vectors and chunks."
BREAD_DOC = "Sourdough bread needs flour water salt and a starter fermented overnight."


def _engine(tmp_path: Path) -> RagEngine:
    return RagEngine(
        Settings(
            deployment_mode="multi_tenant",
            tenant_isolation="shared_namespace",
            openai_api_key=None,
            data_dir=tmp_path / "data",
            chunk_tokens=40,
            chunk_overlap=8,
            api_keys="",
            rerank_strategy="none",
        )
    )


def test_single_shared_index_file(tmp_path: Path):
    engine = _engine(tmp_path)
    engine.tenants.create(tenant_id="a")
    engine.tenants.create(tenant_id="b")
    engine.ingest(tenant="a", text=FAISS_DOC, source="f.md", doc_id="fa")
    engine.ingest(tenant="b", text=BREAD_DOC, source="b.md", doc_id="br")

    data = tmp_path / "data"
    # One shared index, NO per-tenant index files.
    assert (data / "shared" / "index.faiss").exists()
    assert not (data / "tenants").exists()


def test_no_cross_tenant_leakage(tmp_path: Path):
    engine = _engine(tmp_path)
    engine.ingest(tenant="a", text=FAISS_DOC, source="f.md", doc_id="fa")
    engine.ingest(tenant="b", text=BREAD_DOC, source="b.md", doc_id="br")

    # Tenant B asks about FAISS (only tenant A has it) -> no results.
    resp_b = engine.query(tenant="b", question="How does FAISS chunk vectors?")
    assert all(c.doc_id != "fa" for c in resp_b.citations)

    # Tenant A gets its own doc.
    resp_a = engine.query(tenant="a", question="How does FAISS chunk vectors?")
    assert resp_a.citations and resp_a.citations[0].doc_id == "fa"


def test_globally_unique_vector_ids(tmp_path: Path):
    engine = _engine(tmp_path)
    engine.ingest(tenant="a", text=FAISS_DOC, source="f.md", doc_id="fa")
    engine.ingest(tenant="b", text=BREAD_DOC, source="b.md", doc_id="br")
    ids_a = set(engine.docstore.vector_ids_for_tenant("a"))
    ids_b = set(engine.docstore.vector_ids_for_tenant("b"))
    assert ids_a and ids_b
    assert ids_a.isdisjoint(ids_b)  # no id collisions in the shared index


def test_ntotal_is_per_tenant(tmp_path: Path):
    engine = _engine(tmp_path)
    engine.ingest(tenant="a", text=FAISS_DOC, source="f.md", doc_id="fa")
    engine.ingest(tenant="b", text=BREAD_DOC, source="b.md", doc_id="br")
    # Each namespace view reports only its own vectors.
    n_a = engine.vector_store.for_tenant("a").ntotal
    n_b = engine.vector_store.for_tenant("b").ntotal
    assert n_a >= 1 and n_b >= 1
    assert engine.vector_store._shared.ntotal == n_a + n_b  # noqa: SLF001


def test_delete_document_removes_from_shared_index(tmp_path: Path):
    engine = _engine(tmp_path)
    engine.ingest(tenant="a", text=FAISS_DOC, source="f.md", doc_id="fa")
    engine.ingest(tenant="b", text=BREAD_DOC, source="b.md", doc_id="br")
    shared_before = engine.vector_store._shared.ntotal  # noqa: SLF001
    removed = engine.delete_document(tenant="a", doc_id="fa")
    assert removed >= 1
    assert engine.vector_store._shared.ntotal == shared_before - removed  # noqa: SLF001
    # Tenant B is untouched.
    assert engine.vector_store.for_tenant("b").ntotal >= 1


def test_purge_tenant_removes_only_that_namespace(tmp_path: Path):
    engine = _engine(tmp_path)
    engine.ingest(tenant="a", text=FAISS_DOC, source="f.md", doc_id="fa")
    engine.ingest(tenant="b", text=BREAD_DOC, source="b.md", doc_id="br")
    engine.purge_tenant("a")
    assert engine.vector_store.for_tenant("a").ntotal == 0
    assert engine.vector_store.for_tenant("b").ntotal >= 1
    # Tenant B still queryable.
    resp = engine.query(tenant="b", question="how to bake sourdough bread?")
    assert resp.citations and resp.citations[0].doc_id == "br"


def test_reingest_update_in_shared_index(tmp_path: Path):
    engine = _engine(tmp_path)
    engine.ingest(tenant="a", text="old content about apples", source="s", doc_id="d")
    engine.ingest(tenant="a", text="new content about oranges", source="s", doc_id="d")
    resp = engine.query(tenant="a", question="what fruit is discussed?")
    joined = " ".join(c.snippet.lower() for c in resp.citations)
    assert "apples" not in joined


def test_persistence_roundtrip_shared(tmp_path: Path):
    engine = _engine(tmp_path)
    engine.ingest(tenant="a", text=FAISS_DOC, source="f.md", doc_id="fa")
    # Fresh engine loads the shared index from disk.
    engine2 = _engine(tmp_path)
    resp = engine2.query(tenant="a", question="FAISS similarity search")
    assert resp.citations and resp.citations[0].doc_id == "fa"
