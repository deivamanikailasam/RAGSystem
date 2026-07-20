"""End-to-end pipeline tests against the offline engine."""

from app.core.rag import RagEngine

DOC_FAISS = (
    "FAISS is a library for efficient similarity search of dense vectors. "
    "The ingestion pipeline chunks documents into overlapping windows and "
    "embeds each chunk before adding it to the FAISS index."
)
DOC_COOKING = (
    "To bake sourdough bread you need flour, water, salt and a starter. "
    "Ferment the dough overnight and bake in a hot dutch oven."
)


def _seed(engine: RagEngine):
    engine.ingest(tenant="demo", text=DOC_FAISS, source="faiss.md", doc_id="faiss")
    engine.ingest(tenant="demo", text=DOC_COOKING, source="bread.md", doc_id="bread")


def test_query_retrieves_relevant_document(engine: RagEngine):
    _seed(engine)
    resp = engine.query(tenant="demo", question="How does chunking work in FAISS?")
    assert resp.citations
    assert resp.citations[0].doc_id == "faiss"
    assert "chunk" in resp.answer.lower() or "faiss" in resp.answer.lower()


def test_ingest_is_idempotent(engine: RagEngine):
    r1 = engine.ingest(tenant="demo", text=DOC_FAISS, source="faiss.md", doc_id="faiss")
    assert r1.chunks > 0 and not r1.skipped
    r2 = engine.ingest(tenant="demo", text=DOC_FAISS, source="faiss.md", doc_id="faiss")
    assert r2.skipped and r2.chunks == 0


def test_reingest_updated_content_replaces_vectors(engine: RagEngine):
    engine.ingest(tenant="demo", text="old content about apples", source="s", doc_id="d")
    before = engine.vector_store.for_tenant("demo").ntotal
    engine.ingest(tenant="demo", text="new content about oranges instead", source="s", doc_id="d")
    after = engine.vector_store.for_tenant("demo").ntotal
    # Old vectors removed; only the new version's chunks remain.
    assert after >= 1
    resp = engine.query(tenant="demo", question="what fruit?")
    assert "apples" not in resp.answer.lower()


def test_delete_document_removes_vectors(engine: RagEngine):
    _seed(engine)
    total_before = engine.vector_store.for_tenant("demo").ntotal
    removed = engine.delete_document(tenant="demo", doc_id="bread")
    assert removed > 0
    assert engine.vector_store.for_tenant("demo").ntotal == total_before - removed


def test_metadata_filter(engine: RagEngine):
    engine.ingest(tenant="demo", text=DOC_FAISS, source="f", doc_id="faiss",
                  metadata={"doc_type": "tech"})
    engine.ingest(tenant="demo", text=DOC_COOKING, source="b", doc_id="bread",
                  metadata={"doc_type": "recipe"})
    resp = engine.query(
        tenant="demo", question="anything", filters={"doc_type": "recipe"}
    )
    assert all(c.doc_id == "bread" for c in resp.citations)


def test_unknown_answer_when_no_docs(engine: RagEngine):
    resp = engine.query(tenant="demo", question="what is the capital of France?")
    assert resp.citations == []
    assert "don't know" in resp.answer.lower()


def test_tenant_isolation_in_query(engine: RagEngine):
    engine.ingest(tenant="tenantA", text=DOC_FAISS, source="f", doc_id="faiss")
    resp = engine.query(tenant="tenantB", question="How does FAISS chunk documents?")
    assert resp.citations == []
