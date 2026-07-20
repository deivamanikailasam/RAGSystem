"""Smoke test for the retrieval evaluation harness and engine.retrieve()."""

from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.core.rag import RagEngine
from eval.metrics import evaluate_queries
from eval.retrieval_eval import TENANT, load_jsonl, ranked_doc_ids


def test_engine_retrieve_returns_ranked_chunks(tmp_path: Path):
    engine = RagEngine(
        Settings(
            deployment_mode="single_tenant",
            openai_api_key=None,
            data_dir=tmp_path / "d",
            retrieval_top_k=5,
        )
    )
    engine.ingest(tenant="default", text="FAISS indexes dense vectors for search.",
                  source="a.md", doc_id="a")
    engine.ingest(tenant="default", text="Sourdough bread needs flour and starter.",
                  source="b.md", doc_id="b")
    chunks = engine.retrieve(tenant="default", question="vector search with FAISS")
    assert chunks
    assert chunks[0].record.doc_id == "a"


def test_end_to_end_metrics_on_labeled_set(tmp_path: Path):
    """Run the real labeled corpus/dataset and assert sane metric bounds."""
    eval_dir = Path(__file__).resolve().parent.parent / "eval"
    corpus = load_jsonl(eval_dir / "retrieval_corpus.jsonl")
    cases = load_jsonl(eval_dir / "retrieval_dataset.jsonl")

    engine = RagEngine(
        Settings(
            deployment_mode="single_tenant",
            openai_api_key=None,
            data_dir=tmp_path / "d",
            retrieval_top_k=10,
            max_context_chunks=10,
            retrieval_mode="hybrid",
            rerank_strategy="lexical",
        )
    )
    # ranked_doc_ids queries the harness tenant (TENANT), so ingest into it.
    for doc in corpus:
        engine.ingest(tenant=TENANT, text=doc["text"], source=doc["source"],
                      doc_id=doc["doc_id"])

    pairs = [
        (ranked_doc_ids(engine, c["question"]), set(c["relevant_doc_ids"]))
        for c in cases
    ]
    metrics = evaluate_queries(pairs, ks=(1, 3, 5))

    # All metrics are valid fractions.
    assert all(0.0 <= v <= 1.0 for v in metrics.values())
    # The labeled set is easy enough that every query finds a relevant doc by k=5.
    assert metrics["hit@5"] == 1.0
    # Sanity: the harness's own CI floors hold.
    assert metrics["recall@5"] >= 0.75
    assert metrics["map"] >= 0.6
