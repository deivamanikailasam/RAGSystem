#!/usr/bin/env python3
"""Compare retrieval modes (vector / bm25 / hybrid) on the offline eval set.

Runs the same corpus + labeled questions through each stage-1 retrieval mode
(reranking disabled) and prints retrieval metrics side by side, so the effect
of adding the sparse BM25 signal and fusing it with dense vectors is visible.

Usage::

    python scripts/compare_retrieval.py

Note: the offline fallback embedder is itself lexical (hashed subwords), so the
dense/sparse signals overlap more than they would with real OpenAI embeddings —
the dramatic hybrid win shows up on a real corpus with real embeddings. This
script still demonstrates that all three modes run and how they rank.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.core.rag import RagEngine  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from run_eval import EVAL_DIR, TENANT, evaluate, load_jsonl  # noqa: E402


def build_engine(tmp: Path, mode: str) -> RagEngine:
    settings = Settings(
        openai_api_key=None,
        data_dir=tmp,
        retrieval_top_k=5,
        retrieval_mode=mode,
        rerank_strategy="none",  # isolate the retrieval stage
    )
    engine = RagEngine(settings)
    for doc in load_jsonl(EVAL_DIR / "corpus.jsonl"):
        engine.ingest(
            tenant=TENANT, text=doc["text"], source=doc["source"], doc_id=doc["doc_id"]
        )
    return engine


def main() -> int:
    cases = load_jsonl(EVAL_DIR / "dataset.jsonl")
    print("=== Retrieval mode comparison (reranking disabled) ===")
    header = f"{'mode':10} {'recall@k':>9} {'mrr':>6} {'keyword_cov':>12}"
    print(header)
    print("-" * len(header))
    for mode in ("vector", "bm25", "hybrid"):
        with tempfile.TemporaryDirectory() as tmp:
            engine = build_engine(Path(tmp), mode)
            r = evaluate(engine, cases, k=5)
        print(
            f"{mode:10} {r['recall_at_k']:>9} {r['mrr']:>6} {r['keyword_coverage']:>12}"
        )
    print("\nHybrid unions dense recall with sparse keyword precision; its edge "
          "grows with corpus size, rare terms, and real (non-lexical) embeddings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
