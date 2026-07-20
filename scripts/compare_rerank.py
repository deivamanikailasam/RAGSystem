#!/usr/bin/env python3
"""Compare reranking strategies on the offline eval set, side by side.

Runs the exact same corpus + labeled questions (from ``eval/``) through each
available reranking strategy and prints retrieval/answer metrics so you can see
the impact of reranking empirically rather than taking it on faith.

Usage::

    python scripts/compare_rerank.py

Only offline-capable strategies (``none``, ``lexical``) run by default. The
``cross_encoder`` and ``llm`` strategies are included automatically when their
dependency / API key is available; otherwise they are skipped with a note.
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


def build_engine(tmp: Path, strategy: str) -> RagEngine:
    settings = Settings(
        openai_api_key=None,
        data_dir=tmp,
        retrieval_top_k=5,
        rerank_candidates=20,
        rerank_strategy=strategy,
    )
    engine = RagEngine(settings)
    for doc in load_jsonl(EVAL_DIR / "corpus.jsonl"):
        engine.ingest(
            tenant=TENANT, text=doc["text"], source=doc["source"], doc_id=doc["doc_id"]
        )
    return engine


def candidate_strategies() -> list[str]:
    strategies = ["none", "lexical"]
    try:
        import sentence_transformers  # noqa: F401

        strategies.append("cross_encoder")
    except ImportError:
        print("(cross_encoder skipped — install requirements-rerank.txt)")
    if Settings().use_openai:
        strategies.append("llm")
    else:
        print("(llm skipped — set OPENAI_API_KEY to include it)")
    return strategies


def main() -> int:
    cases = load_jsonl(EVAL_DIR / "dataset.jsonl")
    print("=== Reranking strategy comparison ===")
    header = f"{'strategy':16} {'recall@k':>9} {'mrr':>6} {'keyword_cov':>12}"
    print(header)
    print("-" * len(header))
    for strategy in candidate_strategies():
        with tempfile.TemporaryDirectory() as tmp:
            engine = build_engine(Path(tmp), strategy)
            r = evaluate(engine, cases, k=5)
        print(
            f"{engine.reranker.name:16} {r['recall_at_k']:>9} "
            f"{r['mrr']:>6} {r['keyword_coverage']:>12}"
        )
    print("\nNote: on a tiny synthetic corpus differences are small; reranking's "
          "advantage grows with corpus size and noise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
