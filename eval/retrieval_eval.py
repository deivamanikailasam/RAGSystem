#!/usr/bin/env python3
"""Retrieval-quality evaluation with precision / recall (and friends).

This harness measures the **retriever**, isolated from the generator: it feeds
each labeled question through `RagEngine.retrieve` (stage 1 + reranking, no LLM),
de-duplicates the ranked chunks to a ranked list of document ids, and scores
that list against the query's relevance judgments using `eval/metrics.py`.

Reported (mean over queries): precision@k, recall@k, F1@k, nDCG@k, hit@k for
k in {1,3,5}, plus MRR and MAP.

Usage::

    python eval/retrieval_eval.py                 # default config (hybrid + lexical)
    python eval/retrieval_eval.py --per-query     # also print a per-query table
    python eval/retrieval_eval.py --compare       # vector vs bm25 vs hybrid

Runs fully offline (no OPENAI_API_KEY) into a throwaway temp index.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.core.rag import RagEngine  # noqa: E402
from eval.metrics import (  # noqa: E402
    average_precision,
    evaluate_queries,
    recall_at_k,
)

EVAL_DIR = Path(__file__).resolve().parent
TENANT = "eval"
KS = (1, 3, 5)
# CI floors: fail if retrieval regresses below these on the default config.
FLOORS = {"recall@5": 0.75, "map": 0.6}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_engine(tmp: Path, retrieval_mode: str, rerank: str) -> RagEngine:
    settings = Settings(
        openai_api_key=None,
        data_dir=tmp,
        retrieval_top_k=10,          # retrieve enough to score @5 on unique docs
        max_context_chunks=10,
        retrieval_mode=retrieval_mode,
        rerank_strategy=rerank,
    )
    engine = RagEngine(settings)
    for doc in load_jsonl(EVAL_DIR / "retrieval_corpus.jsonl"):
        engine.ingest(
            tenant=TENANT, text=doc["text"], source=doc["source"], doc_id=doc["doc_id"]
        )
    return engine


def ranked_doc_ids(engine: RagEngine, question: str, k: int = 10) -> list[str]:
    """Ranked, de-duplicated document ids for a query (best first)."""
    chunks = engine.retrieve(tenant=TENANT, question=question, top_k=k)
    seen: list[str] = []
    for ch in chunks:
        if ch.record.doc_id not in seen:
            seen.append(ch.record.doc_id)
    return seen


def run(retrieval_mode: str, rerank: str, cases: list[dict]) -> tuple[dict, list[tuple]]:
    with tempfile.TemporaryDirectory() as tmp:
        engine = build_engine(Path(tmp), retrieval_mode, rerank)
        pairs: list[tuple[list[str], set[str]]] = []
        per_query = []
        for case in cases:
            retrieved = ranked_doc_ids(engine, case["question"])
            relevant = set(case["relevant_doc_ids"])
            pairs.append((retrieved, relevant))
            per_query.append(
                (
                    case["question"],
                    round(recall_at_k(retrieved, relevant, 5), 3),
                    round(average_precision(retrieved, relevant, 5), 3),
                )
            )
    return evaluate_queries(pairs, ks=KS), per_query


def print_metrics(title: str, metrics: dict) -> None:
    print(f"\n=== {title} ===")
    for k in KS:
        print(
            f"  @{k}:  P={metrics[f'precision@{k}']:.3f}  "
            f"R={metrics[f'recall@{k}']:.3f}  "
            f"F1={metrics[f'f1@{k}']:.3f}  "
            f"nDCG={metrics[f'ndcg@{k}']:.3f}  "
            f"hit={metrics[f'hit@{k}']:.3f}"
        )
    print(f"  MRR={metrics['mrr']:.3f}   MAP={metrics['map']:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    parser.add_argument("--compare", action="store_true",
                        help="compare vector / bm25 / hybrid retrieval modes")
    parser.add_argument("--per-query", action="store_true",
                        help="print recall@5 and AP@5 for each query")
    parser.add_argument("--mode", default="hybrid")
    parser.add_argument("--rerank", default="lexical")
    args = parser.parse_args()

    cases = load_jsonl(EVAL_DIR / "retrieval_dataset.jsonl")
    print(f"Retrieval evaluation over {len(cases)} labeled queries "
          f"({len(load_jsonl(EVAL_DIR / 'retrieval_corpus.jsonl'))} docs).")

    if args.compare:
        for mode in ("vector", "bm25", "hybrid"):
            metrics, _ = run(mode, args.rerank, cases)
            print_metrics(f"mode={mode}  rerank={args.rerank}", metrics)
        return 0

    metrics, per_query = run(args.mode, args.rerank, cases)
    print_metrics(f"mode={args.mode}  rerank={args.rerank}", metrics)

    if args.per_query:
        print("\n  per-query (recall@5 / AP@5):")
        for q, r, ap in per_query:
            print(f"    R={r:.2f} AP={ap:.2f}  {q}")

    # CI gate on the default configuration.
    failures = [
        f"{name} {metrics[name]:.3f} < {floor}"
        for name, floor in FLOORS.items()
        if metrics.get(name, 0.0) < floor
    ]
    if failures:
        print("\nFAIL: " + "; ".join(failures))
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
