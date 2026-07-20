#!/usr/bin/env python3
"""Offline evaluation harness for retrieval + answer quality.

Metrics reported
----------------
* **Recall@k / Hit-rate** — did the expected document appear in the retrieved
  citations? This isolates *retrieval* quality from generation.
* **MRR** (mean reciprocal rank) — how highly was the right document ranked?
* **Keyword coverage** — a cheap, offline proxy for answer relevance: fraction
  of expected keywords present in the generated answer. In production you would
  additionally use an LLM-as-judge or human labels (see
  docs/05-evaluation-and-monitoring.md).

Usage::

    python eval/run_eval.py

Everything runs against the offline engine into a throwaway temp index, so it
needs no API key and never touches your real data.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.core.rag import RagEngine  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
TENANT = "eval"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_engine(tmp: Path) -> RagEngine:
    settings = Settings(openai_api_key=None, data_dir=tmp, retrieval_top_k=5)
    engine = RagEngine(settings)
    for doc in load_jsonl(EVAL_DIR / "corpus.jsonl"):
        engine.ingest(
            tenant=TENANT, text=doc["text"], source=doc["source"], doc_id=doc["doc_id"]
        )
    return engine


def evaluate(engine: RagEngine, cases: list[dict], k: int = 5) -> dict:
    hits = 0
    reciprocal_ranks = 0.0
    keyword_scores: list[float] = []

    for case in cases:
        resp = engine.query(tenant=TENANT, question=case["question"], top_k=k)
        cited_docs = [c.doc_id for c in resp.citations]

        expected = case["expected_doc_id"]
        if expected in cited_docs:
            hits += 1
            rank = cited_docs.index(expected) + 1
            reciprocal_ranks += 1.0 / rank

        answer_lc = resp.answer.lower()
        kws = case.get("expected_keywords", [])
        if kws:
            covered = sum(1 for kw in kws if kw.lower() in answer_lc)
            keyword_scores.append(covered / len(kws))

    n = len(cases)
    return {
        "cases": n,
        "recall_at_k": round(hits / n, 3) if n else 0.0,
        "mrr": round(reciprocal_ranks / n, 3) if n else 0.0,
        "keyword_coverage": round(sum(keyword_scores) / len(keyword_scores), 3)
        if keyword_scores
        else 0.0,
    }


def main() -> int:
    cases = load_jsonl(EVAL_DIR / "dataset.jsonl")
    with tempfile.TemporaryDirectory() as tmp:
        engine = build_engine(Path(tmp))
        results = evaluate(engine, cases, k=5)

    print("=== QASystem offline evaluation ===")
    for key, value in results.items():
        print(f"  {key:20s}: {value}")
    # Fail CI if retrieval regresses below a floor.
    floor = 0.6
    if results["recall_at_k"] < floor:
        print(f"\nFAIL: recall_at_k {results['recall_at_k']} < {floor}")
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
