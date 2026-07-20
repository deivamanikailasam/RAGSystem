# 11. Evaluating Retrieval Quality (Precision / Recall)

> **Goal of this doc:** measure how good the *retriever* is — separately from
> the generator — using standard information-retrieval metrics
> (precision, recall, F1, MRR, MAP, nDCG), step by step.

Retrieval is the foundation of RAG: if the right passage isn't retrieved, no
prompt or model can produce a grounded answer. So evaluate it on its own, with
numbers, before blaming the LLM.

---

## 11.1 Why evaluate retrieval separately

An end-to-end answer can be wrong for two very different reasons:
1. the retriever didn't surface the relevant passage (a **retrieval** failure);
2. the retriever found it but the model ignored/misused it (a **generation** failure).

Mixing them hides the cause. This harness (`eval/retrieval_eval.py`) scores the
retriever alone — it calls `RagEngine.retrieve` (stage-1 retrieval + reranking,
**no LLM**) and compares the returned ranking to human relevance judgments.

## 11.2 Relevance judgments (the labeled set)

You need, per test query, the set of documents that *should* be retrieved. This
system keeps them in two files under `eval/`:

- **`retrieval_corpus.jsonl`** — the documents to index (`doc_id`, `source`, `text`).
- **`retrieval_dataset.jsonl`** — the queries + judgments:

```json
{"question": "What is BM25 and how does hybrid retrieval use it?",
 "relevant_doc_ids": ["bm25", "hybrid"]}
```

Guidelines for building a good set (see [stage 5](05-evaluation-and-monitoring.md)):
- 20–200 real queries per domain; include multi-answer queries
  (`relevant_doc_ids` with several ids) so recall and precision are meaningful.
- Judgments are **binary** here (relevant / not). Grade doc-level.
- Version it; every production miss becomes a new labeled case (regression test).

## 11.3 The metrics

Let *retrieved* be the ranked list of doc ids (best first) and *R* the relevant
set. All are implemented as pure functions in `eval/metrics.py`.

| Metric | Question it answers | Formula (binary relevance) |
|--------|---------------------|----------------------------|
| **Precision@k** | Of the top-k I showed, how many are relevant? | `|top_k ∩ R| / k` |
| **Recall@k** | Of all relevant docs, how many did I find in top-k? | `|top_k ∩ R| / |R|` |
| **F1@k** | Balance of the two | `2·P·R / (P+R)` |
| **MRR** | How high is the *first* correct hit? | `mean(1 / rank_first_relevant)` |
| **MAP** | Are *all* relevant docs ranked highly? | `mean(AP)`, `AP = (Σ_k P@k·rel_k)/|R|` |
| **nDCG@k** | Relevant docs earlier = better, 0–1 scale | `DCG@k / IDCG@k` |
| **Hit@k** | Did I get *anything* right in top-k? | `1 if top_k ∩ R else 0` |

Intuition:
- **Precision vs recall trade-off:** raising `k` usually raises recall but lowers
  precision (more results = more chances to include a relevant one, but also
  more noise). Report both at several `k` (1, 3, 5).
- **MRR** cares only about the first hit — good for "one right answer" queries.
- **MAP / nDCG** reward getting the *whole* relevant set ordered well — better
  for multi-answer queries.

> **Chunks → docs.** The retriever returns *chunks*; several may belong to one
> document. The harness de-duplicates the ranked chunks to a ranked list of
> unique `doc_id`s before scoring, because judgments are doc-level
> (`ranked_doc_ids` in `eval/retrieval_eval.py`).

## 11.4 Step-by-step: run it

```bash
# Default config (hybrid retrieval + lexical rerank), with per-query breakdown
python eval/retrieval_eval.py --per-query
```
```
=== mode=hybrid  rerank=lexical ===
  @1:  P=0.750  R=0.562  F1=0.625  nDCG=0.750  hit=0.750
  @3:  P=0.417  R=0.875  F1=0.550  nDCG=0.829  hit=1.000
  @5:  P=0.250  R=0.875  F1=0.381  nDCG=0.829  hit=1.000
  MRR=0.875   MAP=0.781

  per-query (recall@5 / AP@5):
    R=1.00 AP=1.00  How does FAISS index vectors for nearest neighbor search?
    R=0.50 AP=0.25  How are documents chunked and turned into embeddings?
    ...
PASS
```

How to read it:
- `P@1=0.75` → for 3 of 4 queries the very top result was relevant.
- `R@3=0.875` → by rank 3 the retriever found ~88% of all relevant docs.
- Precision falls from `@1`→`@5` while recall rises — the expected trade-off.
- The per-query lines show *which* queries are weak (here, the
  chunking/embeddings query only recovered half its relevant docs).

## 11.5 Compare configurations

Which retrieval mode / reranker actually helps? Measure it:

```bash
python eval/retrieval_eval.py --compare          # vector vs bm25 vs hybrid
python eval/retrieval_eval.py --mode hybrid --rerank cross_encoder
```

This is the objective way to justify turning on hybrid retrieval (doc 9) or a
heavier reranker (doc 8): run the labeled set through each and compare MAP /
recall@k, rather than guessing.

## 11.6 Wire it into CI

`retrieval_eval.py` exits non-zero if it drops below floors
(`recall@5 ≥ 0.75`, `MAP ≥ 0.6` by default), so a chunking/embedding/fusion
change that quietly hurts retrieval fails the build:

```yaml
# .github/workflows/ci.yml (excerpt)
- run: pytest -q
- run: python eval/retrieval_eval.py
```

Adjust the floors in `FLOORS` as your labeled set grows and stabilizes.

## 11.7 Honest caveats

- **Offline embedder.** With no `OPENAI_API_KEY` the system uses the lexical
  fallback embedder, so `vector`, `bm25`, and `hybrid` score similarly on the
  toy corpus. Set a real key and use a larger labeled set to see the true
  spread — the *methodology* is identical.
- **Binary judgments.** Real relevance is graded; nDCG supports graded gains if
  you extend the judgments (swap the `1.0` in `dcg_at_k` for a gain value).
- **Retrieval ≠ answer quality.** This measures whether the right context was
  found. Pair it with answer-level evaluation (LLM-as-judge / human review,
  [stage 5](05-evaluation-and-monitoring.md)) for the full picture.

## 11.8 Reference map

| Concern | Code |
|---------|------|
| Metric functions | `eval/metrics.py` |
| Labeled corpus + judgments | `eval/retrieval_corpus.jsonl`, `eval/retrieval_dataset.jsonl` |
| Harness + mode comparison + CI floor | `eval/retrieval_eval.py` |
| Generation-free retrieval path | `app/core/rag.py:RagEngine.retrieve` |
| Metric unit tests (hand-computed) | `tests/test_eval_metrics.py` |

Back to the [docs index](README.md).
