# 8. The Reranking Stage (Two-Stage Retrieval)

> **Goal of this doc:** explain, step by step, *why* a RAG pipeline needs a
> reranking stage, *how* the two-stage retrieve-then-rerank design works here,
> and *how* to configure, extend, and evaluate each reranking strategy.

---

## 8.1 Why rerank at all?

Plain vector search uses a **bi-encoder**: the query and each chunk are embedded
*independently*, then compared by cosine similarity. That is fast (it's just a
FAISS lookup) and great for **recall** — the right passage is usually somewhere
in the top 20. But because the two embeddings never "see" each other, the
ordering is coarse: a passage that merely shares topic words can outrank the one
that actually answers the question.

A **reranker** fixes the ordering. It re-scores each candidate *jointly with the
query* and sorts by that sharper signal. The RAG answer is only as good as the
few chunks you feed the LLM, so getting the true best passage into the top-k is
high leverage — and cheaper than it sounds, because you only rerank ~20
candidates, not the whole corpus.

```
                    stage 1: recall                stage 2: precision
 query ─▶ embed ─▶ FAISS top-N (fast, coarse) ─▶ rerank N (slow, sharp) ─▶ top_k ─▶ LLM
                    │  e.g. N = 20                  │  reorder & truncate
                    └─ app/core/retriever.py        └─ app/core/reranker.py
```

## 8.2 How the two stages are wired

The orchestration lives in `app/core/rag.py:RagEngine.query`:

```python
final_k = top_k or settings.retrieval_top_k

# stage 1 — pull a candidate pool larger than final_k
candidate_pool = final_k if reranker.name == "none" else max(rerank_candidates, final_k)
candidates = retriever.retrieve(tenant=..., question=..., limit=candidate_pool, filters=...)

# stage 2 — rerank the pool, keep the best final_k
chunks = reranker.rerank(question, candidates, top_n=final_k)
```

- **Stage 1 (`app/core/retriever.py`)** is now a *pure* recall stage: embed the
  query, FAISS search, apply `MIN_SCORE` and metadata filters, and return up to
  `limit` candidates ordered by vector score. It does **not** truncate to the
  final answer size.
- **Stage 2 (`app/core/reranker.py`)** re-scores those candidates and returns
  the top `final_k`, setting a `rerank_score` on each.

Both `retrieval_ms` and `rerank_ms` are measured and returned per query, and the
strategy used is echoed as `reranker` in the response:

```json
{
  "answer": "...",
  "reranker": "lexical",
  "retrieval_ms": 3.1,
  "rerank_ms": 0.6,
  "citations": [
    {"doc_id": "handbook", "score": 0.55, "rerank_score": 1.15, "snippet": "..."}
  ]
}
```

`score` is the stage-1 vector similarity; `rerank_score` is the stage-2
relevance. Seeing both makes it obvious when reranking changed the order.

## 8.3 The four strategies

Selected by `RERANK_STRATEGY`; all implement one interface
(`Reranker.rerank(query, candidates, top_n)`), so switching is config-only.

| Strategy | Quality | Cost / latency | Needs | Code |
|----------|---------|----------------|-------|------|
| `none` | baseline (vector order) | ~0 | — | `NoOpReranker` |
| `lexical` *(default)* | good | ~0, offline | — | `LexicalReranker` (BM25) |
| `cross_encoder` | best | model inference | `requirements-rerank.txt` | `CrossEncoderReranker` |
| `llm` | very good | 1 LLM call/query | `OPENAI_API_KEY` | `LLMReranker` |

### `none`
Identity — keeps vector order, copies `score` into `rerank_score`. Use it as the
A/B baseline when measuring whether reranking helps.

### `lexical` (default, offline)
A **BM25** reranker (`app/core/reranker.py:LexicalReranker`). BM25 rewards
passages containing the query terms, down-weights common terms via IDF computed
across the candidate set, and normalizes for passage length. It needs no model
and no network, so it is deterministic — ideal for the default, tests, and
air-gapped deployments.

### `cross_encoder` (best quality)
A real cross-encoder (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) via
`sentence-transformers`. It feeds `[query, passage]` through one transformer and
emits a single relevance logit — the gold standard for reranking. It's an
**optional** dependency:

```bash
pip install -r requirements-rerank.txt
export RERANK_STRATEGY=cross_encoder
```
If the package isn't installed, the factory logs a warning and falls back to
`lexical` — the pipeline never breaks.

### `llm` (LLM-as-reranker)
`LLMReranker` asks the model to score each candidate 0–10 in a single batched
JSON call, then sorts by score. On any API/parse error it falls back to vector
order, so a reranker hiccup never fails a query. Requires `OPENAI_API_KEY`.

## 8.4 Configuration reference

```bash
RERANK_STRATEGY=lexical                 # none | lexical | cross_encoder | llm
RERANK_CANDIDATES=20                    # stage-1 pool size fed to the reranker
CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANK_LLM_MODEL=                       # defaults to GENERATION_MODEL
# RERANK_ENABLED=true                   # deprecated alias -> lexical
```

**Tuning `RERANK_CANDIDATES`.** This is the recall/precision/latency dial:
- too small → the reranker can't recover a passage vector search buried;
- too large → more rerank work (and, for `cross_encoder`/`llm`, more cost).
A pool of 20–50 is typical. It only matters when a reranker is active; with
`none` the pool equals `top_k`.

## 8.5 Step-by-step: enable and observe reranking

```bash
# 1. Run with the default lexical reranker
uvicorn app.main:app --port 8000

# 2. Ingest a few docs
python scripts/ingest_dir.py ./docs --tenant demo   # multi-tenant
#   (or POST /v1/ingest; in single-tenant mode no auth is needed)

# 3. Query and inspect the two-stage signals
curl -s localhost:8000/v1/query -H 'content-type: application/json' \
  -H 'Authorization: Bearer demo-key' \
  -d '{"question":"How do I request a refund?","top_k":3}' | jq \
  '{reranker, retrieval_ms, rerank_ms, citations: [.citations[] | {doc_id, score, rerank_score}]}'
```
You'll see `rerank_score` reorder candidates whose `score` (vector) order was
different — that reordering is the reranker doing its job.

Switch strategies without code changes:
```bash
RERANK_STRATEGY=none          uvicorn app.main:app     # baseline
RERANK_STRATEGY=cross_encoder uvicorn app.main:app     # after pip install -r requirements-rerank.txt
OPENAI_API_KEY=sk-... RERANK_STRATEGY=llm uvicorn app.main:app
```

## 8.6 Measuring the impact (don't take it on faith)

`scripts/compare_rerank.py` runs the labeled eval set (`eval/`) through every
available strategy and prints recall@k / MRR / keyword-coverage side by side:

```bash
python scripts/compare_rerank.py
```
```
strategy          recall@k    mrr  keyword_cov
----------------------------------------------
none                   1.0    1.0          0.9
lexical                1.0    0.9          0.8
```
> On the tiny bundled corpus the strategies are near-identical — with only a
> couple of documents there is nothing to reorder. Reranking's advantage grows
> with corpus **size and noise**: the more plausible-but-wrong passages vector
> search surfaces, the more a reranker earns its keep. Add your own labeled set
> (see [stage 5](05-evaluation-and-monitoring.md)) to quantify it on real data,
> and wire the comparison into CI to guard against regressions.

## 8.7 Extending: plug in your own reranker

1. Add a class with a `name` attribute and a
   `rerank(query, candidates, top_n) -> list[RetrievedChunk]` method that sets
   `rerank_score` on each returned chunk.
2. Register it in `app/core/reranker.py:build_reranker` under a new
   `RERANK_STRATEGY` value.
3. That's it — `RagEngine` already calls `reranker.rerank(...)` generically.

Cohere Rerank, Jina Reranker, a hosted cross-encoder endpoint, or a
reciprocal-rank-fusion of several signals all fit this interface.

## 8.8 Reference map

| Concern | Code |
|---------|------|
| Strategy selection | `app/config.py` (`rerank_strategy`, `rerank_candidates`, `cross_encoder_model`, `rerank_llm_model`) |
| Reranker implementations + factory | `app/core/reranker.py` |
| Stage-1 candidate retrieval | `app/core/retriever.py:Retriever.retrieve` |
| Two-stage wiring + timings | `app/core/rag.py:RagEngine.query` |
| Response fields (`reranker`, `rerank_score`, `rerank_ms`) | `app/models.py` |
| Strategy comparison | `scripts/compare_rerank.py` |
| Optional cross-encoder deps | `requirements-rerank.txt` |

Back to the [docs index](README.md).
