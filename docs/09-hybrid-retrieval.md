# 9. Hybrid Retrieval (BM25 + Vector Search)

> **Goal of this doc:** explain, step by step, why a RAG system benefits from
> combining **sparse (BM25)** and **dense (vector)** retrieval, how the fusion
> works here, and how to configure, extend, and evaluate it.

---

## 9.1 Two kinds of search, two blind spots

| | **Dense (vector)** | **Sparse (BM25)** |
|---|---|---|
| Matches on | *meaning* — embeddings | *words* — term overlap |
| Great at | paraphrases, synonyms, "how do I…" | exact terms, IDs, codes, names, acronyms |
| Blind to | rare literal tokens it never learned | anything not sharing surface words |
| Index | FAISS (`app/core/vector_store.py`) | inverted index (`app/core/bm25.py`) |

A user asking *"what does error code **XJ1729** mean?"* is poorly served by dense
search alone — `XJ1729` is a rare token an embedding model has weak signal for.
BM25 nails it (high IDF, exact match). Conversely, *"how do I get my money
back?"* has **no** lexical overlap with a "Refund policy" passage — dense search
wins. **Hybrid retrieval runs both and fuses the results**, so a passage found
by *either* signal is a candidate. That is strictly more recall than either
alone, and it's the single most common production upgrade to naive RAG.

## 9.2 Where it sits in the pipeline

Hybrid is a **stage-1** concern (candidate retrieval). It composes with the
**stage-2** reranker (doc 8): hybrid finds *better candidates*, the reranker
*orders them sharply*.

```
                 ┌─ dense:  embed → FAISS top-N ─┐
 query ──────────┤                               ├─ FUSE (RRF/weighted) ─▶ candidates
                 └─ sparse: BM25 top-N ──────────┘         │
                        (stage 1: retrieval)                ▼
                                                      rerank (stage 2) ─▶ top_k ─▶ LLM
```

Orchestrated in `app/core/rag.py:RagEngine.query`; the retrieval mode is chosen
by `RETRIEVAL_MODE`.

## 9.3 The BM25 index (`app/core/bm25.py`)

- An **inverted index** `term → [(doc, term_freq)]`, so a query only touches
  documents containing its terms — not the whole corpus.
- Standard **Okapi BM25** scoring: term frequency saturated by `k1`, length
  normalized by `b`, terms weighted by IDF.
- Built **lazily from the SQLite docstore** (the source of truth) and cached per
  tenant. `RagEngine` calls `bm25.invalidate(tenant)` after every ingest/delete,
  so the sparse index can never drift from the corpus.

> **No new dependency.** BM25 is implemented in ~60 lines of pure Python, so
> hybrid retrieval works fully offline — consistent with the rest of the system.

## 9.4 Fusion (`app/core/fusion.py`)

Dense cosine (≈0–1) and BM25 scores (unbounded) live on different scales, so you
can't just add them. Two principled options:

### Reciprocal Rank Fusion (RRF) — default
Combine by **rank**, not raw score:

```
score(d) = Σ_i  1 / (k + rank_i(d))
```

summed over each list containing `d`. Scale-free, robust, and the recommended
default. `RRF_K` (≈60) damps how much the very top ranks dominate.

### Weighted (convex) combination
Min-max normalize each list to `[0,1]`, then blend:

```
score(d) = alpha · dense(d) + (1 − alpha) · sparse(d)
```

`HYBRID_ALPHA` (default 0.5) is the dense/sparse dial — raise it to favor
semantic matching, lower it to favor keywords. More tunable than RRF, but
sensitive to score distributions.

## 9.5 Configuration

```bash
RETRIEVAL_MODE=hybrid       # vector | bm25 | hybrid   (default: hybrid)
HYBRID_FUSION=rrf           # rrf | weighted
RRF_K=60                    # RRF damping constant
HYBRID_ALPHA=0.5            # dense weight for weighted fusion (1-alpha = sparse)
RERANK_STRATEGY=lexical     # stage 2, composes on top (see docs/08)
```

Set `RETRIEVAL_MODE=vector` or `bm25` to run a single signal (useful for
A/B baselines and for the comparison script below).

## 9.6 Step-by-step: use and observe hybrid retrieval

```bash
# 1. Run (hybrid is the default mode)
uvicorn app.main:app --port 8000

# 2. Ingest a mix of conceptual and keyword-heavy docs
curl -s localhost:8000/v1/ingest -H 'Authorization: Bearer demo-key' \
  -H 'content-type: application/json' -d '{"documents":[
    {"doc_id":"sem","source":"sem.md","text":"Documents are split into overlapping chunks before embedding."},
    {"doc_id":"code","source":"code.md","text":"Error code XJ1729 means a checksum mismatch on upload."}]}'

# 3. Ask a query that mixes meaning and an exact code
curl -s localhost:8000/v1/query -H 'Authorization: Bearer demo-key' \
  -H 'content-type: application/json' \
  -d '{"question":"overlapping chunks and error code XJ1729"}' | jq \
  '{retrieval_mode, citations: [.citations[] | {doc_id, score, vector_score, bm25_score}]}'
```

Example (real output shape):
```json
{
  "retrieval_mode": "hybrid",
  "citations": [
    {"doc_id": "code", "score": 0.0328, "vector_score": 0.39, "bm25_score": 2.94},
    {"doc_id": "sem",  "score": 0.0323, "vector_score": 0.28, "bm25_score": 1.96},
    {"doc_id": "cal",  "score": 0.0159, "vector_score": 0.06, "bm25_score": null}
  ]
}
```
- `score` is the **fused** rank score; `vector_score`/`bm25_score` show each
  signal's contribution.
- `bm25_score: null` on `cal` means **only the dense arm found it** — exactly
  the recall hybrid adds. (A `vector_score: null` would mean BM25 found a doc the
  vector arm missed.)

## 9.7 Measuring the impact

`scripts/compare_retrieval.py` runs the eval set through each mode
(reranking disabled to isolate stage 1):

```bash
python scripts/compare_retrieval.py
```
```
mode        recall@k    mrr  keyword_cov
vector           1.0    1.0          0.9
bm25             1.0    0.9          0.8
hybrid           1.0    1.0          0.9
```
> **Honest caveat:** the offline fallback embedder is *itself* lexical (hashed
> subwords), so dense and sparse overlap far more than with real embeddings —
> the modes look near-identical on this toy corpus. With real OpenAI embeddings
> and a larger, noisier corpus, hybrid's recall advantage on rare-term and
> paraphrase queries is substantial. Add your own labeled set
> ([stage 5](05-evaluation-and-monitoring.md)) to quantify it on real data.

## 9.8 Scaling notes

- The BM25 index is rebuilt in-memory from the docstore on the first query after
  a write. For a limited corpus this is instant. For large corpora:
  - persist the inverted index and update it **incrementally** on ingest/delete
    instead of full rebuilds, or
  - back BM25 with a dedicated engine (Elasticsearch/OpenSearch, Tantivy,
    Postgres full-text) behind the same `BM25Store.for_tenant(...).search(...)`
    interface.
- Fusion cost is negligible (it operates on the top-N lists, not the corpus).
- Per-tenant isolation is preserved: BM25 indices, like FAISS indices, are keyed
  by tenant.

## 9.9 Extending

- **Swap the sparse engine:** implement the same `search(query, limit) ->
  [(vector_id, score)]` surface as `BM25Index` and hand it to `BM25Store`.
- **Add a third signal** (e.g. title-boost, recency): produce another ranked
  list and pass it into `reciprocal_rank_fusion([...])` — RRF takes any number
  of lists.
- **Tune per query type:** route keyword-looking queries to a lower `HYBRID_ALPHA`
  (more sparse) and natural-language queries to a higher one.

## 9.10 Reference map

| Concern | Code |
|---------|------|
| Mode & fusion config | `app/config.py` (`retrieval_mode`, `hybrid_fusion`, `rrf_k`, `hybrid_alpha`) |
| BM25 index + per-tenant store | `app/core/bm25.py` |
| Fusion (RRF, weighted) | `app/core/fusion.py` |
| Mode selection + fusion wiring | `app/core/retriever.py:Retriever.retrieve` |
| Index invalidation on writes | `app/core/rag.py` (`ingest`/`delete_document`/`purge_tenant`) |
| Response fields (`retrieval_mode`, `vector_score`, `bm25_score`) | `app/models.py` |
| Mode comparison | `scripts/compare_retrieval.py` |

Back to the [docs index](README.md).
