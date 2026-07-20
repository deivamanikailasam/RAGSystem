# 3. Implementation: Indexing + Query Pipelines

> **Goal of this stage:** build the two core pipelines. Every step below maps to
> a function you can read and run.

Run the examples in this doc against the offline engine — no API key needed.

---

## 3.1 Ingestion & indexing pipeline

Implemented in `app/core/ingest.py:IngestionPipeline.ingest_document`.

### Step 1 — Document intake & IDs
Each document arrives with a `source` and optional `doc_id`. If no `doc_id` is
supplied we derive a stable one from `source + content_hash`
(`ingest.py:derive_doc_id`). Tenant comes from auth.

### Step 2 — Idempotency check
We compute `content_hash = sha256(text)`. If a document with the same
`(tenant, doc_id)` and identical hash already exists, ingestion is a **no-op**
(`IngestedDocResult(skipped=True)`). If the content changed, we first remove the
old version's vectors and rows, guaranteeing no stale chunks survive.

```python
existing = self._docs.get_document(tenant, doc_id)
if existing is not None and existing["content_hash"] == chash:
    return IngestedDocResult(..., skipped=True)          # unchanged
if existing is not None:                                  # changed -> replace
    self._store.for_tenant(tenant).remove(old_ids)
    self._docs.delete_doc_chunks(tenant, doc_id)
```

### Step 3 — Chunking
`app/core/chunking.py:chunk_text` splits text into **overlapping token windows**
(default 400 tokens, 60 overlap). Overlap keeps a fact that straddles a boundary
intact in at least one chunk. Paragraph boundaries are used as soft hints; no
chunk exceeds the token budget.

```python
chunks = chunk_text(text, chunk_tokens=400, overlap=60)
```

> **Why token-ish, not tiktoken?** The chunker approximates tokens with a word
> split so it runs offline with zero downloads. Swap `_tokenize` for a
> `tiktoken` encoder if you need exact BPE counts.

### Step 4 — Embedding (batched)
`app/core/embeddings.py` embeds all chunks. The OpenAI provider batches (128 per
request) and retries with exponential backoff on rate limits / timeouts. Vectors
are L2-normalized so cosine == inner product.

```python
vectors = self._embeddings.embed([c.text for c in chunks])  # (n, dim) float32
```

### Step 5 — Assign vector ids & index
We allocate a contiguous block of integer ids (`max_vector_id + 1 …`), add them
to the tenant's FAISS `IndexIDMap2` with `add_with_ids`, then **persist** the
index to disk. IVF indices are trained on first add
(`vector_store.py:TenantIndex._maybe_train`).

### Step 6 — Persist metadata
Chunk rows (`vector_id → text + metadata`) and the document row (`content_hash`,
`version`) are written to SQLite. Vectors and metadata are now consistent.

**Try it:**
```bash
python scripts/ingest_dir.py ./docs --tenant demo
```

---

## 3.2 Query & RAG pipeline

Implemented in `app/core/rag.py:RagEngine.query`, which composes the retriever
and generator.

### Step 1 — Auth & tenant resolution
`app/deps.py:require_tenant` turns the `Authorization: Bearer <key>` header into
a tenant id. All subsequent work is scoped to that tenant.

### Step 2 — Embed the query & search
`app/core/retriever.py:Retriever.retrieve` embeds the question with the *same*
provider used at ingest time and runs top-k search in the tenant's FAISS index.
When metadata filters are present we over-fetch (`k × 4`) so filtering still
leaves enough candidates.

```python
query_vec = self._embeddings.embed([question])[0]
hits = self._store.for_tenant(tenant).search(query_vec, fetch_k)
```

### Step 3 — Hydrate & filter
FAISS returns `(vector_id, score)` pairs. We hydrate them into full chunk
records from SQLite, drop anything below `MIN_SCORE`, and apply metadata
equality filters (e.g. `{"doc_type": "policy"}`).

### Step 4 — Rerank (first-class stage)
Retrieval returns a **candidate pool** (default 20). The reranking stage
(`app/core/reranker.py`) re-scores those candidates *jointly with the query* and
keeps the best `top_k`, setting a `rerank_score` on each. Strategy is chosen by
`RERANK_STRATEGY`: `none`, `lexical` (BM25, the offline default),
`cross_encoder` (a real reranker model), or `llm`. This is a full pipeline stage,
not a hook — see **[docs/08-reranking.md](08-reranking.md)** for the deep dive,
tuning, and how to plug in your own reranker.

### Step 5 — Prompt assembly with guardrails
`app/core/generator.py:build_messages` builds:
- a **system prompt** instructing the model to answer *only* from context, to
  reply "I don't know…" when unsupported, and to cite passages `[1] [2]`;
- a **user message** containing the numbered context block + the question.

Context is capped at `MAX_CONTEXT_CHUNKS` regardless of retrieval depth, which
bounds token cost and keeps the prompt focused.

### Step 6 — Generation
`OpenAIGenerator.generate` calls the chat model at low temperature; the offline
`ExtractiveGenerator` returns the top passage verbatim. Either way the answer is
grounded and citable. Token usage is captured for cost tracking.

### Step 7 — Response assembly + observability
`RagEngine.query` returns the answer plus `citations[]`, measured
`retrieval_ms` / `generation_ms`, token counts, and a `request_id`. Latencies
and counters are recorded in `observability/metrics.py`.

**Try it:**
```bash
curl -s localhost:8000/v1/query -H 'Authorization: Bearer demo-key' \
  -H 'content-type: application/json' \
  -d '{"question":"How does chunking work?","top_k":4}' | jq
```

Example response shape:
```json
{
  "answer": "Documents are split into overlapping token windows … [1]",
  "citations": [{"doc_id":"faiss","source":"faiss.md","chunk_index":0,"score":0.83,"snippet":"…"}],
  "model": "gpt-4.1-mini",
  "retrieval_ms": 12.4,
  "generation_ms": 640.1,
  "tokens": {"prompt": 512, "completion": 88, "total": 600},
  "request_id": "…"
}
```

---

## 3.3 Design principles enforced here

- **Idempotent, versioned ingestion** — updates never leave stale vectors.
- **Same embedder for docs and queries** — a mismatch silently destroys recall.
- **Vectors separate from metadata** — re-index without migrating content.
- **Guardrails in the prompt + a hard refusal path** — hallucination control.
- **Every hook is exercised offline** — you can run, test, and evaluate the full
  pipeline with no API key, then flip to OpenAI by setting one env var.

Proceed to [deployment & ops](04-deployment-and-ops.md).
