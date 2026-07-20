# 2. System Architecture & Data Model

> **Goal of this stage:** define the layers, how data flows between them, and the
> data model — before writing pipeline code.

---

## 2.1 The layers

```
┌──────────────────────────────────────────────────────────────────────┐
│  Serving / API           app/main.py, app/api/routes.py, app/deps.py   │
│    /health /metrics /v1/ingest /v1/query /v1/documents/{id}            │
├──────────────────────────────────────────────────────────────────────┤
│  Orchestration           app/core/rag.py  (RagEngine)                  │
├───────────────┬───────────────────────────┬──────────────────────────┤
│  Ingestion    │  Retrieval                │  Generation              │
│  ingest.py    │  retriever.py             │  generator.py            │
│  chunking.py  │                           │                          │
├───────────────┴───────────────────────────┴──────────────────────────┤
│  Storage         vector_store.py (FAISS)   +   docstore.py (SQLite)    │
├──────────────────────────────────────────────────────────────────────┤
│  Providers       embeddings.py (OpenAI | local)   generator (OpenAI|extractive) │
├──────────────────────────────────────────────────────────────────────┤
│  Observability   observability/metrics.py                             │
└──────────────────────────────────────────────────────────────────────┘
```

| Layer | Responsibility | Code |
|-------|----------------|------|
| Ingestion | fetch, parse, chunk, version | `core/ingest.py`, `core/chunking.py` |
| Embedding & indexing | embed, build/persist FAISS, store metadata | `core/embeddings.py`, `core/vector_store.py`, `core/docstore.py` |
| Retrieval | similarity search + filter + rerank | `core/retriever.py` |
| Generation | prompt assembly, LLM call, formatting | `core/generator.py` |
| Serving/API | HTTP endpoints, auth, rate limiting | `api/routes.py`, `deps.py` |
| Observability | logs, metrics, traces, evals | `observability/metrics.py` |
| Governance | RBAC, data isolation | `deps.py` + per-tenant storage |

## 2.2 Data flow

**Ingestion (write path)**

```
document ──▶ parse to text ──▶ chunk (overlap) ──▶ embed (batch)
        ──▶ assign vector ids ──▶ FAISS.add_with_ids ──▶ persist
        ──▶ write chunk+doc rows to SQLite
```

**Query (read path)**

```
question ──▶ embed ──▶ FAISS top-k ──▶ hydrate metadata (SQLite)
         ──▶ filter by metadata ──▶ (optional) rerank ──▶ cap to N
         ──▶ assemble prompt ──▶ LLM ──▶ answer + citations
```

Both paths are implemented in `app/core/rag.py:RagEngine`.

## 2.3 The key architectural decision: split vectors from metadata

We store **vectors in FAISS** and **everything else in SQLite**, joined by a
stable integer `vector_id`.

Why:
- You can **rebuild or re-tune the FAISS index** (flat → IVF, change nlist)
  without migrating document content.
- You can **change document metadata / permissions** without touching vectors.
- Deletion is a two-step that stays consistent: `remove_ids` on FAISS +
  `DELETE` on SQLite (`app/core/ingest.py:IngestionPipeline.delete_document`).

## 2.4 Data model

**`documents`** — one row per `(tenant, doc_id)`:
`version`, `source`, `content_hash`, `created_at`.
Content-addressing via `content_hash` makes ingestion idempotent.

**`chunks`** — one row per chunk, primary key `(tenant, vector_id)`:
`doc_id`, `version`, `chunk_index`, `source`, `text`, `metadata (JSON)`.

Full schema: `app/core/docstore.py` (`_SCHEMA`).

**FAISS side** — one `IndexIDMap2` per tenant wrapping either `IndexFlatIP`
(exact) or `IndexIVFFlat` (approximate). Vectors are L2-normalized so inner
product equals cosine similarity.

## 2.5 Tenancy & isolation model

- **Physical vector isolation**: each tenant gets its own FAISS file at
  `{DATA_DIR}/tenants/{tenant}/index.faiss` (`vector_store.py:VectorStore`).
- **Logical metadata isolation**: every SQLite row is keyed by `tenant`, and
  every query is scoped to the caller's tenant.
- **Auth → tenant**: the bearer API key resolves to exactly one tenant in
  `app/deps.py:require_tenant`. There is no cross-tenant code path.

## 2.6 Step-by-step checklist for this stage

1. [ ] Confirm the layer boundaries match your team's ownership.
2. [ ] Decide FAISS index type per expected corpus size (`flat` vs `ivf`).
3. [ ] Decide the metadata fields you'll filter on (add columns or keep in JSON).
4. [ ] Decide the tenancy boundary (per customer? per department?).
5. [ ] Confirm the vectors/metadata split fits your update patterns.

Proceed to [implementation](03-implementation.md).
