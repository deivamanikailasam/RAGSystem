# 6. Scaling & Evolution

> **Goal of this stage:** once v1 is solid, scale on cost, performance, and
> multi-tenancy — and evolve the feature set on top of the same retrieval layer.

---

## 6.1 Scaling FAISS

The index type is a single config knob (`FAISS_INDEX_TYPE`) implemented in
`app/core/vector_store.py:TenantIndex._new_index`.

| Corpus size / tenant | Index | Trade-off |
|----------------------|-------|-----------|
| ≲ 100k vectors | `flat` (`IndexFlatIP`) | exact, simple, low memory |
| 100k – millions | `ivf` (`IndexIVFFlat`) | approximate; tune `nlist`/`nprobe` |
| + memory pressure | add PQ (`IndexIVFPQ`) | compresses vectors, some recall loss |
| very high QPS | GPU FAISS | big latency win at scale |

**Tuning IVF** (`IVF_NLIST`, `IVF_NPROBE`):
- `nlist` ≈ `sqrt(N)` cells is a good starting point.
- higher `nprobe` → better recall, higher latency. Sweep it on your eval set
  (§5.1) to hit your recall target at the lowest latency.
- IVF must be **trained** before first use — handled in
  `TenantIndex._maybe_train`.

**Partitioning.** Keep per-index size manageable by partitioning by tenant
(already the default), and optionally further by domain or doc type. Smaller
indices are faster to search and rebuild.

## 6.2 Scaling LLM usage (cost & latency)

- **Cache answers** for repeated/near-duplicate questions (key on normalized
  question + tenant + top doc ids). Biggest single cost lever for support bots.
- **Precompute FAQs** — answer common questions offline and serve from cache.
- **Batch & async generation** for bulk workflows (nightly report Q&A, etc.).
- **Right-size models** — a smaller generation model with good retrieval often
  beats a bigger model with poor retrieval, at a fraction of the cost. Decide
  with the eval set, not vibes.
- **Batch embeddings** — already done at ingest (`embeddings.py`, 128/req);
  extend to a queue for very large corpora.

## 6.3 Multi-tenant platform design

The building blocks are already here:
- **Per-tenant FAISS index** + tenant-scoped metadata rows (hard isolation).
- **Shared services** for ingestion and generation (stateless, scale
  horizontally).
- **Auth → tenant** resolution in one place (`deps.py:require_tenant`) — swap in
  your IdP without touching the pipeline.

To grow into a platform, add:
- a **centralized policy layer** for prompt templates, guardrails, and access
  rules per tenant (generalize `generator.py:SYSTEM_PROMPT` into per-tenant
  config);
- **per-tenant quotas / rate limits / cost budgets** keyed on the tenant id;
- a **control plane** for onboarding tenants, managing sources, and triggering
  reindexes.

## 6.4 Scaling the write path

- Move ingestion to **dedicated workers** (Celery/SQS/Lambda) so heavy
  parse+embed work never blocks the query path.
- Make ingestion **event-driven** (upload webhook, scheduled source sync).
- Keep it **idempotent + versioned** (already true) so retries and replays are
  safe.
- Use **versioned indices**: build the new index alongside the live one and swap
  atomically so queries never see a half-built index.

## 6.5 Feature evolution (on the same retrieval layer)

Because retrieval is decoupled from generation, new features are mostly new
prompts/flows over the existing `Retriever`:

- **Whole-doc summarization** — retrieve all chunks of a `doc_id`, summarize.
- **Doc compare** — retrieve from two docs, ask the model to diff.
- **"Teach me this topic"** — multi-turn, retrieval-grounded tutoring.
- **Streaming chat UX** — stream tokens from `generator.py` over WebSocket/SSE.
- **More sources** — SharePoint, Confluence, Google Drive, internal APIs: add
  connectors that normalize to text + metadata and call the same ingest path.
- **Hybrid retrieval** — combine FAISS (dense) with a keyword/BM25 index and
  fuse scores for better recall on rare terms.
- **Real reranking** — replace the lexical stub in `Retriever._rerank` with a
  cross-encoder or LLM reranker.

## 6.6 Evolution roadmap (suggested order)

1. Ship v1 (flat index, single/few tenants, offline-capable) ✅ this repo.
2. Add streaming + user feedback capture.
3. Add answer/query caching + FAQ precompute.
4. Move ingestion to async workers; add more source connectors.
5. Switch large tenants to IVF (+PQ); tune on the eval set.
6. Add hybrid retrieval + a real reranker.
7. Build the multi-tenant control plane + per-tenant policy/quotas.
8. Consider GPU FAISS / sharding when QPS or corpus demands it.

---

**You've now covered the full lifecycle** — ideation → architecture →
implementation → deployment → evaluation → scaling — with runnable code behind
each stage. Back to the [docs index](README.md) or the
[project README](../README.md).
