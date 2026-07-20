# 1. Ideation & Requirements

> **Goal of this stage:** decide *what* you're building and *for whom*, and turn
> that into concrete, testable requirements that every later stage can be
> measured against.

---

## 1.1 Frame the product

Treat this as a single product — an enterprise document Q&A assistant — even if
it starts as a single-tenant bot. Pick the primary use case, because it drives
almost every downstream decision:

| Use case | Consequence for the system |
|----------|----------------------------|
| **Internal docs QA** (eng/product) | Moderate volume, freshness matters, SSO auth |
| **Customer support** (FAQs, policies) | High QPS, strong caching, tight guardrails |
| **Compliance / policy answering** | Auditability, citations mandatory, on-prem data |

This repository is built as an internal/enterprise doc QA baseline that can grow
into a multi-tenant platform.

## 1.2 Capture requirements

Write these down as numbers, not adjectives — they become your eval thresholds
(see [stage 5](05-evaluation-and-monitoring.md)).

**Functional**
- Supported document types: PDF, Markdown, HTML, TXT (extendable — see
  `app/api/routes.py:_extract_text`).
- Operations: ingest, query, delete, re-ingest (update).
- Answers must cite their sources.
- The system must say "I don't know" rather than hallucinate.

**Non-functional**
- **Latency budget** — e.g. p95 query < 2–3 s. We measure retrieval and
  generation latency separately (`/metrics`) so you can tell which half to
  optimize.
- **Cost** — embedding + generation tokens dominate. We track `tokens_total`
  per query and batch embeddings to cut overhead
  (`app/core/embeddings.py:OpenAIEmbeddingProvider`).
- **Data sensitivity** — PII/confidential docs drive storage, access control,
  and whether the LLM may be a hosted API. Per-tenant isolation is built in.
- **Document dynamics** — static vs frequently updated. Our ingestion is
  idempotent and versioned so updates are cheap
  (`app/core/ingest.py:IngestionPipeline`).

## 1.3 Derive design parameters

From the requirements, fix the knobs (all in `app/config.py`, overridable via
`.env`):

| Requirement | Parameter | Default |
|-------------|-----------|---------|
| How much context per answer | `MAX_CONTEXT_CHUNKS`, `RETRIEVAL_TOP_K` | 6 / 6 |
| Chunk granularity | `CHUNK_TOKENS`, `CHUNK_OVERLAP` | 400 / 60 |
| Corpus size per tenant | `FAISS_INDEX_TYPE` (`flat`→`ivf`) | `flat` |
| Hallucination control | system prompt + `MIN_SCORE` | see `generator.py` |
| Multi-tenancy | per-tenant index + docstore rows | always on |

## 1.4 Guardrails (decided up front, enforced in code)

1. **Grounding** — the model may answer *only* from retrieved context. Enforced
   by `SYSTEM_PROMPT` in `app/core/generator.py`.
2. **"I don't know" behavior** — when retrieval returns nothing (or below
   `MIN_SCORE`), we short-circuit to a fixed refusal instead of calling the LLM.
3. **Citations** — every answer returns `citations[]` with `doc_id`, `source`,
   `score`, and a snippet, so answers are auditable.

## 1.5 Step-by-step checklist for this stage

1. [ ] Write a one-paragraph product definition and primary use case.
2. [ ] List functional + non-functional requirements as numbers.
3. [ ] Enumerate document types and sources you must support.
4. [ ] Decide the guardrail policy (grounding, refusal, citations).
5. [ ] Decide single- vs multi-tenant (this repo supports both).
6. [ ] Fill in `.env` from `.env.example` to encode your parameters.

**Output of this stage:** a filled-in `.env` and an agreed requirements list.
Proceed to [architecture](02-architecture.md).
