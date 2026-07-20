# QASystem — Production-Grade RAG Document Q&A (FAISS + OpenAI)

A reference implementation of an **enterprise document Q&A system** built on the
Retrieval-Augmented Generation (RAG) pattern using **FAISS** as the vector store
and **OpenAI** for embeddings + generation.

It ships as a runnable FastAPI service with an ingestion pipeline, a FAISS-backed
retrieval layer, a generation layer with guardrails, per-tenant isolation,
observability, and an evaluation harness — plus step-by-step documentation that
walks the full product lifecycle from ideation to scaling.

> **Runs offline out of the box.** If no `OPENAI_API_KEY` is configured the
> system transparently falls back to a deterministic local embedding provider
> and an extractive answerer, so you can ingest, query, test, and evaluate
> without spending a cent. Set the key to switch to real OpenAI models.

---

## Table of contents

- [Quickstart](#quickstart)
- [Architecture at a glance](#architecture-at-a-glance)
- [API](#api)
- [Project layout](#project-layout)
- [Configuration](#configuration)
- [Testing & evaluation](#testing--evaluation)
- [Documentation](#documentation)

---

## Quickstart

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. (optional) configure OpenAI — otherwise the local fallback is used
cp .env.example .env
# edit .env and set OPENAI_API_KEY=sk-...

# 3. Run the API
uvicorn app.main:app --reload --port 8000

# 4. Ingest a folder of docs (in another shell)
python scripts/ingest_dir.py ./docs --tenant demo

# 5. Ask a question
curl -s localhost:8000/v1/query \
  -H 'Authorization: Bearer demo-key' \
  -H 'content-type: application/json' \
  -d '{"question": "How does the ingestion pipeline chunk documents?"}' | jq
```

Or run the whole thing with Docker:

```bash
docker compose up --build
```

---

## Architecture at a glance

```
                 ┌─────────────┐        ┌──────────────────┐
   documents ───▶│  Ingestion  │───────▶│  Chunking +       │
  (PDF/MD/TXT/   │  (parse,    │        │  Embedding batch  │
   HTML/DOCX)    │   version)  │        └────────┬─────────┘
                 └─────────────┘                 │
                                                 ▼
                                        ┌──────────────────┐
                                        │  FAISS index      │  (per tenant)
                                        │  + SQLite         │
                                        │    docstore       │
                                        └────────┬─────────┘
                                                 │
   user query ──▶ /v1/query ──┬─ dense: embed → FAISS ─┐
                              └─ sparse: BM25 ─────────┴─▶ fuse (RRF/weighted)
                                                 │  (stage 1: hybrid retrieval)
                                                 ▼
                                        rerank N ──▶ top-k
                                        (stage 2: precision;
                                         none/lexical/cross-encoder/llm)
                                                 │
                                                 ▼
                                        prompt assembly ──▶ OpenAI (or
                                        with guardrails       extractive
                                                 │            fallback)
                                                 ▼
                                        answer + citations ──▶ client
```

Every layer is covered in depth in [`docs/`](docs/).

---

## Deployment modes

QASystem runs in either of two modes, chosen with a single env var
`DEPLOYMENT_MODE` — see [`docs/07-deployment-modes.md`](docs/07-deployment-modes.md)
for the full step-by-step guide.

| | **single_tenant** | **multi_tenant** (default) |
|---|---|---|
| Corpus | one implicit corpus | one FAISS index per tenant |
| Auth | optional (open or one key) | required; per-tenant keys |
| Control plane | — | `/admin/*` tenant lifecycle |
| Per-tenant prompt / index / quotas | — | yes |
| Best for | internal doc bot, limited corpus | SaaS / many teams |

```bash
# Single-tenant internal doc QA:
DEPLOYMENT_MODE=single_tenant SINGLE_TENANT_ID=internal uvicorn app.main:app

# Multi-tenant platform:
DEPLOYMENT_MODE=multi_tenant ADMIN_API_KEY=secret uvicorn app.main:app
```

## API

| Method | Path             | Description                                   | Auth |
|--------|------------------|-----------------------------------------------|------|
| GET    | `/health`        | Liveness/readiness probe                      | none |
| GET    | `/metrics`       | In-process counters & latency histograms      | none |
| GET    | `/v1/me`         | Caller's tenant, usage & quotas               | tenant |
| POST   | `/v1/ingest`     | Ingest raw text                               | tenant |
| POST   | `/v1/ingest/file`| Ingest an uploaded file (PDF/MD/TXT/HTML)     | tenant |
| POST   | `/v1/query`      | Ask a question (RAG)                          | tenant |
| DELETE | `/v1/documents/{doc_id}` | Remove a document and its vectors     | tenant |
| POST   | `/admin/tenants` | Create a tenant (returns key once)            | admin |
| GET    | `/admin/tenants` | List tenants                                  | admin |
| GET/PATCH/DELETE | `/admin/tenants/{id}` | Get / update config / offboard   | admin |

**Tenant auth**: `/v1/*` routes take `Authorization: Bearer <api-key>`; the key
maps to a tenant and every operation is isolated to that tenant's index and
docstore. In single-tenant mode this may be optional. **Admin auth**: `/admin/*`
routes take the separate `ADMIN_API_KEY` (multi-tenant only).

---

## Project layout

```
app/
  config.py            # typed settings (pydantic-settings)
  main.py              # FastAPI app factory
  models.py            # request/response schemas
  deps.py              # auth + tenant resolution
  core/
    chunking.py        # token-aware overlapping chunker
    embeddings.py      # OpenAI embeddings w/ batching + retry + local fallback
    vector_store.py    # FAISS wrapper (Flat/IVF), per-tenant, persistence, delete
    docstore.py        # SQLite metadata store
    tenants.py         # tenant registry: config, quotas, issued keys (multi-tenant)
    ingest.py          # ingestion pipeline (parse → chunk → embed → index)
    retriever.py       # stage 1: vector | bm25 | hybrid retrieval + fusion
    bm25.py            # sparse BM25 inverted index (per tenant)
    fusion.py          # RRF + weighted fusion of dense & sparse results
    reranker.py        # stage 2: none/lexical(BM25)/cross-encoder/llm rerankers
    generator.py       # prompt assembly + LLM call + fallback
    rag.py             # end-to-end orchestration + per-tenant policy/quotas
  api/routes.py        # tenant HTTP endpoints
  api/admin.py         # /admin/* tenant control plane (multi-tenant)
  observability/metrics.py
eval/                  # offline evaluation harness
scripts/               # operational scripts
tests/                 # unit + API tests
docs/                  # the 6-stage lifecycle documentation
```

---

## Configuration

All configuration is environment-driven (12-factor). See
[`.env.example`](.env.example) for the full list. The most important knobs:

| Variable                 | Default            | Purpose                              |
|--------------------------|--------------------|--------------------------------------|
| `OPENAI_API_KEY`         | *(unset)*          | Enables real OpenAI; falls back if unset |
| `EMBEDDING_MODEL`        | `text-embedding-3-small` | Embedding model                |
| `GENERATION_MODEL`       | `gpt-4.1-mini`     | Answer-generation model              |
| `DEPLOYMENT_MODE`        | `multi_tenant`     | `single_tenant` or `multi_tenant`    |
| `SINGLE_TENANT_ID`       | `default`          | Fixed tenant in single-tenant mode   |
| `SINGLE_TENANT_REQUIRE_AUTH` | `false`        | Require a key in single-tenant mode  |
| `ADMIN_API_KEY`          | *(unset)*          | Guards `/admin/*`; unset disables it  |
| `TENANT_ISOLATION`       | `index_per_tenant` | `index_per_tenant` or `shared_namespace` |
| `FAISS_INDEX_TYPE`       | `flat`             | `flat` or `ivf` (per-tenant default) |
| `CHUNK_TOKENS`           | `400`              | Target chunk size                    |
| `CHUNK_OVERLAP`          | `60`               | Overlap between chunks               |
| `RETRIEVAL_TOP_K`        | `6`                | Chunks kept for the answer           |
| `RETRIEVAL_MODE`         | `hybrid`           | `vector`/`bm25`/`hybrid`             |
| `HYBRID_FUSION`          | `rrf`              | `rrf` or `weighted`                  |
| `RERANK_STRATEGY`        | `lexical`          | `none`/`lexical`/`cross_encoder`/`llm` |
| `RERANK_CANDIDATES`      | `20`               | Candidate pool size before reranking |
| `DATA_DIR`               | `./data`           | Where indices + docstore live        |

---

## Testing & evaluation

```bash
pytest -q                        # unit + API tests (no API key required)
python eval/run_eval.py          # retrieval + answer-quality metrics
python scripts/compare_rerank.py    # compare reranking strategies side by side
python scripts/compare_retrieval.py # compare vector / bm25 / hybrid modes
```

---

## Documentation

The `docs/` directory documents the entire lifecycle described in the design
brief, each with step-by-step instructions and pointers to the implementing code:

1. [Ideation & requirements](docs/01-ideation-and-requirements.md)
2. [System architecture & data model](docs/02-architecture.md)
3. [Implementation: indexing + query pipelines](docs/03-implementation.md)
4. [Deployment & ops](docs/04-deployment-and-ops.md)
5. [Evaluation, monitoring & maintenance](docs/05-evaluation-and-monitoring.md)
6. [Scaling & evolution](docs/06-scaling-and-evolution.md)
7. [Deployment modes: single-tenant vs multi-tenant](docs/07-deployment-modes.md)
8. [Reranking stage: two-stage retrieve-then-rerank](docs/08-reranking.md)
9. [Hybrid retrieval: BM25 + vector search](docs/09-hybrid-retrieval.md)
10. [Tenant isolation: index-per-tenant vs shared namespace](docs/10-tenant-isolation.md)

---

## License

MIT — see [`LICENSE`](LICENSE).
