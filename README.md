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
   user query ──▶ /v1/query ──▶ embed ──▶ FAISS top-k ──▶ filter/rerank
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

## API

| Method | Path             | Description                                   |
|--------|------------------|-----------------------------------------------|
| GET    | `/health`        | Liveness/readiness probe                      |
| GET    | `/metrics`       | In-process counters & latency histograms      |
| POST   | `/v1/ingest`     | Ingest raw text or uploaded files             |
| POST   | `/v1/query`      | Ask a question (RAG)                          |
| DELETE | `/v1/documents/{doc_id}` | Remove a document and its vectors     |

All `/v1/*` routes require an `Authorization: Bearer <api-key>` header. The key
maps to a tenant, and every operation is isolated to that tenant's index and
docstore. See [`docs/04-deployment-and-ops.md`](docs/04-deployment-and-ops.md).

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
    vector_store.py    # FAISS wrapper (Flat/IVF), persistence, delete
    docstore.py        # SQLite metadata store
    ingest.py          # ingestion pipeline (parse → chunk → embed → index)
    retriever.py       # similarity search + metadata filter + rerank
    generator.py       # prompt assembly + LLM call + fallback
    rag.py             # end-to-end query orchestration
  api/routes.py        # HTTP endpoints
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
| `FAISS_INDEX_TYPE`       | `flat`             | `flat` or `ivf`                      |
| `CHUNK_TOKENS`           | `400`              | Target chunk size                    |
| `CHUNK_OVERLAP`          | `60`               | Overlap between chunks               |
| `RETRIEVAL_TOP_K`        | `6`                | Chunks retrieved per query           |
| `DATA_DIR`               | `./data`           | Where indices + docstore live        |

---

## Testing & evaluation

```bash
pytest -q                      # unit + API tests (no API key required)
python eval/run_eval.py        # retrieval + answer-quality metrics
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

---

## License

MIT — see [`LICENSE`](LICENSE).
