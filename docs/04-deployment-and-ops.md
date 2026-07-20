# 4. Deployment & Ops

> **Goal of this stage:** run the system reliably, securely, and observably.

---

## 4.1 Packaging & running

**Local (dev):**
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

**Container:**
```bash
docker compose up --build          # persists index to a named volume
```
The `Dockerfile` installs deps in a cached layer and runs a **long-lived**
uvicorn process. This matters: FAISS indices are held in memory, so a warm,
long-running process massively outperforms cold serverless for large indices.

## 4.2 Infrastructure topology

A pragmatic production layout:

| Component | Choice | Notes |
|-----------|--------|-------|
| API service | FastAPI on ECS/EKS or a VM | long-lived, warm FAISS in RAM |
| Ingestion workers | separate service (Celery/SQS/Lambda) | keeps heavy embedding off the query path |
| Raw docs | S3 / object store | source of truth for re-indexing |
| Metadata DB | Postgres (swap `docstore.py`) | tenants, chunks, permissions |
| Vector index | EBS / persistent volume per replica | or shared network storage |
| Secrets | AWS Secrets Manager / Vault | never bake keys into images |

> **Serverless caveat:** for big FAISS indices prefer long-lived pods over pure
> Lambda — you don't want to reload a multi-GB index per invocation.

## 4.3 Configuration (12-factor)

All config is environment-driven via `app/config.py` (`.env` for local). Nothing
is hard-coded. Key groups: OpenAI, chunking, retrieval, FAISS, storage, auth,
guardrails. See `.env.example`.

## 4.4 Security

**AuthN/AuthZ.** `app/deps.py:require_tenant` validates a bearer key and maps it
to a tenant. For production, replace the static `API_KEYS` map with an OIDC/JWT
validator and resolve tenant + roles from verified claims — the dependency's
*signature* (returns a tenant id) stays the same, so nothing downstream changes.

**Secrets.** `OPENAI_API_KEY` is read from the environment only. Inject it from a
secrets manager at deploy time; never commit it (`.gitignore` excludes `.env`).

**Transport & data.** Terminate TLS at the ingress. Encrypt the docs bucket,
the metadata DB, and the index volume at rest.

**Tenant isolation.** Enforced physically (per-tenant FAISS files) and logically
(tenant-scoped SQL). There is no endpoint that reads across tenants.

**Sensitive corpora.** If documents can't leave your boundary, keep FAISS +
docstore on-prem and either use a private LLM endpoint or a fully local model —
only `generator.py` / `embeddings.py` need to change.

## 4.5 Operational practices

**Error handling & resilience.**
- OpenAI calls retry with exponential backoff
  (`embeddings.py:_embed_batch_with_retry`, and the OpenAI SDK's built-in
  retries in `generator.py`).
- Retrieval that returns nothing degrades gracefully to a grounded "I don't
  know" instead of erroring.
- Auth failures return `401` with `WWW-Authenticate`, not `500`.

**Health & readiness.** `GET /health` returns status, version, and whether
OpenAI is enabled — wire it to your orchestrator's liveness/readiness probes
(the compose file already does).

**Rate limiting.** Add it at the ingress/gateway, or as FastAPI middleware keyed
by tenant, using the per-tenant identity from `require_tenant`.

**Index durability.** Indices are persisted to disk on every write
(`TenantIndex.persist`). Back up `{DATA_DIR}` (or snapshot the volume). Because
S3 holds the raw docs, you can always rebuild an index from scratch with
`scripts/ingest_dir.py`.

## 4.6 Observability

- **Metrics:** `GET /metrics` exposes counters (`queries`, `documents_ingested`,
  `tokens_total`) and latency summaries (`retrieval_ms`, `generation_ms`,
  `ingest_ms`) with p50/p95/p99 (`observability/metrics.py`). In production,
  export these to Prometheus/OpenTelemetry — the call sites (`increment`,
  `observe`, `timer`) map directly onto those SDKs.
- **Cost tracking:** token usage is captured per query and summed into
  `tokens_total`, enabling per-tenant cost attribution.
- **Tracing:** each query carries a `request_id`; log retrieval hits, the
  assembled prompt, and the answer against it to debug retrieval/prompt issues
  end to end.

## 4.7 Step-by-step deploy checklist

1. [ ] Provision object store (docs), metadata DB, index volume, secrets store.
2. [ ] Put `OPENAI_API_KEY` in the secrets manager; inject as env at runtime.
3. [ ] Replace the static `API_KEYS` auth with your IdP (edit `require_tenant`).
4. [ ] Point `DATA_DIR` at the persistent volume.
5. [ ] Wire `/health` to liveness/readiness; scrape `/metrics`.
6. [ ] Add ingress TLS + rate limiting.
7. [ ] Schedule index backups (or rely on rebuild-from-S3).
8. [ ] Run `pytest` and `python eval/run_eval.py` in CI as gates.

Proceed to [evaluation & monitoring](05-evaluation-and-monitoring.md).
