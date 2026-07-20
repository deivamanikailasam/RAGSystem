# 7. Deployment Modes: Single-Tenant vs Multi-Tenant

QASystem ships **two deployment modes** behind a single config switch, so you
can start with an internal doc bot on a limited corpus and grow into a
multi-tenant platform without re-architecting.

| | **single_tenant** | **multi_tenant** |
|---|---|---|
| Corpus | one implicit corpus | one FAISS index per tenant |
| Auth | optional (open or one key) | required; per-tenant keys |
| Tenant registry | not used | source of truth (SQLite) |
| Admin control plane | disabled (404) | `/admin/*` (admin-key guarded) |
| Per-tenant prompt / index type | n/a | yes |
| Quotas (docs, queries/day) | none | enforced per tenant |
| Best for | internal eng/product docs | SaaS / many customers or teams |

Select the mode with one variable:

```bash
DEPLOYMENT_MODE=single_tenant    # or: multi_tenant  (default)
```

Both modes share the exact same ingestion, retrieval, and generation pipelines
(stages [3](03-implementation.md)); only tenancy, auth, and governance differ.

---

## Part A — Single-tenant internal doc QA (limited corpus)

Use this when one team owns one corpus and you want the least ceremony.

### Step 1 — Configure

`.env`:
```bash
DEPLOYMENT_MODE=single_tenant
SINGLE_TENANT_ID=internal          # the fixed tenant every request maps to
SINGLE_TENANT_REQUIRE_AUTH=false   # true to require a bearer key
# OPENAI_API_KEY=sk-...            # omit to run the offline fallback
FAISS_INDEX_TYPE=flat              # exact search; ideal for a limited corpus
```

**How it works:** `app/deps.py:require_tenant` short-circuits to
`SINGLE_TENANT_ID` for every request. The tenant registry is never consulted,
and no quotas apply (`app/core/rag.py:_tenant_config` returns `None`).

### Step 2 — Ingest the corpus

In-process (no server), great for seeding a limited corpus:
```bash
python scripts/ingest_dir.py ./company-docs --tenant internal
```
Or over HTTP:
```bash
curl -s localhost:8000/v1/ingest -H 'content-type: application/json' \
  -d '{"documents":[{"doc_id":"handbook","source":"handbook.md","text":"..."}]}'
```
> With `SINGLE_TENANT_REQUIRE_AUTH=false` no `Authorization` header is needed —
> appropriate on a trusted internal network behind SSO/VPN.

### Step 3 — Query

```bash
curl -s localhost:8000/v1/query -H 'content-type: application/json' \
  -d '{"question":"What is our PTO policy?"}' | jq
```

### Step 4 — (Optional) turn on auth

Set `SINGLE_TENANT_REQUIRE_AUTH=true` and `API_KEYS=some-key:internal`. Now every
request must send `Authorization: Bearer some-key`; the tenant is still fixed.

### Step 5 — Inspect

- `GET /health` → confirms `deployment_mode: single_tenant`.
- `GET /v1/me` → document/vector counts for the single corpus.
- `GET /metrics` → latency, tokens, query counts.

**When to graduate to multi-tenant:** you need to serve separate customers or
departments with isolated data, per-tenant limits, or self-service onboarding.
Because vectors are already stored per tenant, the migration is a config change
plus registering your existing tenant (below) — no data migration.

---

## Part B — Multi-tenant platform (per-tenant FAISS indices)

Use this to serve many isolated tenants from shared services.

### Step 1 — Configure

`.env`:
```bash
DEPLOYMENT_MODE=multi_tenant
ADMIN_API_KEY=change-me-admin-secret   # guards /admin/*; unset = admin disabled
# OPENAI_API_KEY=sk-...
FAISS_INDEX_TYPE=flat                   # per-tenant default; override per tenant
# API_KEYS=demo-key:demo                # optional static keys (auto-registered)
```

**Isolation model** (see [architecture §2.5](02-architecture.md)):
- **Vectors** — one FAISS file per tenant at
  `{DATA_DIR}/tenants/{tenant}/index.faiss` (`vector_store.py`).
- **Metadata** — every SQLite row is tenant-keyed (`docstore.py`).
- **Identity** — a bearer key resolves to exactly one tenant
  (`deps.py:require_tenant`). There is no cross-tenant code path.

### Step 2 — Onboard a tenant (admin control plane)

The admin API (`app/api/admin.py`) manages the tenant lifecycle and is guarded
by `ADMIN_API_KEY`.

```bash
curl -s -X POST localhost:8000/admin/tenants \
  -H "Authorization: Bearer $ADMIN_API_KEY" -H 'content-type: application/json' \
  -d '{
        "tenant_id": "acme",
        "name": "Acme Corp",
        "index_type": "flat",
        "max_documents": 5000,
        "max_queries_per_day": 10000,
        "prompt_template": "You are Acme’s support assistant. Answer only from context."
      }'
```
Response (the API key is shown **exactly once** — store it now):
```json
{
  "tenant": {"tenant_id":"acme","index_type":"flat","max_documents":5000, ...},
  "api_key": "qas_XXXXXXXXXXXXXXXXXXXX"
}
```
Only the key's SHA-256 **hash** is persisted (`tenants.py:hash_api_key`); the
plaintext is never stored.

### Step 3 — The tenant ingests & queries with its issued key

```bash
KEY=qas_XXXXXXXXXXXXXXXXXXXX

curl -s localhost:8000/v1/ingest -H "Authorization: Bearer $KEY" \
  -H 'content-type: application/json' \
  -d '{"documents":[{"doc_id":"faq","source":"faq.md","text":"..."}]}'

curl -s localhost:8000/v1/query -H "Authorization: Bearer $KEY" \
  -H 'content-type: application/json' \
  -d '{"question":"How do I reset my password?"}' | jq
```

All work is scoped to `acme`: its own FAISS index, its own prompt template, its
own quotas.

### Step 4 — Governance in action

- **Per-tenant prompt** — `acme`'s `prompt_template` overrides the default system
  prompt at generation time (`rag.py:query` → `generator.generate`).
- **Per-tenant index type** — a large tenant can run `ivf` while others stay
  `flat`; the type is fixed when the tenant's index is first created
  (`vector_store.py:for_tenant`).
- **Quotas** — exceeding `max_documents` (on ingest) or `max_queries_per_day`
  (on query) returns HTTP **429** with a `quota` field. `0` means unlimited.
- **Disable a tenant** — `PATCH /admin/tenants/acme {"disabled": true}`; its
  requests then return **403** until re-enabled.

### Step 5 — Manage the fleet

```bash
# list all tenants
curl -s localhost:8000/admin/tenants -H "Authorization: Bearer $ADMIN_API_KEY"

# inspect one
curl -s localhost:8000/admin/tenants/acme -H "Authorization: Bearer $ADMIN_API_KEY"

# change a quota
curl -s -X PATCH localhost:8000/admin/tenants/acme \
  -H "Authorization: Bearer $ADMIN_API_KEY" -H 'content-type: application/json' \
  -d '{"max_queries_per_day": 50000}'

# offboard: delete registry entry AND purge vectors + documents
curl -s -X DELETE 'localhost:8000/admin/tenants/acme?purge=true' \
  -H "Authorization: Bearer $ADMIN_API_KEY"
```

A tenant (or the platform operator) can check usage at any time:
```bash
curl -s localhost:8000/v1/me -H "Authorization: Bearer $KEY"
# {"tenant":"acme","documents":42,"vectors":517,"queries_today":133,
#  "quotas":{"max_documents":5000,"max_queries_per_day":10000}}
```

### Step 6 — Static keys (optional)

For fixed internal integrations you can still use `API_KEYS=key:tenant`. Any
tenant named there is auto-registered in the registry at startup
(`rag.py:_seed_static_tenants`), so both static and issued keys work side by side.

---

## Migrating single → multi

1. Set `DEPLOYMENT_MODE=multi_tenant` and an `ADMIN_API_KEY`.
2. Register your existing corpus as a tenant whose `tenant_id` matches your old
   `SINGLE_TENANT_ID`, supplying your existing key via the `api_key` field so
   current clients keep working:
   ```bash
   curl -X POST localhost:8000/admin/tenants -H "Authorization: Bearer $ADMIN_API_KEY" \
     -H 'content-type: application/json' \
     -d '{"tenant_id":"internal","api_key":"some-key"}'
   ```
   The on-disk index at `tenants/internal/index.faiss` and its docstore rows are
   already in the right place — **no data migration needed**.
3. Onboard additional tenants via the admin API.

---

## Reference: what each mode touches in code

| Concern | Code |
|---------|------|
| Mode switch & knobs | `app/config.py` (`deployment_mode`, `single_tenant_*`, `admin_api_key`) |
| Mode-aware auth | `app/deps.py:require_tenant`, `require_admin` |
| Tenant registry | `app/core/tenants.py:TenantRegistry` |
| Per-tenant config + quotas | `app/core/rag.py` (`_tenant_config`, `ingest`, `query`) |
| Admin control plane | `app/api/admin.py` |
| Per-tenant index | `app/core/vector_store.py:VectorStore.for_tenant` |
| Usage / stats | `GET /v1/me` → `rag.py:tenant_stats` |
| Error → HTTP mapping | `app/main.py` exception handlers |

Back to the [docs index](README.md).
