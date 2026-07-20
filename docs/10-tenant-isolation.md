# 10. Tenant Isolation: Index-per-Tenant vs Shared Namespace

> **Goal of this doc:** explain the two ways this system isolates tenants' data
> in the vector layer, when to choose each, and how the shared-namespace mode
> stays exact and leak-free.

Applies to multi-tenant mode (see [doc 7](07-deployment-modes.md)). Selected by
one variable:

```bash
TENANT_ISOLATION=index_per_tenant   # default
# or
TENANT_ISOLATION=shared_namespace
```

Both are **physically/logically isolated and leak-free** — they differ in
*how* the vectors are partitioned, which drives cost and operational behavior.

---

## 10.1 The two strategies

### `index_per_tenant` (default) — physical isolation

Each tenant gets its **own FAISS index file**:

```
{DATA_DIR}/tenants/{tenant}/index.faiss
```

`VectorStore.for_tenant(t)` opens *that tenant's file only*. A query physically
cannot see another tenant's vectors — different file, different in-memory index.

**Pros**
- Hardest data boundary; the simplest isolation story for audits/compliance.
- Per-tenant **index type** (`flat` vs `ivf`) and independent rebuilds.
- Offboarding is a file delete (`drop_tenant`).

**Cons**
- One index/file per tenant → memory + file-handle overhead. With thousands of
  tiny tenants, that overhead dominates.

### `shared_namespace` — one index, partitioned by tenant id

All tenants' vectors live in **one shared FAISS index**:

```
{DATA_DIR}/shared/index.faiss
```

Each vector's *namespace* is its tenant id. A query is restricted to the
caller's vectors using a FAISS **`IDSelectorBatch`**, so FAISS scores **only**
that tenant's ids.

**Pros**
- One index for everyone → far less per-tenant overhead; ideal for many small
  tenants.
- A single index to warm, back up, and operate.

**Cons**
- All tenants share one index type and one rebuild lifecycle.
- Requires globally-unique vector ids (handled automatically, below).

## 10.2 Why shared-namespace is exact and leak-free

Naive shared-index designs "post-filter": fetch top-k from the whole index, then
drop other tenants' rows. That leaks recall — a small tenant's results can be
crowded out by a big tenant's vectors and return nothing.

This system avoids that. `SearchHit` selection is scoped **before** scoring:

```python
# app/core/vector_store.py — SharedNamespaceView.search
ids = self._store._tenant_ids(self._tenant)      # this tenant's vector ids
selector = faiss.IDSelectorBatch(ids)            # restrict the search
return self._shared.search(query, top_k, selector=selector)
```

FAISS evaluates the query against *only* those ids (works for both `flat` and
`ivf` via `SearchParameters` / `SearchParametersIVF`), so:

- **Exact**: the top-k is the true top-k *within the namespace* — no starvation.
- **Leak-free**: ids from other tenants are never scored or returned.

Belt and suspenders: even the metadata hydration is tenant-scoped
(`docstore.get_chunks(tenant, ids)` filters by `tenant`), so a stray id could
never resolve to another tenant's content.

## 10.3 Globally-unique ids (handled for you)

In `index_per_tenant` mode, ids only need to be unique within a tenant's file.
In `shared_namespace` mode they must be unique across the whole shared index, or
two tenants could collide on id `0`. Ingestion allocates accordingly
(`app/core/ingest.py`):

```python
if settings.tenant_isolation == "shared_namespace":
    base_id = docstore.max_vector_id_all() + 1   # global sequence
else:
    base_id = docstore.max_vector_id(tenant) + 1 # per-tenant sequence
```

## 10.4 What is (and isn't) shared

| Layer | `index_per_tenant` | `shared_namespace` |
|-------|--------------------|--------------------|
| FAISS vectors | one file **per tenant** | **one** shared file, id-scoped per query |
| BM25 sparse index | per tenant (in-memory) | per tenant (in-memory) |
| SQLite metadata | shared DB, **row-scoped** by tenant | shared DB, **row-scoped** by tenant |
| Per-tenant index type | ✅ yes | ✗ one type for all |
| Whole-tenant drop | delete file | remove tenant's ids from shared index |

> Note: the metadata store (`docstore.db`) and the BM25 layer behave the same in
> both modes — only the FAISS vector partitioning changes.

## 10.5 Step-by-step: run shared-namespace mode

```bash
# 1. Configure
export DEPLOYMENT_MODE=multi_tenant
export TENANT_ISOLATION=shared_namespace
export ADMIN_API_KEY=change-me
uvicorn app.main:app --port 8000

# 2. Onboard two tenants
KA=$(curl -s -X POST localhost:8000/admin/tenants -H "Authorization: Bearer change-me" \
      -H 'content-type: application/json' -d '{"tenant_id":"acme"}'   | jq -r .api_key)
KB=$(curl -s -X POST localhost:8000/admin/tenants -H "Authorization: Bearer change-me" \
      -H 'content-type: application/json' -d '{"tenant_id":"globex"}' | jq -r .api_key)

# 3. Each ingests into the SAME shared index, different namespace
curl -s localhost:8000/v1/ingest -H "Authorization: Bearer $KA" -H 'content-type: application/json' \
  -d '{"documents":[{"doc_id":"x","source":"a.md","text":"Acme confidential: launch code ZX-9."}]}'
curl -s localhost:8000/v1/ingest -H "Authorization: Bearer $KB" -H 'content-type: application/json' \
  -d '{"documents":[{"doc_id":"y","source":"b.md","text":"Globex revenue grew twelve percent."}]}'

# 4. Globex CANNOT retrieve Acme's secret — isolation holds
curl -s localhost:8000/v1/query -H "Authorization: Bearer $KB" \
  -H 'content-type: application/json' -d '{"question":"what is launch code ZX-9?"}'
#  -> citations never include Acme's doc "x"
```

On disk you'll find a single `data/shared/index.faiss` and **no** `data/tenants/`
directory — the observable difference from the default mode.

## 10.6 Choosing a mode

| If you… | Use |
|---------|-----|
| Have a handful of tenants, or need per-tenant index types / the strictest boundary / independent rebuilds | `index_per_tenant` |
| Have many small tenants and want to minimize per-tenant memory & file overhead | `shared_namespace` |
| Are unsure | start with `index_per_tenant` (the default) |

Both keep the same API, auth, quotas, and per-tenant prompt templates — only the
vector partitioning changes, so you can switch strategies for a fresh deployment
without touching application code.

> **Switching an existing deployment** requires a reindex (the on-disk layout
> differs): stand up the new mode and re-ingest from your source docs
> (`scripts/ingest_dir.py`) per tenant.

## 10.7 Reference map

| Concern | Code |
|---------|------|
| Mode selection | `app/config.py` (`tenant_isolation`) |
| Per-tenant index / shared view | `app/core/vector_store.py` (`VectorStore`, `SharedNamespaceView`) |
| Exact per-namespace search | `TenantIndex.search(..., selector=IDSelectorBatch)` |
| Global id allocation | `app/core/ingest.py`, `docstore.max_vector_id_all` |
| Whole-tenant drop | `VectorStore.drop_tenant`, `RagEngine.purge_tenant` |

Back to the [docs index](README.md).
