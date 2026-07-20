"""Admin control plane for the multi-tenant platform.

These endpoints manage the tenant lifecycle and are guarded by the separate
``ADMIN_API_KEY`` (see ``app/deps.py:require_admin``). They are unavailable in
single-tenant mode (the guard returns 404).

Workflow::

    # 1. create a tenant — the API key is returned exactly once
    curl -X POST localhost:8000/admin/tenants \
      -H 'Authorization: Bearer $ADMIN_API_KEY' -H 'content-type: application/json' \
      -d '{"tenant_id":"acme","name":"Acme Corp","max_documents":1000}'
    # -> {"tenant": {...}, "api_key": "qas_..."}

    # 2. the tenant now ingests/queries with that key
    curl localhost:8000/v1/query -H 'Authorization: Bearer qas_...' ...
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.core.rag import RagEngine
from app.core.tenants import Tenant, TenantExists, TenantNotFound
from app.deps import get_engine, require_admin
from app.models import (
    TenantCreate,
    TenantCreateResponse,
    TenantInfo,
    TenantListResponse,
    TenantUpdate,
)

router = APIRouter(prefix="/admin", tags=["admin"])


def _to_info(t: Tenant) -> TenantInfo:
    return TenantInfo(**t.public_dict())


@router.post("/tenants", response_model=TenantCreateResponse)
def create_tenant(
    body: TenantCreate,
    _: bool = Depends(require_admin),
    engine: RagEngine = Depends(get_engine),
) -> TenantCreateResponse:
    try:
        tenant, api_key = engine.tenants.create(
            tenant_id=body.tenant_id,
            name=body.name,
            api_key=body.api_key,
            prompt_template=body.prompt_template,
            index_type=body.index_type,
            max_documents=body.max_documents,
            max_queries_per_day=body.max_queries_per_day,
        )
    except TenantExists as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return TenantCreateResponse(tenant=_to_info(tenant), api_key=api_key)


@router.get("/tenants", response_model=TenantListResponse)
def list_tenants(
    _: bool = Depends(require_admin),
    engine: RagEngine = Depends(get_engine),
) -> TenantListResponse:
    return TenantListResponse(tenants=[_to_info(t) for t in engine.tenants.list()])


@router.get("/tenants/{tenant_id}", response_model=TenantInfo)
def get_tenant(
    tenant_id: str,
    _: bool = Depends(require_admin),
    engine: RagEngine = Depends(get_engine),
) -> TenantInfo:
    tenant = engine.tenants.get(tenant_id)
    if tenant is None:
        raise HTTPException(status_code=404, detail="Tenant not found.")
    return _to_info(tenant)


@router.patch("/tenants/{tenant_id}", response_model=TenantInfo)
def update_tenant(
    tenant_id: str,
    body: TenantUpdate,
    _: bool = Depends(require_admin),
    engine: RagEngine = Depends(get_engine),
) -> TenantInfo:
    try:
        tenant = engine.tenants.update(
            tenant_id,
            name=body.name,
            prompt_template=body.prompt_template,
            index_type=body.index_type,
            max_documents=body.max_documents,
            max_queries_per_day=body.max_queries_per_day,
            disabled=body.disabled,
        )
    except TenantNotFound as exc:
        raise HTTPException(status_code=404, detail="Tenant not found.") from exc
    return _to_info(tenant)


@router.delete("/tenants/{tenant_id}")
def delete_tenant(
    tenant_id: str,
    purge: bool = True,
    _: bool = Depends(require_admin),
    engine: RagEngine = Depends(get_engine),
) -> dict[str, object]:
    """Delete a tenant. With ``purge=true`` (default) also deletes its vectors
    and documents; set ``purge=false`` to keep the data and only remove the
    registry entry."""
    if engine.tenants.get(tenant_id) is None:
        raise HTTPException(status_code=404, detail="Tenant not found.")
    if purge:
        engine.purge_tenant(tenant_id)
    engine.tenants.delete(tenant_id)
    return {"tenant_id": tenant_id, "deleted": True, "purged": purge}
