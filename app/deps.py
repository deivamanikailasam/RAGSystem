"""Request dependencies: engine singleton + mode-aware auth.

Auth here is intentionally simple (static bearer keys and registry-issued keys)
so the system is self-contained. In production, replace the token→tenant logic
with an OIDC/JWT validator and resolve the tenant + roles from verified claims.
The call site (a FastAPI dependency returning a tenant id) does not change.

Two deployment modes (see docs/07-deployment-modes.md):

* **single_tenant** — every request maps to ``settings.single_tenant_id``. Auth
  is optional (``SINGLE_TENANT_REQUIRE_AUTH``); when required, any configured
  static key is accepted.
* **multi_tenant** — the bearer token resolves to a tenant via the static
  ``API_KEYS`` map or a registry-issued key. The tenant must exist and be
  enabled. ``/admin/*`` endpoints require the separate ``ADMIN_API_KEY``.
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, Header, HTTPException, status

from app.config import Settings, get_settings
from app.core.rag import RagEngine


@lru_cache
def get_engine() -> RagEngine:
    """Process-wide RagEngine, built once on first request."""
    return RagEngine(get_settings())


def _bearer_token(authorization: str | None) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return authorization.split(" ", 1)[1].strip()


def require_tenant(
    authorization: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
    engine: RagEngine = Depends(get_engine),
) -> str:
    """Resolve the caller's tenant, enforcing the active deployment mode."""
    # ---- single-tenant mode ---------------------------------------------- #
    if settings.is_single_tenant:
        if settings.single_tenant_require_auth:
            token = _bearer_token(authorization)
            if token not in settings.api_key_map:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid API key.",
                )
        return settings.single_tenant_id

    # ---- multi-tenant mode ----------------------------------------------- #
    token = _bearer_token(authorization)

    # 1) static env key → tenant, or 2) registry-issued key → tenant.
    tenant_id = settings.api_key_map.get(token)
    if tenant_id is None:
        record = engine.tenants.get_by_api_key(token)
        tenant_id = record.tenant_id if record else None
    if tenant_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key."
        )

    record = engine.tenants.get(tenant_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Unknown tenant."
        )
    if record.disabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Tenant is disabled."
        )
    return tenant_id


def require_admin(
    authorization: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> bool:
    """Guard for /admin/* tenant-management endpoints (multi-tenant only)."""
    if settings.is_single_tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Admin API is unavailable in single-tenant mode.",
        )
    if not settings.admin_api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin API is disabled (set ADMIN_API_KEY to enable).",
        )
    token = _bearer_token(authorization)
    if token != settings.admin_api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Invalid admin key."
        )
    return True
