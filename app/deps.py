"""Request dependencies: engine singleton + API-key → tenant auth.

Auth here is intentionally simple (static bearer keys mapped to tenants) so the
system is self-contained. In production, replace :func:`require_tenant` with an
OIDC/JWT validator and resolve the tenant + roles from verified claims. The
call site (a FastAPI dependency returning a tenant id) does not change.
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


def require_tenant(
    authorization: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> str:
    """Resolve the caller's tenant from a bearer API key.

    Raises 401 if the header is missing/malformed or the key is unknown.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = authorization.split(" ", 1)[1].strip()
    tenant = settings.api_key_map.get(token)
    if tenant is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )
    return tenant
