"""FastAPI application factory.

Run with::

    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app import __version__
from app.api.admin import router as admin_router
from app.api.routes import router
from app.config import get_settings
from app.core.tenants import (
    QuotaExceeded,
    TenantDisabled,
    TenantError,
    TenantNotFound,
)


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        mode = "OpenAI" if settings.use_openai else "local-fallback (offline)"
        logging.getLogger("ragsystem").info(
            "RAGSystem %s starting in %s mode (index=%s)",
            __version__, mode, settings.faiss_index_type,
        )
        yield

    app = FastAPI(
        title="RAGSystem — RAG Document Q&A",
        version=__version__,
        description="FAISS + OpenAI retrieval-augmented document question answering.",
        lifespan=lifespan,
    )
    app.include_router(router)
    app.include_router(admin_router)

    # Translate tenant/quota errors raised deep in the engine into HTTP codes.
    @app.exception_handler(QuotaExceeded)
    async def _quota_handler(_: Request, exc: QuotaExceeded) -> JSONResponse:
        return JSONResponse(
            status_code=429,
            content={"detail": str(exc), "quota": exc.quota, "limit": exc.limit},
        )

    @app.exception_handler(TenantNotFound)
    async def _not_found_handler(_: Request, exc: TenantNotFound) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(TenantDisabled)
    async def _disabled_handler(_: Request, exc: TenantDisabled) -> JSONResponse:
        return JSONResponse(status_code=403, content={"detail": str(exc)})

    @app.exception_handler(TenantError)
    async def _tenant_error_handler(_: Request, exc: TenantError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    return app


app = create_app()
