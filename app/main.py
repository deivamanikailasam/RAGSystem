"""FastAPI application factory.

Run with::

    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.api.routes import router
from app.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        mode = "OpenAI" if settings.use_openai else "local-fallback (offline)"
        logging.getLogger("qasystem").info(
            "QASystem %s starting in %s mode (index=%s)",
            __version__, mode, settings.faiss_index_type,
        )
        yield

    app = FastAPI(
        title="QASystem — RAG Document Q&A",
        version=__version__,
        description="FAISS + OpenAI retrieval-augmented document question answering.",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()
