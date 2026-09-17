"""FastAPI application entrypoint.

Lifespan configures logging only. It does **not** open a connection to the
remote weather MCP — that happens lazily on the first weather tool call.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.exceptions import ChatServiceError, ProviderConfigError
from app.core.logging import setup_logging

logger = logging.getLogger(__name__)


def _try_attach_graph(app: FastAPI) -> None:
    """Compile the chat graph at startup when the graph package is present.

    Fail-fast for an explicit real LLM provider happens inside the LLM factory
    / graph builder. If those modules are not imported yet, ``GET /health``
    still works and ``GET /chat`` returns 503 until they are wired.
    """
    app.state.graph = None
    try:
        from app.graph.builder import get_graph
    except ImportError:
        logger.info("Chat graph is not wired yet; GET /health remains available")
        return

    app.state.graph = get_graph()
    logger.info("Chat graph attached")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    setup_logging()
    logger.info(
        "Starting AI-Powered Chat Service (MCP is not connected at startup)"
    )
    try:
        from app.core.config import settings

        logger.info("LLM_PROVIDER=%s", getattr(settings, "llm_provider", "mock"))
    except ImportError:
        logger.info("Settings module not loaded")

    _try_attach_graph(app)
    yield
    logger.info("Shutting down AI-Powered Chat Service")


def create_app() -> FastAPI:
    """Build the ASGI app (also used by tests)."""
    application = FastAPI(
        title="AI-Powered Chat Service",
        description=(
            "Async GET chat API backed by a LangGraph agent. "
            "Swagger UI: `/docs`. Default bind: `localhost:18080`."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
        servers=[
            {
                "url": "http://localhost:18080",
                "description": "Local development",
            }
        ],
    )
    application.include_router(router)

    @application.exception_handler(ChatServiceError)
    async def handle_chat_error(
        request: Request, exc: ChatServiceError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.message},
        )

    @application.exception_handler(ProviderConfigError)
    async def handle_provider_config_error(
        request: Request, exc: ProviderConfigError
    ) -> JSONResponse:
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    return application


app = create_app()
