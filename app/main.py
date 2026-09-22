"""FastAPI application entrypoint.

Lifespan configures logging, the LLM, weather MCP client config, and the chat
graph. It does **not** open a connection to the remote weather MCP — that
happens lazily on the first weather tool call.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.config import settings
from app.core.exceptions import ChatServiceError, ProviderConfigError
from app.core.logging import setup_logging
from app.graph.builder import get_graph
from app.llm.factory import create_chat_model
from app.tools.mcp_weather import configure_weather_client

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    setup_logging()
    logger.info(
        "Starting AI-Powered Chat Service (MCP is not connected at startup)"
    )

    # Fail fast for unknown / incomplete real-provider config. Never mock.
    app.state.llm = create_chat_model(settings)
    logger.info(
        "LLM_PROVIDER=%s client=%s",
        settings.llm_provider,
        type(app.state.llm).__name__,
    )

    # Config only: URL + transport. No handshake, no get_tools() on the wire.
    configure_weather_client(
        url=settings.mcp_weather_url,
        transport=settings.mcp_weather_transport,
    )

    app.state.graph = get_graph(llm=app.state.llm)
    logger.info("Chat graph attached")

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
