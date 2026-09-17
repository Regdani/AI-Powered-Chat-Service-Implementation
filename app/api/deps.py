"""FastAPI dependencies for the chat HTTP API."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from fastapi import Request

from app.core.exceptions import ChatServiceError


class ChatGraph(Protocol):
    """Compiled LangGraph (or test double) used by ``GET /chat``."""

    async def ainvoke(
        self,
        input: Mapping[str, Any],
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        """Run one chat turn. Input/output follow ``GraphState``."""
        ...


def get_graph(request: Request) -> ChatGraph:
    """Return the compiled graph, loading it lazily on first chat request.

    Startup must not depend on the remote weather MCP. The graph object is
    expected to bind tools that connect lazily. Missing graph wiring is a 503
    rather than a crash of ``GET /health``.
    """
    graph = getattr(request.app.state, "graph", None)
    if graph is not None:
        return graph

    try:
        from app.graph.builder import get_graph as load_graph
    except ImportError as exc:
        raise ChatServiceError(
            "Chat graph is not configured.",
            status_code=503,
        ) from exc

    graph = load_graph()
    request.app.state.graph = graph
    return graph
