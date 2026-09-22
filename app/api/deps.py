"""FastAPI dependencies for the chat HTTP API."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from fastapi import Request


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
    """Return the compiled chat graph from ``app.state``.

    Lifespan attaches the graph at startup. This dependency rebuilds it on
    first ``GET /chat`` only if ``app.state.graph`` is still unset (fallback).
    """
    graph = getattr(request.app.state, "graph", None)
    if graph is not None:
        return graph

    from app.graph.builder import get_graph as load_graph

    graph = load_graph()
    request.app.state.graph = graph
    return graph
