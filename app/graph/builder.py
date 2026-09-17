"""Compile the chat graph: START → agent → END."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.graph.nodes import AGENT_NODE, make_agent_node
from app.graph.state import GraphState
from app.llm.factory import create_chat_model

_compiled_graph: CompiledStateGraph | None = None


def build_graph(
    llm: BaseChatModel | None = None,
    *,
    tools: Sequence[Any] | None = None,
) -> CompiledStateGraph:
    """Build a compiled graph. Pass ``llm`` from tests to inject a scripted model."""

    chat_model = create_chat_model() if llm is None else llm
    graph = StateGraph(GraphState)
    graph.add_node(AGENT_NODE, make_agent_node(chat_model, tools=tools))
    graph.add_edge(START, AGENT_NODE)
    graph.add_edge(AGENT_NODE, END)
    return graph.compile()


def get_graph(llm: BaseChatModel | None = None) -> CompiledStateGraph:
    """Return the process-wide compiled graph, creating it on first use."""

    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph(llm)
    return _compiled_graph
