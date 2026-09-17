"""Compile the chat graph: START → load_history → agent → save_history → END."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.graph.nodes import (
    AGENT_NODE,
    LOAD_HISTORY_NODE,
    SAVE_HISTORY_NODE,
    make_agent_node,
    make_load_history_node,
    make_save_history_node,
)
from app.graph.state import GraphState
from app.llm.factory import create_chat_model
from app.memory.store import InMemorySessionStore, get_session_store

_compiled_graph: CompiledStateGraph | None = None


def build_graph(
    llm: BaseChatModel | None = None,
    *,
    tools: Sequence[Any] | None = None,
    store: InMemorySessionStore | None = None,
) -> CompiledStateGraph:
    """Build a compiled graph. Pass ``llm`` / ``store`` from tests to isolate runs."""

    chat_model = create_chat_model() if llm is None else llm
    session_store = store if store is not None else get_session_store()
    graph = StateGraph(GraphState)
    graph.add_node(LOAD_HISTORY_NODE, make_load_history_node(session_store))
    graph.add_node(AGENT_NODE, make_agent_node(chat_model, tools=tools))
    graph.add_node(SAVE_HISTORY_NODE, make_save_history_node(session_store))
    graph.add_edge(START, LOAD_HISTORY_NODE)
    graph.add_edge(LOAD_HISTORY_NODE, AGENT_NODE)
    graph.add_edge(AGENT_NODE, SAVE_HISTORY_NODE)
    graph.add_edge(SAVE_HISTORY_NODE, END)
    return graph.compile()


def get_graph(llm: BaseChatModel | None = None) -> CompiledStateGraph:
    """Return the process-wide compiled graph, creating it on first use."""

    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph(llm)
    return _compiled_graph
