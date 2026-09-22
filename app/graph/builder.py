"""Compile the chat graph: START → load_history → agent ⇄ tools → translate → save → END."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from app.graph.nodes import (
    AGENT_NODE,
    LOAD_HISTORY_NODE,
    SAVE_HISTORY_NODE,
    TOOLS_NODE,
    TRANSLATE_NODE,
    make_agent_node,
    make_load_history_node,
    make_save_history_node,
    make_translate_node,
    route_after_agent,
)
from app.graph.state import GraphState
from app.llm.factory import create_chat_model
from app.memory.store import InMemorySessionStore, get_session_store
from app.tools import default_tools

_compiled_graph: CompiledStateGraph | None = None


def build_graph(
    llm: BaseChatModel | None = None,
    *,
    tools: Sequence[Any] | None = None,
    store: InMemorySessionStore | None = None,
) -> CompiledStateGraph:
    """Build a compiled graph. Pass ``llm`` / ``store`` from tests to isolate runs.

    ``tools`` defaults to Megafon and the weather MCP wrapper. The mock LLM
    ignores bound tools; a real provider or a scripted test LLM is what
    actually emits ``tool_calls``. The weather wrapper is local: MCP is
    configured at process start and connects only on the first weather call.

    The translate node is a separate graph step (``language`` on GraphState)
    so the agent never translates. Translation runs only when ``language`` is
    a non-empty code and the agent produced text; otherwise the reply language
    is left to the agent (typically matching the user message).
    """

    chat_model = create_chat_model() if llm is None else llm
    session_store = store if store is not None else get_session_store()
    bound_tools = list(default_tools() if tools is None else tools)
    graph = StateGraph(GraphState)
    graph.add_node(LOAD_HISTORY_NODE, make_load_history_node(session_store))
    graph.add_node(AGENT_NODE, make_agent_node(chat_model, tools=bound_tools))
    graph.add_node(TRANSLATE_NODE, make_translate_node(chat_model))
    graph.add_node(SAVE_HISTORY_NODE, make_save_history_node(session_store))
    graph.add_edge(START, LOAD_HISTORY_NODE)
    graph.add_edge(LOAD_HISTORY_NODE, AGENT_NODE)
    if bound_tools:
        graph.add_node(TOOLS_NODE, ToolNode(bound_tools))
        graph.add_conditional_edges(
            AGENT_NODE,
            route_after_agent,
            {TOOLS_NODE: TOOLS_NODE, TRANSLATE_NODE: TRANSLATE_NODE},
        )
        graph.add_edge(TOOLS_NODE, AGENT_NODE)
    else:
        graph.add_edge(AGENT_NODE, TRANSLATE_NODE)
    graph.add_edge(TRANSLATE_NODE, SAVE_HISTORY_NODE)
    graph.add_edge(SAVE_HISTORY_NODE, END)
    return graph.compile()


def get_graph(llm: BaseChatModel | None = None) -> CompiledStateGraph:
    """Return the process-wide compiled graph, creating it on first use.

    When ``llm`` is passed (startup / tests), rebuild so the attached model is
    the one the caller configured. Bare calls reuse the cached graph.
    """

    global _compiled_graph
    if _compiled_graph is None or llm is not None:
        _compiled_graph = build_graph(llm)
    return _compiled_graph
