"""Graph core: GraphState, START → load_history → agent → save_history → END."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, get_type_hints

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult

from app.graph.builder import build_graph
from app.graph.nodes import (
    AGENT_NODE,
    AGENT_SYSTEM_PROMPT,
    LOAD_HISTORY_NODE,
    SAVE_HISTORY_NODE,
)
from app.graph.state import GraphState
from app.llm.mock_chat import FIXED_JOKE, MockChatModel, echo_with_joke
from app.memory.store import InMemorySessionStore


def _turn(
    message: str,
    *,
    language: str = "en",
    session_id: str = "session-1",
) -> dict[str, Any]:
    return {
        "user_message": message,
        "language": language,
        "session_id": session_id,
        "messages": [],
        "final_answer": "",
    }


def _build(llm: Any = None, store: InMemorySessionStore | None = None):
    """Compile a graph with a fresh in-memory store unless one is injected."""
    return build_graph(
        MockChatModel() if llm is None else llm,
        store=InMemorySessionStore() if store is None else store,
    )


class CountingMock(MockChatModel):
    """Records async LLM calls so tests can prove ``ainvoke`` is used.

    Extra attributes are underscore-prefixed because ``BaseChatModel`` is a
    Pydantic model and rejects undeclared public fields.
    """

    def __init__(self) -> None:
        super().__init__()
        self._async_calls = 0
        self._seen: list[list[BaseMessage]] = []

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._async_calls += 1
        self._seen.append(list(messages))
        return await super()._agenerate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )


def test_graph_state_has_required_fields() -> None:
    hints = get_type_hints(GraphState, include_extras=True)
    assert set(hints) == {
        "messages",
        "user_message",
        "language",
        "session_id",
        "final_answer",
    }


def test_compiled_graph_loads_and_saves_history() -> None:
    graph = _build()
    assert LOAD_HISTORY_NODE in graph.nodes
    assert AGENT_NODE in graph.nodes
    assert SAVE_HISTORY_NODE in graph.nodes
    edges = {(edge[0], edge[1]) for edge in graph.get_graph().edges}
    assert ("__start__", LOAD_HISTORY_NODE) in edges
    assert (LOAD_HISTORY_NODE, AGENT_NODE) in edges
    assert (AGENT_NODE, SAVE_HISTORY_NODE) in edges
    assert (SAVE_HISTORY_NODE, "__end__") in edges


def test_agent_echoes_user_text_and_fixed_joke() -> None:
    graph = _build()
    result = asyncio.run(graph.ainvoke(_turn("Hello there")))
    assert result["final_answer"] == echo_with_joke("Hello there")
    assert "Hello there" in result["final_answer"]
    assert FIXED_JOKE in result["final_answer"]
    assert result["session_id"] == "session-1"


def test_agent_uses_async_ainvoke() -> None:
    llm = CountingMock()
    graph = _build(llm)
    asyncio.run(graph.ainvoke(_turn("Ping")))
    assert llm._async_calls >= 1


def test_agent_system_prompt_does_not_request_target_language() -> None:
    llm = CountingMock()
    graph = _build(llm)
    asyncio.run(graph.ainvoke(_turn("Szia", language="hu")))
    assert llm._seen
    system = llm._seen[0][0]
    assert isinstance(system, SystemMessage)
    assert system.content == AGENT_SYSTEM_PROMPT
    assert "hu" not in system.content.lower()
    assert "translate" in system.content.lower()
    human = [m for m in llm._seen[0] if isinstance(m, HumanMessage)]
    assert human[-1].content == "Szia"


def test_agent_appends_human_and_ai_messages() -> None:
    graph = _build()
    result = asyncio.run(graph.ainvoke(_turn("Hello")))
    messages = result["messages"]
    assert len(messages) >= 2
    assert isinstance(messages[-2], HumanMessage)
    assert messages[-2].content == "Hello"
    assert isinstance(messages[-1], AIMessage)
    assert messages[-1].content == echo_with_joke("Hello")


def test_agent_logs_the_reply(caplog: Any) -> None:
    graph = _build()
    with caplog.at_level(logging.INFO, logger="app.graph.nodes"):
        result = asyncio.run(graph.ainvoke(_turn("Hello", session_id="log-session")))
    assert "log-session" in caplog.text
    assert result["final_answer"] in caplog.text
    assert "Agent reply" in caplog.text


def test_save_history_persists_messages_in_store() -> None:
    store = InMemorySessionStore()
    graph = _build(store=store)
    asyncio.run(graph.ainvoke(_turn("Hello", session_id="s1")))
    saved = asyncio.run(store.get_messages("s1"))
    assert len(saved) >= 2
    assert isinstance(saved[0], HumanMessage)
    assert saved[0].content == "Hello"
    assert isinstance(saved[1], AIMessage)
    assert saved[1].content == echo_with_joke("Hello")


def test_second_turn_loads_prior_messages_into_llm() -> None:
    llm = CountingMock()
    store = InMemorySessionStore()
    graph = _build(llm, store)
    asyncio.run(graph.ainvoke(_turn("Hello", session_id="s1")))
    asyncio.run(graph.ainvoke(_turn("Follow up", session_id="s1")))
    assert len(llm._seen) == 2
    second = llm._seen[1]
    human_texts = [m.content for m in second if isinstance(m, HumanMessage)]
    ai_texts = [m.content for m in second if isinstance(m, AIMessage)]
    assert human_texts[0] == "Hello"
    assert human_texts[-1] == "Follow up"
    assert any(FIXED_JOKE in text for text in ai_texts)


def test_sessions_do_not_share_history() -> None:
    llm = CountingMock()
    store = InMemorySessionStore()
    graph = _build(llm, store)
    asyncio.run(graph.ainvoke(_turn("Alpha", session_id="a")))
    asyncio.run(graph.ainvoke(_turn("Beta", session_id="b")))
    second = llm._seen[1]
    human_texts = [m.content for m in second if isinstance(m, HumanMessage)]
    assert "Alpha" not in human_texts
    assert human_texts[-1] == "Beta"
