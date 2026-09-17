"""Graph core: GraphState, START → agent → END, echo+joke, async ainvoke, logging."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, get_type_hints

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult

from app.graph.builder import build_graph
from app.graph.nodes import AGENT_NODE, AGENT_SYSTEM_PROMPT
from app.graph.state import GraphState
from app.llm.mock_chat import FIXED_JOKE, MockChatModel, echo_with_joke


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


def test_compiled_graph_is_start_agent_end() -> None:
    graph = build_graph(MockChatModel())
    assert AGENT_NODE in graph.nodes
    edges = {(edge[0], edge[1]) for edge in graph.get_graph().edges}
    assert ("__start__", AGENT_NODE) in edges
    assert (AGENT_NODE, "__end__") in edges


def test_agent_echoes_user_text_and_fixed_joke() -> None:
    graph = build_graph(MockChatModel())
    result = asyncio.run(graph.ainvoke(_turn("Hello there")))
    assert result["final_answer"] == echo_with_joke("Hello there")
    assert "Hello there" in result["final_answer"]
    assert FIXED_JOKE in result["final_answer"]
    assert result["session_id"] == "session-1"


def test_agent_uses_async_ainvoke() -> None:
    llm = CountingMock()
    graph = build_graph(llm)
    asyncio.run(graph.ainvoke(_turn("Ping")))
    assert llm._async_calls >= 1


def test_agent_system_prompt_does_not_request_target_language() -> None:
    llm = CountingMock()
    graph = build_graph(llm)
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
    graph = build_graph(MockChatModel())
    result = asyncio.run(graph.ainvoke(_turn("Hello")))
    messages = result["messages"]
    assert len(messages) >= 2
    assert isinstance(messages[-2], HumanMessage)
    assert messages[-2].content == "Hello"
    assert isinstance(messages[-1], AIMessage)
    assert messages[-1].content == echo_with_joke("Hello")


def test_agent_logs_the_reply(caplog: Any) -> None:
    graph = build_graph(MockChatModel())
    with caplog.at_level(logging.INFO, logger="app.graph.nodes"):
        result = asyncio.run(graph.ainvoke(_turn("Hello", session_id="log-session")))
    assert "log-session" in caplog.text
    assert result["final_answer"] in caplog.text
    assert "Agent reply" in caplog.text
