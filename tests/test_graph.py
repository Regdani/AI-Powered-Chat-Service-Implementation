"""Graph core: GraphState, START → load_history → agent ⇄ tools → translate → save → END."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, get_type_hints

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable

from app.graph.builder import build_graph
from app.graph.nodes import (
    AGENT_NODE,
    AGENT_SYSTEM_PROMPT,
    LOAD_HISTORY_NODE,
    SAVE_HISTORY_NODE,
    TOOLS_NODE,
    TRANSLATE_NODE,
    TRANSLATE_SYSTEM_PROMPT,
    make_translate_node,
    needs_translation,
    normalize_language,
    route_after_agent,
)
from app.graph.state import GraphState
from app.llm.mock_chat import (
    FIXED_JOKE,
    MockChatModel,
    echo_with_joke,
    mock_language_marker,
)
from app.memory.store import InMemorySessionStore
from app.tools.mcp_weather import (
    UNAVAILABLE_PREFIX,
    WEATHER_TOOL_NAME,
    configure_weather_client,
    reset_weather_client,
)
from app.tools.megafon import MEGAFON_TOOL_NAME


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


class ScriptedChatModel(BaseChatModel):
    """Replay canned ``AIMessage``s so tests can drive the tool loop.

    Extra attributes are underscore-prefixed because ``BaseChatModel`` is a
    Pydantic model and rejects undeclared public fields.
    """

    def __init__(self, responses: list[AIMessage]) -> None:
        super().__init__()
        self._responses = list(responses)
        self._index = 0
        self._async_calls = 0
        self._seen: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "scripted-test"

    def bind_tools(
        self,
        tools: Any,
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable:
        del tools, tool_choice, kwargs
        return self

    def _next(self) -> AIMessage:
        if self._index >= len(self._responses):
            raise AssertionError("ScriptedChatModel has no remaining responses")
        message = self._responses[self._index]
        self._index += 1
        return message

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        return ChatResult(generations=[ChatGeneration(message=self._next())])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._async_calls += 1
        self._seen.append(list(messages))
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def _megafon_scripted_llm() -> ScriptedChatModel:
    """First turn requests Megafon; second turn is the final echo+joke reply."""
    return ScriptedChatModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": MEGAFON_TOOL_NAME,
                        "args": {"text": "Announce: hello world"},
                        "id": "call_megafon_1",
                    }
                ],
            ),
            AIMessage(content="You said: HELLO WORLD. " + FIXED_JOKE),
        ]
    )


def _weather_scripted_llm(final_answer: str) -> ScriptedChatModel:
    """First turn requests weather; second turn is the final reply."""
    return ScriptedChatModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": WEATHER_TOOL_NAME,
                        "args": {"location": "Paris"},
                        "id": "call_weather_1",
                    }
                ],
            ),
            AIMessage(content=final_answer),
        ]
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
    assert TOOLS_NODE in graph.nodes
    assert TRANSLATE_NODE in graph.nodes
    assert SAVE_HISTORY_NODE in graph.nodes
    edges = {(edge[0], edge[1]) for edge in graph.get_graph().edges}
    assert ("__start__", LOAD_HISTORY_NODE) in edges
    assert (LOAD_HISTORY_NODE, AGENT_NODE) in edges
    assert (AGENT_NODE, TOOLS_NODE) in edges
    assert (TOOLS_NODE, AGENT_NODE) in edges
    assert (AGENT_NODE, TRANSLATE_NODE) in edges
    assert (TRANSLATE_NODE, SAVE_HISTORY_NODE) in edges
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
    result = asyncio.run(graph.ainvoke(_turn("Szia", language="hu")))
    assert llm._async_calls == 1
    system = llm._seen[0][0]
    assert isinstance(system, SystemMessage)
    assert system.content == AGENT_SYSTEM_PROMPT
    assert "hu" not in system.content.lower()
    assert "translate" in system.content.lower()
    human = [m for m in llm._seen[0] if isinstance(m, HumanMessage)]
    assert human[-1].content == "Szia"
    assert result["final_answer"] == mock_language_marker(
        echo_with_joke("Szia"), "hu"
    )


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


def test_mock_llm_does_not_call_megafon_for_announce() -> None:
    """Default mock has no Announce: heuristic; echo+joke is not a bug."""
    graph = _build()
    result = asyncio.run(graph.ainvoke(_turn("Announce: hello world")))
    assert not any(getattr(m, "type", None) == "tool" for m in result["messages"])
    assert result["final_answer"] == echo_with_joke("Announce: hello world")


def test_route_after_agent_uses_tool_calls() -> None:
    with_tools: GraphState = {
        "messages": [
            HumanMessage(content="Announce: hi"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": MEGAFON_TOOL_NAME,
                        "args": {"text": "Announce: hi"},
                        "id": "c1",
                    }
                ],
            ),
        ],
        "user_message": "Announce: hi",
        "language": "en",
        "session_id": "s",
        "final_answer": "",
    }
    without_tools: GraphState = {
        "messages": [
            HumanMessage(content="Hello"),
            AIMessage(content=echo_with_joke("Hello")),
        ],
        "user_message": "Hello",
        "language": "en",
        "session_id": "s",
        "final_answer": echo_with_joke("Hello"),
    }
    assert route_after_agent(with_tools) == TOOLS_NODE
    assert route_after_agent(without_tools) == TRANSLATE_NODE


def test_graph_megafon_tool_path_with_scripted_llm() -> None:
    llm = _megafon_scripted_llm()
    store = InMemorySessionStore()
    graph = _build(llm, store)
    result = asyncio.run(graph.ainvoke(_turn("Announce: hello world")))

    tool_messages = [m for m in result["messages"] if getattr(m, "type", None) == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0].content == "HELLO WORLD"
    assert result["final_answer"] == "You said: HELLO WORLD. " + FIXED_JOKE
    assert llm._async_calls == 2
    assert llm._index == 2

    second = llm._seen[1]
    assert any(getattr(m, "type", None) == "tool" for m in second)
    humans = [m.content for m in second if isinstance(m, HumanMessage)]
    assert humans[-1] == "Announce: hello world"

    saved = asyncio.run(store.get_messages("session-1"))
    saved_types = [getattr(m, "type", None) for m in saved]
    assert "tool" in saved_types
    assert saved_types[-1] == "ai"


def test_graph_tool_path_then_one_translator_call() -> None:
    """START → agent → tools → agent → translate → END; agent still does not translate."""
    shouted = "You said: HELLO WORLD. " + FIXED_JOKE
    translated = "Te mondtad: HELLO WORLD."
    llm = ScriptedChatModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": MEGAFON_TOOL_NAME,
                        "args": {"text": "Announce: hello world"},
                        "id": "call_megafon_1",
                    }
                ],
            ),
            AIMessage(content=shouted),
            AIMessage(content=translated),
        ]
    )
    graph = _build(llm)
    result = asyncio.run(
        graph.ainvoke(_turn("Announce: hello world", language="hu"))
    )
    assert llm._async_calls == 3
    assert result["final_answer"] == translated
    for agent_call in llm._seen[:2]:
        assert agent_call[0].content == AGENT_SYSTEM_PROMPT
        assert "hu" not in agent_call[0].content.lower()
    assert llm._seen[2][0].content == TRANSLATE_SYSTEM_PROMPT.format(language="hu")
    assert llm._seen[2][1].content == shouted


def test_scripted_tool_followup_does_not_duplicate_human_message() -> None:
    llm = _megafon_scripted_llm()
    graph = _build(llm)
    asyncio.run(graph.ainvoke(_turn("Announce: hello world")))
    second = llm._seen[1]
    human_contents = [m.content for m in second if isinstance(m, HumanMessage)]
    assert human_contents.count("Announce: hello world") == 1


def test_graph_megafon_tool_path_logs_example(caplog: Any) -> None:
    llm = _megafon_scripted_llm()
    graph = _build(llm)
    with caplog.at_level(logging.INFO):
        result = asyncio.run(
            graph.ainvoke(_turn("Announce: hello world", session_id="megafon-demo"))
        )
    assert "Megafon announce" in caplog.text
    assert "HELLO WORLD" in caplog.text
    assert "Agent tool calls" in caplog.text
    assert MEGAFON_TOOL_NAME in caplog.text
    assert "megafon-demo" in caplog.text
    assert "Agent reply" in caplog.text
    assert result["final_answer"] in caplog.text


def test_mock_llm_does_not_call_weather() -> None:
    """Default mock has no weather heuristic; echo+joke is not a bug."""
    graph = _build()
    result = asyncio.run(graph.ainvoke(_turn("What is the weather in Paris?")))
    assert not any(getattr(m, "type", None) == "tool" for m in result["messages"])
    assert result["final_answer"] == echo_with_joke("What is the weather in Paris?")


def test_graph_weather_tool_path_with_scripted_llm() -> None:
    async def call_remote(name: str, arguments: dict[str, Any]) -> Any:
        if name == "geocode_location":
            return {
                "results": [
                    {"latitude": 48.8566, "longitude": 2.3522, "name": "Paris"}
                ]
            }
        if name == "get_weather_forecast":
            return {"current_weather": {"temperature": 18.5, "windspeed": 10}}
        raise AssertionError(name)

    reset_weather_client()
    configure_weather_client("https://weather.example/mcp", call_remote=call_remote)
    try:
        final = "You said: Weather for Paris: temperature 18.5°C. " + FIXED_JOKE
        llm = _weather_scripted_llm(final)
        graph = _build(llm)
        result = asyncio.run(graph.ainvoke(_turn("What is the weather in Paris?")))
        tool_messages = [
            m for m in result["messages"] if getattr(m, "type", None) == "tool"
        ]
        assert len(tool_messages) == 1
        assert "Weather for Paris" in tool_messages[0].content
        assert "18.5" in tool_messages[0].content
        assert result["final_answer"] == final
        assert llm._async_calls == 2
    finally:
        reset_weather_client()


def test_graph_weather_network_error_does_not_crash(caplog: Any) -> None:
    async def boom(name: str, arguments: dict[str, Any]) -> Any:
        del name, arguments
        raise ConnectionError("weather.chukai.io is unreachable")

    reset_weather_client()
    configure_weather_client("https://weather.example/mcp", call_remote=boom)
    try:
        final = "You said: I could not reach the weather service. " + FIXED_JOKE
        llm = _weather_scripted_llm(final)
        graph = _build(llm)
        with caplog.at_level(logging.INFO):
            result = asyncio.run(
                graph.ainvoke(
                    _turn("What is the weather in Paris?", session_id="weather-demo")
                )
            )
        tool_messages = [
            m for m in result["messages"] if getattr(m, "type", None) == "tool"
        ]
        assert len(tool_messages) == 1
        assert UNAVAILABLE_PREFIX in tool_messages[0].content
        assert "ConnectionError" in tool_messages[0].content
        assert result["final_answer"] == final
        assert "Agent tool calls" in caplog.text
        assert WEATHER_TOOL_NAME in caplog.text
        assert "weather-demo" in caplog.text
    finally:
        reset_weather_client()


def _translate_state(
    *,
    language: str,
    final_answer: str,
    session_id: str = "s",
) -> GraphState:
    return {
        "messages": [AIMessage(content=final_answer)] if final_answer else [],
        "user_message": "Hello",
        "language": language,
        "session_id": session_id,
        "final_answer": final_answer,
    }


def test_normalize_language_defaults_blank_to_english() -> None:
    assert normalize_language(None) == "en"
    assert normalize_language("  ") == "en"
    assert normalize_language("HU") == "hu"


def test_needs_translation_skips_english_and_empty_text() -> None:
    assert needs_translation("en", "Hello") is False
    assert needs_translation("EN", "Hello") is False
    assert needs_translation("hu", "") is False
    assert needs_translation("hu", "Hello") is True


def test_translate_node_skips_llm_for_english() -> None:
    llm = ScriptedChatModel([])
    node = make_translate_node(llm)
    result = asyncio.run(node(_translate_state(language="en", final_answer="Hello")))
    assert result["final_answer"] == "Hello"
    assert llm._async_calls == 0


def test_translate_node_skips_llm_when_there_is_nothing_to_translate() -> None:
    llm = ScriptedChatModel([])
    node = make_translate_node(llm)
    result = asyncio.run(node(_translate_state(language="hu", final_answer="")))
    assert result["final_answer"] == ""
    assert llm._async_calls == 0


def test_translate_node_mock_prefixes_without_llm_call() -> None:
    llm = CountingMock()
    node = make_translate_node(llm)
    result = asyncio.run(
        node(_translate_state(language="hu", final_answer="You said: Hello."))
    )
    assert result["final_answer"] == "[hu] You said: Hello."
    assert llm._async_calls == 0


def test_translate_node_makes_one_llm_call_for_other_language() -> None:
    llm = ScriptedChatModel([AIMessage(content="Szia ott.")])
    node = make_translate_node(llm)
    result = asyncio.run(
        node(_translate_state(language="hu", final_answer=echo_with_joke("Hello")))
    )
    assert result["final_answer"] == "Szia ott."
    assert llm._async_calls == 1
    seen = llm._seen[0]
    assert isinstance(seen[0], SystemMessage)
    assert seen[0].content == TRANSLATE_SYSTEM_PROMPT.format(language="hu")
    assert isinstance(seen[1], HumanMessage)
    assert seen[1].content == echo_with_joke("Hello")


def test_graph_english_does_not_call_translator() -> None:
    llm = ScriptedChatModel([AIMessage(content=echo_with_joke("Hello"))])
    graph = _build(llm)
    result = asyncio.run(graph.ainvoke(_turn("Hello", language="en")))
    assert llm._async_calls == 1
    assert result["final_answer"] == echo_with_joke("Hello")
    assert not result["final_answer"].startswith("[en]")


def test_graph_non_english_makes_one_translator_call() -> None:
    translated = "Szia ott."
    llm = ScriptedChatModel(
        [
            AIMessage(content=echo_with_joke("Hello")),
            AIMessage(content=translated),
        ]
    )
    graph = _build(llm)
    result = asyncio.run(graph.ainvoke(_turn("Hello", language="hu")))
    assert llm._async_calls == 2
    assert result["final_answer"] == translated
    agent_system = llm._seen[0][0]
    assert isinstance(agent_system, SystemMessage)
    assert agent_system.content == AGENT_SYSTEM_PROMPT
    assert "hu" not in agent_system.content.lower()
    translator = llm._seen[1]
    assert translator[0].content == TRANSLATE_SYSTEM_PROMPT.format(language="hu")
    assert translator[1].content == echo_with_joke("Hello")
    assert result["messages"][-1].content == echo_with_joke("Hello")


def test_graph_mock_language_prefix_is_not_real_translation() -> None:
    graph = _build()
    result = asyncio.run(graph.ainvoke(_turn("Hello", language="hu")))
    expected = mock_language_marker(echo_with_joke("Hello"), "hu")
    assert result["final_answer"] == expected
    assert result["messages"][-1].content == echo_with_joke("Hello")


def test_graph_mock_english_has_no_language_prefix() -> None:
    graph = _build()
    result = asyncio.run(graph.ainvoke(_turn("Hello", language="EN")))
    assert result["final_answer"] == echo_with_joke("Hello")


def test_translate_keeps_history_in_agent_language() -> None:
    store = InMemorySessionStore()
    graph = _build(store=store)
    asyncio.run(graph.ainvoke(_turn("Hello", language="hu", session_id="s1")))
    saved = asyncio.run(store.get_messages("s1"))
    assert saved[-1].content == echo_with_joke("Hello")
    assert not str(saved[-1].content).startswith("[hu]")
