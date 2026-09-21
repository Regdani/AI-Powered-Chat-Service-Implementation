"""Weather MCP wrapper: config at startup, lazy connect, errors stay in-tool."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.graph.builder import build_graph
from app.llm.mock_chat import FIXED_JOKE, MockChatModel, echo_with_joke
from app.main import create_app
from app.memory.store import InMemorySessionStore
from app.tools import default_tools
from app.tools.mcp_weather import (
    DEFAULT_TRANSPORT,
    DEFAULT_WEATHER_MCP_URL,
    FORECAST_REMOTE_TOOL,
    GEOCODE_REMOTE_TOOL,
    UNAVAILABLE_PREFIX,
    WEATHER_TOOL_NAME,
    WeatherMcpClient,
    configure_weather_client,
    get_weather,
    get_weather_client,
    reset_weather_client,
)
from app.tools.megafon import MEGAFON_TOOL_NAME

PARIS_GEO = {
    "results": [{"latitude": 48.8566, "longitude": 2.3522, "name": "Paris"}]
}
PARIS_FORECAST = {
    "current_weather": {"temperature": 18.5, "windspeed": 10, "weathercode": 0}
}


@pytest.fixture(autouse=True)
def _reset_weather_client() -> Any:
    reset_weather_client()
    yield
    reset_weather_client()


def _scripted_remote(
    geo: Any = PARIS_GEO,
    forecast: Any = PARIS_FORECAST,
) -> tuple[AsyncMock, list[tuple[str, dict[str, Any]]]]:
    calls: list[tuple[str, dict[str, Any]]] = []

    async def call_remote(name: str, arguments: dict[str, Any]) -> Any:
        calls.append((name, arguments))
        if name == GEOCODE_REMOTE_TOOL:
            return geo
        if name == FORECAST_REMOTE_TOOL:
            return forecast
        raise AssertionError(f"unexpected remote tool {name}")

    return AsyncMock(side_effect=call_remote), calls


def test_configure_stores_url_and_transport_without_connecting() -> None:
    client = configure_weather_client(
        url="https://weather.example/mcp",
        transport="streamable_http",
        call_remote=AsyncMock(side_effect=AssertionError("no remote at configure")),
    )
    assert client.connection_config == {
        "url": "https://weather.example/mcp",
        "transport": "streamable_http",
    }
    assert client.connect_attempts == 0
    assert get_weather_client() is client


def test_configure_builds_adapter_without_session_or_get_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    async def boom_get_tools(self: Any, *args: Any, **kwargs: Any) -> list[Any]:
        events.append("get_tools")
        raise AssertionError("startup must not list remote MCP tools")

    @asynccontextmanager
    async def boom_session(self: Any, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        events.append("session")
        raise AssertionError("startup must not open an MCP session")
        yield None  # pragma: no cover

    monkeypatch.setattr(MultiServerMCPClient, "get_tools", boom_get_tools)
    monkeypatch.setattr(MultiServerMCPClient, "session", boom_session)

    client = configure_weather_client(DEFAULT_WEATHER_MCP_URL, DEFAULT_TRANSPORT)
    assert client.connect_attempts == 0
    assert client._adapter_client is not None
    assert events == []


def test_lookup_connects_lazily_and_proxies_geocode_then_forecast() -> None:
    remote, calls = _scripted_remote()
    client = configure_weather_client(
        "https://weather.example/mcp", call_remote=remote
    )
    assert client.connect_attempts == 0

    result = asyncio.run(client.lookup("Paris"))

    assert client.connect_attempts == 1
    assert [name for name, _args in calls] == [
        GEOCODE_REMOTE_TOOL,
        FORECAST_REMOTE_TOOL,
    ]
    assert calls[0][1]["name"] == "Paris"
    assert calls[1][1]["latitude"] == 48.8566
    assert calls[1][1]["longitude"] == 2.3522
    assert "Weather for Paris" in result
    assert "18.5" in result


def test_tool_ainvoke_uses_configured_client() -> None:
    remote, _calls = _scripted_remote()
    configure_weather_client("https://weather.example/mcp", call_remote=remote)
    result = asyncio.run(get_weather.ainvoke({"location": "Paris"}))
    assert get_weather.name == WEATHER_TOOL_NAME
    assert "Weather for Paris" in result
    assert "18.5" in result


def test_network_error_is_returned_not_raised() -> None:
    async def boom(name: str, arguments: dict[str, Any]) -> Any:
        del name, arguments
        raise ConnectionError("Name or service not known")

    configure_weather_client("https://weather.example/mcp", call_remote=boom)
    result = asyncio.run(get_weather.ainvoke({"location": "Paris"}))
    assert result.startswith(UNAVAILABLE_PREFIX)
    assert "ConnectionError" in result
    assert get_weather_client().connect_attempts == 1


def test_timeout_error_is_returned_not_raised() -> None:
    async def boom(name: str, arguments: dict[str, Any]) -> Any:
        del name, arguments
        raise TimeoutError("weather.chukai.io timed out")

    client = configure_weather_client(
        "https://weather.example/mcp", call_remote=boom
    )
    result = asyncio.run(client.lookup("Budapest"))
    assert UNAVAILABLE_PREFIX in result
    assert "TimeoutError" in result


def test_unknown_location_does_not_call_forecast() -> None:
    remote, calls = _scripted_remote(geo={"results": []})
    client = configure_weather_client(
        "https://weather.example/mcp", call_remote=remote
    )
    result = asyncio.run(client.lookup("Narnia"))
    assert "no coordinates" in result
    assert [name for name, _args in calls] == [GEOCODE_REMOTE_TOOL]


def test_blank_location_does_not_connect() -> None:
    remote, calls = _scripted_remote()
    client = configure_weather_client(
        "https://weather.example/mcp", call_remote=remote
    )
    result = asyncio.run(client.lookup("   "))
    assert UNAVAILABLE_PREFIX in result
    assert client.connect_attempts == 0
    assert calls == []


def test_lookup_logs_lazy_connect_and_errors(caplog: Any) -> None:
    async def boom(name: str, arguments: dict[str, Any]) -> Any:
        del name, arguments
        raise ConnectionError("offline")

    client = configure_weather_client(
        "https://weather.example/mcp", call_remote=boom
    )
    with caplog.at_level(logging.INFO, logger="app.tools.mcp_weather"):
        asyncio.run(client.lookup("Paris"))
    assert "Weather MCP lookup" in caplog.text
    assert "Weather MCP lookup failed" in caplog.text


def test_lookup_logs_mcp_tool_result(caplog: Any) -> None:
    remote, _calls = _scripted_remote()
    client = configure_weather_client(
        "https://weather.example/mcp", call_remote=remote
    )
    with caplog.at_level(logging.INFO, logger="app.tools.mcp_weather"):
        result = asyncio.run(client.lookup("Paris"))
    assert "Weather MCP tool result source=mcp" in caplog.text
    assert "18.5" in caplog.text
    assert result in caplog.text


def test_real_session_path_opens_session_only_on_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = MagicMock()

    async def call_tool(name: str, arguments: dict[str, Any] | None = None) -> Any:
        if name == GEOCODE_REMOTE_TOOL:
            return PARIS_GEO
        if name == FORECAST_REMOTE_TOOL:
            return PARIS_FORECAST
        raise AssertionError(name)

    session.call_tool = AsyncMock(side_effect=call_tool)
    opened: list[str] = []

    @asynccontextmanager
    async def fake_session(
        self: Any, server_name: str, **kwargs: Any
    ) -> AsyncIterator[Any]:
        del kwargs
        opened.append(server_name)
        yield session

    monkeypatch.setattr(MultiServerMCPClient, "session", fake_session)

    client = WeatherMcpClient(DEFAULT_WEATHER_MCP_URL)
    client.attach_adapter()
    assert opened == []
    assert client.connect_attempts == 0

    result = asyncio.run(client.lookup("Paris"))

    assert opened == ["weather"]
    assert client.connect_attempts == 1
    assert "Weather for Paris" in result
    assert session.call_tool.await_count == 2


def test_default_tools_includes_megafon_and_weather() -> None:
    names = [item.name for item in default_tools()]
    assert names == [MEGAFON_TOOL_NAME, WEATHER_TOOL_NAME]


def test_startup_does_not_handshake_and_health_stays_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    async def boom_get_tools(self: Any, *args: Any, **kwargs: Any) -> list[Any]:
        events.append("get_tools")
        raise AssertionError("startup must not list remote MCP tools")

    @asynccontextmanager
    async def boom_session(self: Any, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        events.append("session")
        raise AssertionError("startup must not open an MCP session")
        yield None  # pragma: no cover

    monkeypatch.setattr(MultiServerMCPClient, "get_tools", boom_get_tools)
    monkeypatch.setattr(MultiServerMCPClient, "session", boom_session)

    application = create_app()
    with TestClient(application) as client:
        health = client.get("/health")
        chat = client.get(
            "/chat", params={"message": "What is the weather in Paris?"}
        )
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert chat.status_code == 200
    assert chat.json()["answer"] == echo_with_joke(
        "What is the weather in Paris?"
    )
    assert events == []


def test_chat_still_works_when_weather_client_would_fail() -> None:
    async def boom(name: str, arguments: dict[str, Any]) -> Any:
        del name, arguments
        raise ConnectionError("weather host down")

    application = create_app()
    with TestClient(application) as client:
        # Lifespan already configured MCP; replace with a failing remote.
        configure_weather_client("https://weather.example/mcp", call_remote=boom)
        client.app.state.graph = build_graph(
            llm=MockChatModel(), store=InMemorySessionStore()
        )
        health = client.get("/health")
        chat = client.get("/chat", params={"message": "Hello"})
    assert health.status_code == 200
    assert chat.status_code == 200
    assert FIXED_JOKE in chat.json()["answer"]
    # Mock LLM never called the weather tool, so the boom remote was unused.
    assert get_weather_client().connect_attempts == 0
