"""Local weather wrapper that proxies to the public Open-Meteo MCP server.

Startup only stores client configuration (URL + transport). There is no remote
handshake and no ``get_tools()`` on the network until the first weather tool
call. If that call cannot reach the server, the tool returns an error string so
the agent can reply; the API process stays up.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

WEATHER_TOOL_NAME = "get_weather"
WEATHER_MCP_SERVER_NAME = "weather"
DEFAULT_WEATHER_MCP_URL = "https://weather.chukai.io/mcp"
DEFAULT_TRANSPORT = "streamable_http"
GEOCODE_REMOTE_TOOL = "geocode_location"
FORECAST_REMOTE_TOOL = "get_weather_forecast"
UNAVAILABLE_PREFIX = "Weather MCP is unavailable"

RemoteCall = Callable[[str, dict[str, Any]], Awaitable[Any]]

_client: WeatherMcpClient | None = None


class WeatherMcpError(Exception):
    """Remote weather MCP handshake or tool call failed."""


class WeatherMcpClient:
    """MCP weather client: configure now, connect on the first lookup."""

    def __init__(
        self,
        url: str,
        transport: str = DEFAULT_TRANSPORT,
        *,
        call_remote: RemoteCall | None = None,
    ) -> None:
        self.url = url
        self.transport = transport
        self._call_remote = call_remote
        self._adapter_client: Any = None
        self._connect_attempts = 0

    @property
    def connection_config(self) -> dict[str, str]:
        """URL and transport only — constructing this is not a handshake."""
        return {"url": self.url, "transport": self.transport}

    @property
    def connect_attempts(self) -> int:
        """How many times a lookup has tried to talk to the remote MCP."""
        return self._connect_attempts

    def attach_adapter(self) -> None:
        """Build the LangChain MCP adapter from stored config.

        ``MultiServerMCPClient`` construction stores the connection dict. It
        does **not** open a session or list remote tools.
        """
        if self._call_remote is not None:
            return
        from langchain_mcp_adapters.client import MultiServerMCPClient

        self._adapter_client = MultiServerMCPClient(
            {
                WEATHER_MCP_SERVER_NAME: {
                    "transport": self.transport,
                    "url": self.url,
                }
            }
        )
        logger.info(
            "Weather MCP client configured url=%s transport=%s "
            "(handshake deferred until first tool call)",
            self.url,
            self.transport,
        )

    @asynccontextmanager
    async def _session(self):
        """Open an MCP session (initialize handshake). Call-time only."""
        if self._adapter_client is None:
            self.attach_adapter()
        logger.info("Opening weather MCP session url=%s (lazy connect)", self.url)
        async with self._adapter_client.session(WEATHER_MCP_SERVER_NAME) as session:
            yield session

    async def lookup(self, location: str) -> str:
        """Geocode ``location`` then fetch current weather. Never raises."""
        place = (location or "").strip()
        if not place:
            return f"{UNAVAILABLE_PREFIX}: location is required."

        logger.info("Weather MCP lookup location=%r", place)
        try:
            self._connect_attempts += 1
            geo_payload, forecast_payload = await self._geocode_and_forecast(place)
            coords = _coordinates_from_geocode(geo_payload)
            if coords is None:
                return f"Weather MCP found no coordinates for {place!r}."
            _latitude, _longitude, name = coords
            return _format_forecast(forecast_payload, location=name or place)
        except Exception as exc:
            logger.warning(
                "Weather MCP lookup failed location=%r error=%s",
                place,
                exc,
            )
            return f"{UNAVAILABLE_PREFIX}: {type(exc).__name__}: {exc}"

    async def _geocode_and_forecast(self, place: str) -> tuple[Any, Any]:
        """Handshake (if needed), geocode, then forecast. Raises on transport errors."""
        if self._call_remote is not None:
            geo_payload = _as_payload(
                await self._call_remote(
                    GEOCODE_REMOTE_TOOL, {"name": place, "count": 1}
                )
            )
            coords = _coordinates_from_geocode(geo_payload)
            if coords is None:
                return geo_payload, None
            latitude, longitude, _name = coords
            forecast_payload = _as_payload(
                await self._call_remote(
                    FORECAST_REMOTE_TOOL,
                    _forecast_args(latitude, longitude),
                )
            )
            return geo_payload, forecast_payload

        async with self._session() as session:
            geo_payload = _as_payload(
                await session.call_tool(
                    GEOCODE_REMOTE_TOOL, {"name": place, "count": 1}
                )
            )
            coords = _coordinates_from_geocode(geo_payload)
            if coords is None:
                return geo_payload, None
            latitude, longitude, _name = coords
            forecast_payload = _as_payload(
                await session.call_tool(
                    FORECAST_REMOTE_TOOL,
                    _forecast_args(latitude, longitude),
                )
            )
            return geo_payload, forecast_payload


def _forecast_args(latitude: float, longitude: float) -> dict[str, Any]:
    return {
        "latitude": latitude,
        "longitude": longitude,
        "current_weather": True,
        "forecast_days": 1,
    }


def _result_text(result: Any) -> str:
    content = getattr(result, "content", result)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text = getattr(item, "text", None)
            if text:
                parts.append(str(text))
            elif isinstance(item, Mapping) and item.get("text"):
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def _as_payload(result: Any) -> Any:
    if getattr(result, "isError", False):
        raise WeatherMcpError(_result_text(result) or "remote tool error")
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return structured
    if hasattr(result, "content"):
        text = _result_text(result)
        return _maybe_json(text)
    if isinstance(result, str):
        return _maybe_json(result)
    return result


def _maybe_json(text: str) -> Any:
    if not text:
        return text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _coordinates_from_geocode(payload: Any) -> tuple[float, float, str] | None:
    data = payload
    if isinstance(data, str):
        data = _maybe_json(data)
    if not isinstance(data, Mapping):
        return None
    results = data.get("results")
    if not results:
        return None
    first = results[0]
    if not isinstance(first, Mapping):
        return None
    try:
        latitude = float(first["latitude"])
        longitude = float(first["longitude"])
    except (KeyError, TypeError, ValueError):
        return None
    name = str(first.get("name") or "")
    return latitude, longitude, name


def _format_forecast(payload: Any, *, location: str) -> str:
    data = payload
    if isinstance(data, str):
        parsed = _maybe_json(data)
        if parsed is data:
            return f"Weather for {location}: {data}"
        data = parsed
    if isinstance(data, Mapping):
        current = data.get("current_weather") or data.get("current") or {}
        if isinstance(current, Mapping) and current:
            parts = [f"Weather for {location}"]
            temperature = current.get("temperature")
            if temperature is not None:
                parts.append(f"temperature {temperature}°C")
            wind = current.get("windspeed", current.get("wind_speed"))
            if wind is not None:
                parts.append(f"wind {wind}")
            code = current.get("weathercode", current.get("weather_code"))
            if code is not None:
                parts.append(f"weather code {code}")
            return ". ".join(parts)
        try:
            body = json.dumps(data)
        except (TypeError, ValueError):
            body = str(data)
        return f"Weather for {location}: {body}"
    return f"Weather for {location}: {data}"


def configure_weather_client(
    url: str = DEFAULT_WEATHER_MCP_URL,
    transport: str = DEFAULT_TRANSPORT,
    *,
    call_remote: RemoteCall | None = None,
) -> WeatherMcpClient:
    """Store MCP client config for this process. Does not handshake."""
    global _client
    _client = WeatherMcpClient(url, transport, call_remote=call_remote)
    _client.attach_adapter()
    return _client


def get_weather_client() -> WeatherMcpClient:
    """Return the process-wide client, configuring it from settings if needed."""
    global _client
    if _client is None:
        from app.core.config import settings

        _client = WeatherMcpClient(
            url=settings.mcp_weather_url,
            transport=settings.mcp_weather_transport,
        )
        _client.attach_adapter()
    return _client


def reset_weather_client() -> None:
    """Drop the process-wide client (tests)."""
    global _client
    _client = None


@tool(WEATHER_TOOL_NAME)
async def get_weather(location: str) -> str:
    """Look up current weather for a city or place name via the Open-Meteo MCP server.

    Connects to the remote weather MCP on first use. If the server cannot be
    reached, returns an error description instead of crashing the chat.

    Args:
        location: City or place name, e.g. "Paris" or "Budapest".
    """
    return await get_weather_client().lookup(location)
