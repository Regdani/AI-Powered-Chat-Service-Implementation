"""Tools bound to the chat agent (Megafon + weather MCP wrapper)."""

from __future__ import annotations

from langchain_core.tools import BaseTool

from app.tools.mcp_weather import get_weather
from app.tools.megafon import megafon

__all__ = ["default_tools", "get_weather", "megafon"]


def default_tools() -> list[BaseTool]:
    """Tools attached to the graph when the caller does not inject a list."""
    return [megafon, get_weather]
