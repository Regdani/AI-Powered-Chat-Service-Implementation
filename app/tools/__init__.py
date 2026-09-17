"""Tools bound to the chat agent (Megafon; weather MCP is added later)."""

from __future__ import annotations

from langchain_core.tools import BaseTool

from app.tools.megafon import megafon

__all__ = ["default_tools", "megafon"]


def default_tools() -> list[BaseTool]:
    """Tools attached to the graph when the caller does not inject a list."""
    return [megafon]
