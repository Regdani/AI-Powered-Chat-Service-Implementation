"""Megafon tool: shout the text after ``Announce:`` in uppercase.

Uppercasing is this tool's job. The default mock LLM never detects the
``Announce:`` prefix or emits tool calls; the graph tool-loop is driven by a
real provider or a scripted test LLM.
"""

from __future__ import annotations

import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

MEGAFON_TOOL_NAME = "megafon"
ANNOUNCE_MARKER = "announce:"


def announce_upper(text: str) -> str:
    """Return the announcement payload in uppercase.

    If ``text`` contains ``Announce:`` (any case), only the substring after that
    marker is used. Otherwise the whole string is uppercased so a model can
    pass either the raw user message or the payload alone.
    """
    raw = text if isinstance(text, str) else str(text or "")
    index = raw.lower().find(ANNOUNCE_MARKER)
    if index == -1:
        payload = raw.strip()
    else:
        payload = raw[index + len(ANNOUNCE_MARKER) :].strip()
    result = payload.upper()
    logger.info("Megafon announce input=%r output=%r", raw, result)
    return result


@tool(MEGAFON_TOOL_NAME)
def megafon(text: str) -> str:
    """Shout text through a megafon by uppercasing the part after 'Announce:'.

    Pass the raw user message (including the Announce: prefix) or the payload
    to announce. Returns that payload in uppercase.

    Args:
        text: User message or announcement payload.
    """
    return announce_upper(text)
