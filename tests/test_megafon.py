"""Unit tests for the Megafon tool (Announce: → UPPER)."""

from __future__ import annotations

import logging
from typing import Any

from app.tools.megafon import MEGAFON_TOOL_NAME, announce_upper, megafon


def test_announce_prefix_is_uppercased() -> None:
    assert announce_upper("Announce: hello world") == "HELLO WORLD"


def test_announce_prefix_is_case_insensitive() -> None:
    assert announce_upper("ANNOUNCE: mixed Case") == "MIXED CASE"
    assert announce_upper("announce: quietly") == "QUIETLY"


def test_payload_without_prefix_is_uppercased() -> None:
    assert announce_upper("hello world") == "HELLO WORLD"


def test_whitespace_around_payload_is_stripped() -> None:
    assert announce_upper("Announce:   hello   ") == "HELLO"
    assert announce_upper("  Announce: hi") == "HI"


def test_empty_payload_after_prefix() -> None:
    assert announce_upper("Announce:") == ""
    assert announce_upper("Announce:   ") == ""


def test_text_before_prefix_is_ignored() -> None:
    assert announce_upper("please Announce: doors closing") == "DOORS CLOSING"


def test_tool_name_and_invoke() -> None:
    assert megafon.name == MEGAFON_TOOL_NAME
    assert megafon.invoke({"text": "Announce: hello world"}) == "HELLO WORLD"


def test_tool_invoke_without_prefix() -> None:
    assert megafon.invoke({"text": "loud enough"}) == "LOUD ENOUGH"


def test_megafon_logs_input_and_output(caplog: Any) -> None:
    with caplog.at_level(logging.INFO, logger="app.tools.megafon"):
        result = announce_upper("Announce: hello world")
    assert result == "HELLO WORLD"
    assert "Megafon announce" in caplog.text
    assert "hello world" in caplog.text
    assert "HELLO WORLD" in caplog.text
