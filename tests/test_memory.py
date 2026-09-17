"""In-memory session store: isolated transcripts, copies, missing sessions."""

from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage, HumanMessage

from app.memory.store import InMemorySessionStore


def test_missing_session_returns_empty_list() -> None:
    store = InMemorySessionStore()
    assert asyncio.run(store.get_messages("unknown")) == []


def test_save_then_get_returns_same_transcript() -> None:
    store = InMemorySessionStore()
    messages = [HumanMessage(content="Hi"), AIMessage(content="Hello")]
    asyncio.run(store.save_messages("s1", messages))
    loaded = asyncio.run(store.get_messages("s1"))
    assert len(loaded) == 2
    assert loaded[0].content == "Hi"
    assert loaded[1].content == "Hello"


def test_get_returns_a_copy() -> None:
    store = InMemorySessionStore()
    asyncio.run(store.save_messages("s1", [HumanMessage(content="Hi")]))
    loaded = asyncio.run(store.get_messages("s1"))
    loaded.append(AIMessage(content="mutated"))
    assert len(asyncio.run(store.get_messages("s1"))) == 1


def test_sessions_are_isolated() -> None:
    store = InMemorySessionStore()
    asyncio.run(store.save_messages("a", [HumanMessage(content="Alpha")]))
    asyncio.run(store.save_messages("b", [HumanMessage(content="Beta")]))
    assert asyncio.run(store.get_messages("a"))[0].content == "Alpha"
    assert asyncio.run(store.get_messages("b"))[0].content == "Beta"


def test_save_replaces_previous_transcript() -> None:
    store = InMemorySessionStore()
    asyncio.run(store.save_messages("s1", [HumanMessage(content="old")]))
    asyncio.run(store.save_messages("s1", [HumanMessage(content="new")]))
    loaded = asyncio.run(store.get_messages("s1"))
    assert len(loaded) == 1
    assert loaded[0].content == "new"
