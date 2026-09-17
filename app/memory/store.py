"""In-memory conversation history keyed by ``session_id``.

This is a process-local dict. It is not shared across workers and is lost on
restart — see the README ``Conversation history`` section.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from langchain_core.messages import BaseMessage


class InMemorySessionStore:
    """Thread-safe (per event loop) map of session id → chat messages."""

    def __init__(self) -> None:
        self._sessions: dict[str, list[BaseMessage]] = {}
        self._lock = asyncio.Lock()

    async def get_messages(self, session_id: str) -> list[BaseMessage]:
        """Return a copy of the stored transcript, or an empty list."""
        async with self._lock:
            stored = self._sessions.get(session_id)
            return list(stored) if stored else []

    async def save_messages(
        self, session_id: str, messages: Sequence[BaseMessage]
    ) -> None:
        """Replace the stored transcript for ``session_id``."""
        async with self._lock:
            self._sessions[session_id] = list(messages)


_default_store: InMemorySessionStore | None = None


def get_session_store() -> InMemorySessionStore:
    """Return the process-wide store used by the compiled chat graph."""
    global _default_store
    if _default_store is None:
        _default_store = InMemorySessionStore()
    return _default_store
