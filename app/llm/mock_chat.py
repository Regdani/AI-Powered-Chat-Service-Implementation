"""Deterministic LangChain chat model: echo the user text and append a fixed joke.

This is intentionally not a mini-agent. It does not inspect keywords, does not
route to tools, and never emits ``tool_calls``. Tool-calling is covered by a
scripted test LLM or a real provider.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable

FIXED_JOKE = (
    "Why did the scarecrow win an award? Because he was outstanding in his field."
)


def echo_with_joke(user_text: str) -> str:
    """Format the only response this mock ever produces."""
    return f"You said: {user_text}. {FIXED_JOKE}"


def mock_language_marker(text: str, language: str) -> str:
    """Deterministic language tag used by the translate node in mock mode.

    This is not a translation. Real ``LLM_PROVIDER`` values run an actual
    translator LLM call; mock only prefixes so the node can be tested.
    """
    code = language.strip().lower()
    return f"[{code}] {text}"


def _as_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            elif hasattr(block, "text"):
                parts.append(str(block.text))
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def _last_user_text(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if getattr(message, "type", None) == "human":
            return _as_text(message.content)
    if messages:
        return _as_text(messages[-1].content)
    return ""


class MockChatModel(BaseChatModel):
    """Minimal ``BaseChatModel`` used when ``LLM_PROVIDER`` is unset or ``mock``."""

    @property
    def _llm_type(self) -> str:
        return "mock-echo-joke"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self._llm_type}

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        message = AIMessage(content=echo_with_joke(_last_user_text(messages)))
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del run_manager
        return self._generate(messages, stop=stop, **kwargs)

    def bind_tools(
        self,
        tools: Sequence[Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable:
        """Accept ``bind_tools`` so the graph can attach tools; never call them."""
        del tools, tool_choice, kwargs
        return self
