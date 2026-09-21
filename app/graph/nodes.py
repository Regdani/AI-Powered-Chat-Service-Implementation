"""Graph nodes: load history → agent (echo + joke, optional tools) → translate → save."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from app.graph.state import GraphState
from app.llm.mock_chat import MockChatModel, mock_language_marker
from app.memory.store import InMemorySessionStore

logger = logging.getLogger(__name__)

LOAD_HISTORY_NODE = "load_history"
AGENT_NODE = "agent"
TOOLS_NODE = "tools"
TRANSLATE_NODE = "translate"
SAVE_HISTORY_NODE = "save_history"

DEFAULT_LANGUAGE = "en"

AGENT_SYSTEM_PROMPT = (
    "You are a chat agent. For every user message you must (1) repeat the "
    "user's text back and (2) add a short joke. If the user wants something "
    "announced (text after 'Announce:'), call the megafon tool with that text "
    "and include the shouted result in your reply. If the user asks about "
    "current weather, call the get_weather tool with the location and include "
    "the result in your reply. When the user asks about previous messages, "
    "use only the conversation history of the current session. If the "
    "requested information is not available in the conversation history, "
    "explicitly state that you don't have that information. Do not call "
    "external tools to reconstruct or guess information from previous "
    "messages. Do not translate or rewrite the reply into a requested "
    "language; another graph node handles language."
)

TRANSLATE_SYSTEM_PROMPT = (
    "You are a translator. Translate the text into the language identified "
    "by this language code: {language}. Return only the translated text. "
    "Do not add a joke, preface, or explanation."
)


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    text = getattr(message, "text", None)
    if isinstance(text, str):
        return text
    return str(content or "")


def _bind_model(llm: BaseChatModel, tools: Sequence[Any] | None) -> Runnable:
    if tools:
        return llm.bind_tools(list(tools))
    return llm


def _last_is_tool_result(messages: Sequence[BaseMessage]) -> bool:
    """True when the agent is being re-invoked after the tools node."""
    if not messages:
        return False
    return getattr(messages[-1], "type", None) == "tool"


def _tool_call_names(message: AIMessage) -> list[str]:
    names: list[str] = []
    for call in message.tool_calls or []:
        if isinstance(call, dict):
            names.append(str(call.get("name") or ""))
        else:
            names.append(str(getattr(call, "name", "") or ""))
    return names


def normalize_language(language: str | None) -> str:
    """Return a lowercase language code; blank values become ``en``."""
    if language is None:
        return DEFAULT_LANGUAGE
    stripped = str(language).strip().lower()
    return stripped or DEFAULT_LANGUAGE


def needs_translation(language: str | None, text: str | None) -> bool:
    """True when the translate node should run (non-English with text)."""
    if not (text or "").strip():
        return False
    return normalize_language(language) != DEFAULT_LANGUAGE


def route_after_agent(state: GraphState) -> str:
    """Route to tools when the last AI message requested tool calls."""
    messages = state.get("messages") or []
    if messages:
        last = messages[-1]
        if getattr(last, "tool_calls", None):
            return TOOLS_NODE
    return TRANSLATE_NODE


def _answer_from_state(state: GraphState) -> str:
    answer = state.get("final_answer") or ""
    if answer.strip():
        return answer
    messages = state.get("messages") or []
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not getattr(message, "tool_calls", None):
            return _message_text(message)
    return ""


def make_agent_node(
    llm: BaseChatModel,
    tools: Sequence[Any] | None = None,
):
    """Return an async agent node: LLM ``ainvoke``, then log the reply.

    After the tools node runs, this node is invoked again. In that follow-up
    the current user turn is already in ``messages``, so it is not appended.
    """

    model = _bind_model(llm, tools)

    async def agent(state: GraphState) -> dict[str, Any]:
        user_message = state["user_message"]
        history = list(state.get("messages") or [])
        follow_up = _last_is_tool_result(history)

        llm_messages: list[BaseMessage] = [
            SystemMessage(content=AGENT_SYSTEM_PROMPT),
            *history,
        ]
        outgoing: list[BaseMessage] = []
        if not follow_up:
            human = HumanMessage(content=user_message)
            llm_messages.append(human)
            outgoing.append(human)

        response = await model.ainvoke(llm_messages)
        if not isinstance(response, AIMessage):
            response = AIMessage(content=_message_text(response))
        outgoing.append(response)

        update: dict[str, Any] = {"messages": outgoing}
        if response.tool_calls:
            logger.info(
                "Agent tool calls session_id=%s names=%s",
                state.get("session_id"),
                _tool_call_names(response),
            )
            return update

        answer = _message_text(response)
        logger.info(
            "Agent reply session_id=%s answer=%s",
            state.get("session_id"),
            answer,
        )
        update["final_answer"] = answer
        return update

    return agent


def make_load_history_node(store: InMemorySessionStore):
    """Return an async node that hydrates ``messages`` from the session store."""

    async def load_history(state: GraphState) -> dict[str, Any]:
        session_id = state["session_id"]
        messages = await store.get_messages(session_id)
        logger.debug(
            "Loaded history session_id=%s message_count=%s",
            session_id,
            len(messages),
        )
        return {"messages": messages}

    return load_history


def make_translate_node(llm: BaseChatModel):
    """Return an async node that is the only place ``language`` is applied.

    English (default) or an empty answer is a no-op: no extra LLM call.
    Any other language makes **one** translator call on the agent's finished
    reply. The mock LLM is not asked to translate; it only gets a ``[hu] ``
    style prefix so the node can be tested without a real model.

    History is left in the agent's original language so the next turn is not
    double-translated and the agent never sees a target-language instruction.
    """

    async def translate(state: GraphState) -> dict[str, Any]:
        language = normalize_language(state.get("language"))
        answer = _answer_from_state(state)
        session_id = state.get("session_id")

        if not needs_translation(language, answer):
            logger.debug(
                "Translate skip session_id=%s language=%s",
                session_id,
                language,
            )
            return {"final_answer": answer}

        if isinstance(llm, MockChatModel):
            tagged = mock_language_marker(answer, language)
            logger.info(
                "Translate mock marker session_id=%s language=%s answer=%s",
                session_id,
                language,
                tagged,
            )
            return {"final_answer": tagged}

        response = await llm.ainvoke(
            [
                SystemMessage(content=TRANSLATE_SYSTEM_PROMPT.format(language=language)),
                HumanMessage(content=answer),
            ]
        )
        translated = _message_text(response).strip()
        if not translated:
            logger.warning(
                "Translate produced empty output session_id=%s language=%s; "
                "keeping the agent reply",
                session_id,
                language,
            )
            return {"final_answer": answer}

        logger.info(
            "Translate applied session_id=%s language=%s answer=%s",
            session_id,
            language,
            translated,
        )
        return {"final_answer": translated}

    return translate


def make_save_history_node(store: InMemorySessionStore):
    """Return an async node that persists ``messages`` for this session."""

    async def save_history(state: GraphState) -> dict[str, Any]:
        session_id = state["session_id"]
        messages = list(state.get("messages") or [])
        await store.save_messages(session_id, messages)
        logger.debug(
            "Saved history session_id=%s message_count=%s",
            session_id,
            len(messages),
        )
        return {}

    return save_history
