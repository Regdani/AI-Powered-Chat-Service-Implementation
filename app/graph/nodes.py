"""Graph nodes: load history → agent (echo + joke, optional tools) → save history."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from app.graph.state import GraphState
from app.memory.store import InMemorySessionStore

logger = logging.getLogger(__name__)

LOAD_HISTORY_NODE = "load_history"
AGENT_NODE = "agent"
TOOLS_NODE = "tools"
SAVE_HISTORY_NODE = "save_history"

AGENT_SYSTEM_PROMPT = (
    "You are a chat agent. For every user message you must (1) repeat the "
    "user's text back and (2) add a short joke. If the user wants something "
    "announced (text after 'Announce:'), call the megafon tool with that text "
    "and include the shouted result in your reply. Do not translate or rewrite "
    "the reply into a requested language; another graph node handles language."
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


def route_after_agent(state: GraphState) -> str:
    """Route to tools when the last AI message requested tool calls."""
    messages = state.get("messages") or []
    if messages:
        last = messages[-1]
        if getattr(last, "tool_calls", None):
            return TOOLS_NODE
    return SAVE_HISTORY_NODE


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
