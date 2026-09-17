"""Shared LangGraph state for one chat turn."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    """Custom graph state required by the spec.

    ``language`` is applied only by the translate node. The agent node does
    not use it to choose a reply language, so translation is never done twice.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    user_message: str
    language: str
    session_id: str
    final_answer: str
