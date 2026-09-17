"""Async HTTP routes: ``GET /chat`` and ``GET /health``."""

from __future__ import annotations

import logging
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, Query

from app.api.deps import ChatGraph, get_graph
from app.api.schemas import ChatRequest, ChatResponse, ErrorResponse, HealthResponse
from app.core.exceptions import ChatServiceError

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
    summary="Liveness probe",
    description=(
        "Returns 200 when the API process is up. Independent of the remote "
        "weather MCP and of LLM provider availability."
    ),
)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get(
    "/chat",
    response_model=ChatResponse,
    responses={
        422: {"description": "Missing or invalid query parameters."},
        502: {
            "model": ErrorResponse,
            "description": "LLM or graph failure while producing a reply.",
        },
        503: {
            "model": ErrorResponse,
            "description": "Chat graph is not available.",
        },
    },
    tags=["chat"],
    summary="Send a message to the agent",
    description=(
        "Runs one turn of the LangGraph agent. Echo + joke is agent behaviour, "
        "not a separate response field. ``language`` is applied inside the graph "
        "and is not returned. Pass ``session_id`` to continue a conversation; "
        "omit it to start a new one."
    ),
)
async def chat(
    params: Annotated[ChatRequest, Query()],
    graph: Annotated[ChatGraph, Depends(get_graph)],
) -> ChatResponse:
    session_id = params.session_id or str(uuid4())
    try:
        result = await graph.ainvoke(
            {
                "user_message": params.message,
                "language": params.language,
                "session_id": session_id,
                "messages": [],
                "final_answer": "",
            }
        )
    except ChatServiceError:
        raise
    except Exception as exc:
        logger.exception("Chat graph invocation failed")
        raise ChatServiceError(
            f"The language model failed to produce a response: {exc}",
            status_code=502,
        ) from exc

    answer = result.get("final_answer") if result is not None else None
    if not isinstance(answer, str) or not answer.strip():
        raise ChatServiceError(
            "The agent produced an empty response.",
            status_code=502,
        )

    returned_session = result.get("session_id")
    if isinstance(returned_session, str) and returned_session.strip():
        session_id = returned_session.strip()

    return ChatResponse(session_id=session_id, answer=answer)
