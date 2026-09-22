"""Pydantic request and response models for the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    """Query parameters for ``GET /chat`` (shown in Swagger)."""

    message: str = Field(
        ...,
        min_length=1,
        description="User message sent to the agent.",
        examples=["Hello there"],
    )
    language: str | None = Field(
        default=None,
        description=(
            "Optional desired response language code (e.g. en, hu). When omitted "
            "or blank, no translation runs and the agent replies in the language "
            "of the user message when possible. When set, the translate node "
            "rewrites the reply into that language. Not echoed in the response body."
        ),
        examples=["en", "hu"],
    )
    session_id: str | None = Field(
        default=None,
        description=(
            "Conversation session identifier. If omitted, the server generates a UUID. "
            "Pass the same value on later GET /chat calls so history is loaded and saved."
        ),
        examples=["3f1c8a6e-9b2d-4e71-a4c8-1d2e3f4a5b6c"],
    )

    @field_validator("message", mode="before")
    @classmethod
    def strip_required_text(cls, value: object) -> object:
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                raise ValueError("must not be blank")
            return stripped
        return value

    @field_validator("language", "session_id", mode="before")
    @classmethod
    def blank_optional_becomes_none(cls, value: object) -> object:
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value


class ChatResponse(BaseModel):
    """JSON body returned by ``GET /chat``."""

    session_id: str = Field(
        ...,
        description="Session used for this turn (client-supplied or newly generated).",
        examples=["3f1c8a6e-9b2d-4e71-a4c8-1d2e3f4a5b6c"],
    )
    answer: str = Field(
        ...,
        description="Agent reply for this turn (echo + joke, plus any tool output).",
        examples=[
            "You said: Hello there. Why did the scarecrow win an award? "
            "Because he was outstanding in his field."
        ],
    )


class HealthResponse(BaseModel):
    """JSON body returned by ``GET /health``."""

    status: str = Field(..., description="Liveness indicator.", examples=["ok"])


class ErrorResponse(BaseModel):
    """JSON body for mapped chat / LLM failures (``502`` / ``503``)."""

    detail: str = Field(
        ...,
        description=(
            "Human-readable explanation of the failure. Clients should show "
            "or log this string; it is not a machine-stable error code."
        ),
        examples=[
            "The language model failed to produce a response: 401 Unauthorized",
            "The agent produced an empty response.",
            "Chat graph is not configured.",
        ],
    )
