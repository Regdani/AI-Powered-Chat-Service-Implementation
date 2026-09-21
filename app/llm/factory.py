"""Create the configured chat model.

Mock is the explicit default (missing or ``LLM_PROVIDER=mock``). Selecting a
real provider with incomplete or unknown configuration fails fast — the factory
never falls back to mock.
"""

from __future__ import annotations

from collections.abc import Mapping

from langchain_core.language_models.chat_models import BaseChatModel

from app.core.config import Settings, settings as default_settings
from app.core.exceptions import ProviderConfigError
from app.llm.mock_chat import MockChatModel

SUPPORTED_PROVIDERS = ("mock", "azure", "gemini")

_NO_MOCK_FALLBACK = "The process will not fall back to the mock LLM."


def _normalize_provider(value: str | None) -> str:
    if value is None:
        return "mock"
    return value.strip().lower() or "mock"


def _missing(required: Mapping[str, str | None]) -> list[str]:
    return [
        name
        for name, value in required.items()
        if value is None or not str(value).strip()
    ]


def _require(provider: str, required: Mapping[str, str | None]) -> None:
    missing = _missing(required)
    if missing:
        raise ProviderConfigError(
            f"LLM_PROVIDER={provider} is missing required environment "
            f"variable(s): {', '.join(missing)}. {_NO_MOCK_FALLBACK}"
        )


def _create_azure(cfg: Settings) -> BaseChatModel:
    _require(
        "azure",
        {
            "AZURE_OPENAI_API_KEY": cfg.azure_openai_api_key,
            "AZURE_OPENAI_ENDPOINT": cfg.azure_openai_endpoint,
            "AZURE_OPENAI_DEPLOYMENT": cfg.azure_openai_deployment,
        },
    )
    try:
        from langchain_openai import AzureChatOpenAI
    except ImportError as exc:
        raise ProviderConfigError(
            "LLM_PROVIDER=azure requires the langchain-openai package. "
            f"{_NO_MOCK_FALLBACK}"
        ) from exc
    return AzureChatOpenAI(
        azure_deployment=cfg.azure_openai_deployment,
        azure_endpoint=cfg.azure_openai_endpoint,
        api_key=cfg.azure_openai_api_key,
        api_version=cfg.azure_openai_api_version,
    )


def _create_gemini(cfg: Settings) -> BaseChatModel:
    _require("gemini", {"GOOGLE_API_KEY": cfg.google_api_key})
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        raise ProviderConfigError(
            "LLM_PROVIDER=gemini requires the langchain-google-genai package. "
            f"{_NO_MOCK_FALLBACK}"
        ) from exc
    return ChatGoogleGenerativeAI(
        model=cfg.gemini_model,
        google_api_key=cfg.google_api_key,
    )


def create_chat_model(settings: Settings | None = None) -> BaseChatModel:
    """Return the chat model for ``settings.llm_provider``.

    Unknown providers and incomplete real-provider config raise
    ``ProviderConfigError`` (a ``ValueError``) so startup can fail fast.
    """
    cfg = default_settings if settings is None else settings
    provider = _normalize_provider(cfg.llm_provider)

    if provider == "mock":
        return MockChatModel()
    if provider == "azure":
        return _create_azure(cfg)
    if provider == "gemini":
        return _create_gemini(cfg)

    supported = ", ".join(SUPPORTED_PROVIDERS)
    raise ProviderConfigError(
        f"Unknown LLM_PROVIDER={cfg.llm_provider!r}. Expected one of: "
        f"{supported}. {_NO_MOCK_FALLBACK}"
    )
