"""LLM factory and mock chat model tests (no paid API keys required)."""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI

from app.core.config import Settings
from app.core.exceptions import ProviderConfigError
from app.llm.factory import create_chat_model
from app.llm.mock_chat import FIXED_JOKE, MockChatModel, echo_with_joke


def _settings(**kwargs: object) -> Settings:
    """Build settings that do not pick up a local ``.env`` file.

    Explicit kwargs override process environment (pydantic-settings init order).
    """
    values: dict[str, object] = {
        "llm_provider": "mock",
        "azure_openai_api_key": None,
        "azure_openai_endpoint": None,
        "azure_openai_deployment": None,
        "google_api_key": None,
    }
    values.update(kwargs)
    return Settings(_env_file=None, **values)


def test_default_and_explicit_mock_are_the_same_type() -> None:
    assert isinstance(create_chat_model(_settings()), MockChatModel)
    assert isinstance(create_chat_model(_settings(llm_provider="mock")), MockChatModel)
    assert isinstance(create_chat_model(_settings(llm_provider="MOCK")), MockChatModel)
    assert isinstance(create_chat_model(_settings(llm_provider="  ")), MockChatModel)


def test_mock_echoes_user_text_and_fixed_joke() -> None:
    model = MockChatModel()
    result = model.invoke([HumanMessage(content="Hello there")])
    assert result.content == echo_with_joke("Hello there")
    assert "Hello there" in result.content
    assert FIXED_JOKE in result.content
    assert result.tool_calls == []


def test_mock_is_deterministic() -> None:
    model = MockChatModel()
    first = model.invoke([HumanMessage(content="Ping")])
    second = model.invoke([HumanMessage(content="Ping")])
    assert first.content == second.content


def test_mock_has_no_keyword_or_tool_heuristics() -> None:
    model = MockChatModel()
    announce = model.invoke([HumanMessage(content="Announce: hello world")])
    weather = model.invoke([HumanMessage(content="What is the weather in Paris?")])
    assert announce.content == echo_with_joke("Announce: hello world")
    assert weather.content == echo_with_joke("What is the weather in Paris?")
    assert announce.tool_calls == []
    assert weather.tool_calls == []
    bound = model.bind_tools([])
    still_dumb = bound.invoke([HumanMessage(content="Announce: loud")])
    assert still_dumb.tool_calls == []
    assert still_dumb.content == echo_with_joke("Announce: loud")


def test_mock_ainvoke_matches_invoke() -> None:
    model = MockChatModel()
    sync = model.invoke([HumanMessage(content="Hi")])
    async_result = asyncio.run(model.ainvoke([HumanMessage(content="Hi")]))
    assert async_result.content == sync.content


def test_unknown_provider_fails_fast_without_mock_fallback() -> None:
    with pytest.raises(ProviderConfigError, match="ollama") as exc_info:
        create_chat_model(_settings(llm_provider="ollama"))
    assert isinstance(exc_info.value, ValueError)
    assert "will not fall back" in str(exc_info.value)


def test_removed_openai_provider_fails_as_unknown() -> None:
    with pytest.raises(ProviderConfigError, match="openai") as exc_info:
        create_chat_model(_settings(llm_provider="openai"))
    assert "will not fall back" in str(exc_info.value)
    assert "mock" in str(exc_info.value)
    assert "azure" in str(exc_info.value)
    assert "gemini" in str(exc_info.value)


def test_azure_missing_env_lists_each_variable() -> None:
    with pytest.raises(ProviderConfigError) as exc_info:
        create_chat_model(_settings(llm_provider="azure"))
    message = str(exc_info.value)
    assert "AZURE_OPENAI_API_KEY" in message
    assert "AZURE_OPENAI_ENDPOINT" in message
    assert "AZURE_OPENAI_DEPLOYMENT" in message
    assert "will not fall back" in message


def test_azure_partial_config_still_fails_fast() -> None:
    with pytest.raises(ProviderConfigError, match="AZURE_OPENAI_DEPLOYMENT"):
        create_chat_model(
            _settings(
                llm_provider="azure",
                azure_openai_api_key="key",
                azure_openai_endpoint="https://example.openai.azure.com",
            )
        )


def test_azure_with_full_config_does_not_return_mock() -> None:
    model = create_chat_model(
        _settings(
            llm_provider="azure",
            azure_openai_api_key="key",
            azure_openai_endpoint="https://example.openai.azure.com",
            azure_openai_deployment="gpt-4o-mini",
        )
    )
    assert isinstance(model, AzureChatOpenAI)
    assert not isinstance(model, MockChatModel)


def test_gemini_missing_key_fails_fast() -> None:
    with pytest.raises(ProviderConfigError, match="GOOGLE_API_KEY"):
        create_chat_model(_settings(llm_provider="gemini", google_api_key=None))


def test_gemini_with_key_does_not_return_mock() -> None:
    model = create_chat_model(
        _settings(llm_provider="gemini", google_api_key="test-key")
    )
    assert isinstance(model, ChatGoogleGenerativeAI)
    assert not isinstance(model, MockChatModel)
