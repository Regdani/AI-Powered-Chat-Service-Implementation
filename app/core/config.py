"""Environment-backed application settings.

Provider selection is loaded here; validation and fail-fast live in the LLM
factory so a missing/invalid real-provider config never silently becomes mock.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration from environment variables and optional ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: str = Field(
        default="mock",
        description="LLM backend: mock (default), azure, or gemini.",
    )

    azure_openai_api_key: str | None = None
    azure_openai_endpoint: str | None = None
    azure_openai_deployment: str | None = None
    azure_openai_api_version: str = "2024-10-21"

    google_api_key: str | None = None
    gemini_model: str = "gemini-3.1-flash-lite"

    mcp_weather_url: str = "https://weather.chukai.io/mcp"
    mcp_weather_transport: str = "streamable_http"
    app_host: str = "0.0.0.0"
    app_port: int = 18080


settings = Settings()
