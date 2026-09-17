"""Shared application errors for startup and request-time failures."""


class ChatServiceError(Exception):
    """Raised when a chat request cannot be completed.

    Mapped to an HTTP 4xx/5xx response. The app never falls back to the mock
    LLM when a real provider was selected and then failed.
    """

    def __init__(self, message: str, *, status_code: int = 502) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class ProviderConfigError(ValueError):
    """Invalid or incomplete LLM provider configuration.

    Raised at startup so the process fails fast instead of silently using mock.
    """
