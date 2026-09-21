"""Run the API with host/port from settings (``.env`` / defaults).

Usage::

    python -m app

Defaults to ``0.0.0.0:18080``. Override with ``APP_HOST`` / ``APP_PORT``.
"""

from __future__ import annotations

import uvicorn

from app.core.config import settings


def main() -> None:
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
    )


if __name__ == "__main__":
    main()
