# Runtime image for the AI-Powered Chat Service (FastAPI + LangGraph).
# Build:  docker build -t ai-chat-service .
# Run:    docker run --rm -p 18080:18080 --env-file .env ai-chat-service

FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_HOST=0.0.0.0 \
    APP_PORT=18080

RUN useradd --create-home --uid 1000 appuser

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

USER appuser

EXPOSE 18080

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:18080/health')"

CMD ["python", "-m", "app"]
