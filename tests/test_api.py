"""HTTP API tests for GET /chat and GET /health (FastAPI TestClient)."""

from __future__ import annotations

import asyncio
from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_graph
from app.core.exceptions import ChatServiceError
from app.graph.builder import build_graph
from app.llm.mock_chat import FIXED_JOKE, MockChatModel
from app.main import create_app
from app.memory.store import InMemorySessionStore


class FakeGraph:
    def __init__(self, *, answer: str | None = None, error: Exception | None = None) -> None:
        self.calls: list[dict] = []
        self._answer = answer
        self._error = error

    async def ainvoke(self, input: dict, **kwargs: object) -> dict:
        self.calls.append(dict(input))
        if self._error is not None:
            raise self._error
        answer = self._answer
        if answer is None:
            answer = (
                f"You said: {input['user_message']}. "
                "Why did the scarecrow win an award? Because he was outstanding in his field."
            )
        return {
            "session_id": input["session_id"],
            "language": input["language"],
            "final_answer": answer,
        }


@pytest.fixture
def graph() -> FakeGraph:
    return FakeGraph()


@pytest.fixture
def client(graph: FakeGraph) -> Generator[TestClient, None, None]:
    application = create_app()
    application.dependency_overrides[get_graph] = lambda: graph
    with TestClient(application) as test_client:
        yield test_client


def test_health_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_requires_message(client: TestClient) -> None:
    response = client.get("/chat")
    assert response.status_code == 422


def test_chat_rejects_blank_message(client: TestClient) -> None:
    response = client.get("/chat", params={"message": "   "})
    assert response.status_code == 422


def test_chat_happy_path_generates_session_id(client: TestClient, graph: FakeGraph) -> None:
    response = client.get("/chat", params={"message": "Hello there"})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"session_id", "answer"}
    assert body["session_id"]
    assert "Hello there" in body["answer"]
    assert "language" not in body
    assert graph.calls[0]["user_message"] == "Hello there"
    assert graph.calls[0]["language"] == ""


def test_chat_blank_language_means_no_translation(
    client: TestClient, graph: FakeGraph
) -> None:
    response = client.get("/chat", params={"message": "Szia", "language": "  "})
    assert response.status_code == 200
    assert graph.calls[0]["language"] == ""


def test_chat_uses_provided_session_and_language(client: TestClient, graph: FakeGraph) -> None:
    response = client.get(
        "/chat",
        params={
            "message": "Hi",
            "language": "hu",
            "session_id": "session-1",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == "session-1"
    assert graph.calls[0]["language"] == "hu"
    assert graph.calls[0]["session_id"] == "session-1"


def test_chat_maps_llm_failure_to_502(graph: FakeGraph) -> None:
    graph._error = RuntimeError("invalid api key")
    application = create_app()
    application.dependency_overrides[get_graph] = lambda: graph
    with TestClient(application) as client:
        response = client.get("/chat", params={"message": "Hello"})
    assert response.status_code == 502
    assert "invalid api key" in response.json()["detail"]


def test_chat_maps_chat_service_error_status(graph: FakeGraph) -> None:
    graph._error = ChatServiceError("provider unavailable", status_code=503)
    application = create_app()
    application.dependency_overrides[get_graph] = lambda: graph
    with TestClient(application) as client:
        response = client.get("/chat", params={"message": "Hello"})
    assert response.status_code == 503
    assert response.json()["detail"] == "provider unavailable"


def test_chat_compiled_graph_echoes_message_and_joke() -> None:
    application = create_app()
    with TestClient(application) as client:
        client.app.state.graph = build_graph(
            llm=MockChatModel(), store=InMemorySessionStore()
        )
        health = client.get("/health")
        chat = client.get("/chat", params={"message": "Hello"})
    assert health.status_code == 200
    assert chat.status_code == 200
    body = chat.json()
    assert set(body) == {"session_id", "answer"}
    assert "Hello" in body["answer"]
    assert FIXED_JOKE in body["answer"]
    assert "language" not in body


def test_chat_compiled_graph_mock_language_prefix() -> None:
    application = create_app()
    with TestClient(application) as client:
        client.app.state.graph = build_graph(
            llm=MockChatModel(), store=InMemorySessionStore()
        )
        chat = client.get("/chat", params={"message": "Hello", "language": "hu"})
    assert chat.status_code == 200
    body = chat.json()
    assert set(body) == {"session_id", "answer"}
    assert body["answer"].startswith("[hu] ")
    assert "Hello" in body["answer"]
    assert FIXED_JOKE in body["answer"]
    assert "language" not in body


def test_chat_generated_session_id_continues_conversation() -> None:
    store = InMemorySessionStore()
    application = create_app()
    with TestClient(application) as client:
        client.app.state.graph = build_graph(llm=MockChatModel(), store=store)
        first = client.get("/chat", params={"message": "Hello"})
        session_id = first.json()["session_id"]
        second = client.get(
            "/chat", params={"message": "Again", "session_id": session_id}
        )
    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["session_id"] == session_id
    saved = asyncio.run(store.get_messages(session_id))
    humans = [m.content for m in saved if getattr(m, "type", None) == "human"]
    assert humans == ["Hello", "Again"]


def test_chat_sessions_do_not_share_history() -> None:
    store = InMemorySessionStore()
    application = create_app()
    with TestClient(application) as client:
        client.app.state.graph = build_graph(llm=MockChatModel(), store=store)
        client.get("/chat", params={"message": "Hello", "session_id": "keep-me"})
        client.get("/chat", params={"message": "Again", "session_id": "keep-me"})
        client.get("/chat", params={"message": "Other", "session_id": "other"})
    kept = asyncio.run(store.get_messages("keep-me"))
    other = asyncio.run(store.get_messages("other"))
    kept_humans = [m.content for m in kept if getattr(m, "type", None) == "human"]
    other_humans = [m.content for m in other if getattr(m, "type", None) == "human"]
    assert kept_humans == ["Hello", "Again"]
    assert other_humans == ["Other"]


def test_openapi_documents_chat_query_params() -> None:
    application = create_app()
    with TestClient(application) as client:
        spec = client.get("/openapi.json").json()
    chat = spec["paths"]["/chat"]["get"]
    params = {item["name"]: item for item in chat["parameters"]}
    assert params["message"]["required"] is True
    assert params["language"]["required"] is False
    assert params["session_id"]["required"] is False
    assert "/docs" not in spec["paths"]
    assert "/health" in spec["paths"]
