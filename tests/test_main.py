"""Functional unit tests for API endpoints."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


# Health check
def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# Chat endpoint: success with mocked RAG pipeline
@patch("app.main.get_rag_response", new_callable=AsyncMock)
def test_chat_success(mock_rag, client):
    mock_rag.return_value = ("Bonjour ! Romain est développeur.", "thread-123")
    response = client.post(
        "/chat",
        json={"message": "Qui est Romain ?", "lang": "fr"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data
    assert data["reply"] == "Bonjour ! Romain est développeur."
    assert "thread_id" in data
    assert data["thread_id"] == "thread-123"


@patch("app.main.get_rag_response", new_callable=AsyncMock)
def test_chat_english(mock_rag, client):
    mock_rag.return_value = ("Romain is a developer.", "thread-123")
    response = client.post(
        "/chat",
        json={"message": "Who is Romain?", "lang": "en"},
    )
    assert response.status_code == 200
    assert response.json()["reply"] == "Romain is a developer."


@patch("app.main.get_rag_response", new_callable=AsyncMock)
def test_chat_default_lang(mock_rag, client):
    mock_rag.return_value = ("Réponse en français.", "thread-123")
    response = client.post("/chat", json={"message": "Bonjour"})
    assert response.status_code == 200


@patch("app.main.get_rag_response", new_callable=AsyncMock)
def test_chat_with_thread_id(mock_rag, client):
    mock_rag.return_value = ("Réponse.", "session-abc")
    response = client.post(
        "/chat",
        json={"message": "Bonjour", "lang": "fr", "thread_id": "session-abc"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["thread_id"] == "session-abc"


# Validation errors
def test_chat_empty_message(client):
    response = client.post("/chat", json={"message": "", "lang": "fr"})
    assert response.status_code == 422


def test_chat_missing_message(client):
    response = client.post("/chat", json={"lang": "fr"})
    assert response.status_code == 422


def test_chat_message_too_long(client):
    response = client.post(
        "/chat",
        json={"message": "a" * 251, "lang": "fr"},
    )
    assert response.status_code == 422


def test_chat_invalid_lang(client):
    response = client.post(
        "/chat",
        json={"message": "Hello", "lang": "de"},
    )
    assert response.status_code == 422


def test_chat_invalid_json(client):
    response = client.post(
        "/chat",
        content="not json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


def test_history_pruning_filters_tool_messages():
    """Verify that conversation history pruning filters out past ToolCall and ToolMessage objects."""
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

    messages = [
        HumanMessage(content="Question 1"),
        AIMessage(content="", tool_calls=[{"name": "retrieve_documents", "args": {}, "id": "call_1"}]),
        ToolMessage(content="Heavy RAG doc content...", tool_call_id="call_1"),
        AIMessage(content="Réponse 1"),
        HumanMessage(content="Question 2"),
    ]

    human_indices = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    last_human_idx = human_indices[-1]
    pruned_past = []
    for msg in messages[:last_human_idx]:
        if isinstance(msg, HumanMessage):
            pruned_past.append(msg)
        elif isinstance(msg, AIMessage) and msg.content and str(msg.content).strip():
            pruned_past.append(AIMessage(content=str(msg.content).strip()))

    final_messages = [SystemMessage(content="sys")] + pruned_past + list(messages[last_human_idx:])

    assert len(final_messages) == 4
    assert isinstance(final_messages[0], SystemMessage)
    assert final_messages[1].content == "Question 1"
    assert final_messages[2].content == "Réponse 1"
    assert final_messages[3].content == "Question 2"
    assert not any(isinstance(m, ToolMessage) for m in final_messages)


# Admin daily stats endpoint tests
def test_admin_daily_stats_unauthorized(client):
    response = client.get("/admin/daily-stats")
    assert response.status_code == 401
    assert "Unauthorized" in response.json()["detail"]


def test_admin_daily_stats_authorized(client):
    from app.config import settings
    response = client.get(f"/admin/daily-stats?key={settings.ADMIN_KEY}")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)


def test_system_prompts_definition_and_meta_language_rules():
    """Verify that system prompts explicitly handle technical definitions and ban RAG meta-language."""
    from app.prompts import SYSTEM_PROMPT_PASS1, SYSTEM_PROMPT_PASS2

    assert "AI/IA" in SYSTEM_PROMPT_PASS1
    assert "LLM" in SYSTEM_PROMPT_PASS1

    assert "NO RAG META-LANGUAGE" in SYSTEM_PROMPT_PASS2
    assert "The documents do not specify" in SYSTEM_PROMPT_PASS2


