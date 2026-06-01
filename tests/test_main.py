"""Tests fonctionnels de l'API."""

from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Chat endpoint — succès (avec mock du pipeline RAG)
# ---------------------------------------------------------------------------
@patch("app.main.get_rag_response", new_callable=AsyncMock)
def test_chat_success(mock_rag, client):
    mock_rag.return_value = "Bonjour ! Romain est développeur."
    response = client.post(
        "/chat",
        json={"message": "Qui est Romain ?", "lang": "fr"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data
    assert data["reply"] == "Bonjour ! Romain est développeur."
    assert "thread_id" in data


@patch("app.main.get_rag_response", new_callable=AsyncMock)
def test_chat_english(mock_rag, client):
    mock_rag.return_value = "Romain is a developer."
    response = client.post(
        "/chat",
        json={"message": "Who is Romain?", "lang": "en"},
    )
    assert response.status_code == 200
    assert response.json()["reply"] == "Romain is a developer."


@patch("app.main.get_rag_response", new_callable=AsyncMock)
def test_chat_default_lang(mock_rag, client):
    """lang defaults to 'fr' when omitted."""
    mock_rag.return_value = "Réponse en français."
    response = client.post("/chat", json={"message": "Bonjour"})
    assert response.status_code == 200


@patch("app.main.get_rag_response", new_callable=AsyncMock)
def test_chat_with_thread_id(mock_rag, client):
    """thread_id is passed through and returned."""
    mock_rag.return_value = "Réponse."
    response = client.post(
        "/chat",
        json={"message": "Bonjour", "lang": "fr", "thread_id": "session-abc"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["thread_id"] == "session-abc"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------
def test_chat_empty_message(client):
    response = client.post("/chat", json={"message": "", "lang": "fr"})
    assert response.status_code == 422


def test_chat_missing_message(client):
    response = client.post("/chat", json={"lang": "fr"})
    assert response.status_code == 422


def test_chat_message_too_long(client):
    response = client.post(
        "/chat",
        json={"message": "a" * 501, "lang": "fr"},
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
