"""Pydantic models — validation stricte des requêtes et réponses."""

from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=500,
        strip_whitespace=True,
        description="Message de l'utilisateur",
    )
    lang: Literal["fr", "en"] = Field(
        default="fr",
        description="Langue de la réponse",
    )
    thread_id: str | None = Field(
        default=None,
        max_length=100,
        description="Identifiant de conversation pour la mémoire multi-turn",
    )


class ChatResponse(BaseModel):
    reply: str
    thread_id: str


class ErrorResponse(BaseModel):
    detail: str
