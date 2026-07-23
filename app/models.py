"""Pydantic models for request and response validation."""

from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=250,
        description="User message content",
    )
    lang: Literal["fr", "en"] = Field(
        default="fr",
        description="Response language preference",
    )
    thread_id: str | None = Field(
        default=None,
        max_length=100,
        description="Conversation thread identifier for multi-turn memory",
    )


class ChatResponse(BaseModel):
    reply: str
    thread_id: str


class ErrorResponse(BaseModel):
    detail: str
