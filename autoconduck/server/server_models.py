"""Pydantic request payload schemas for OpenAI and Anthropic endpoints."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class CompletionRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]] = Field(default_factory=list)
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None


class MessagesRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]] = Field(default_factory=list)
    system: Any | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop_sequences: list[str] | None = None
    stream: bool = True
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None
    thinking: Any | None = None
    metadata: dict[str, Any] | None = None
    cache_control: Any | None = None
