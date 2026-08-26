"""Data models for Selective Knowledge / RAG subsystem."""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class CodeChunk(BaseModel):
    """Represents a discrete semantic code symbol, contract, or file chunk."""

    id: str
    symbol: str
    file: str
    content: str
    vector: list[float] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResult(BaseModel):
    """Search match result with distance metrics."""

    chunk: CodeChunk
    score: float = 0.0
    distance: float = 0.0
