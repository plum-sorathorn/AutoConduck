"""Coding agent harnesses and tool integration adapters.

AutoConduck connects to coding agent harnesses (Claude Code, OpenCode, Pi, Aider,
Cursor, Continue, KiloCode, Generic OpenAI).
"""
from __future__ import annotations

from .base import BaseAdapter
from .aider import AiderAdapter
from .claude_code import ClaudeCodeAdapter
from .continue_dev import ContinueDevAdapter
from .cursor import CursorAdapter
from .generic_openai import GenericOpenAIAdapter
from .kilocode import KiloCodeAdapter
from .opencode import OpenCodeAdapter
from .pi import PiAdapter

__all__ = [
    "BaseAdapter",
    "AiderAdapter",
    "ClaudeCodeAdapter",
    "ContinueDevAdapter",
    "CursorAdapter",
    "GenericOpenAIAdapter",
    "KiloCodeAdapter",
    "OpenCodeAdapter",
    "PiAdapter",
    "all_adapters",
    "all_harnesses",
    "binary_name_for",
]


def all_adapters() -> list[BaseAdapter]:
    """Return instances of all supported coding harnesses/adapters."""
    return [
        ClaudeCodeAdapter(),
        OpenCodeAdapter(),
        PiAdapter(),
        AiderAdapter(),
        CursorAdapter(),
        GenericOpenAIAdapter(),
        KiloCodeAdapter(),
        ContinueDevAdapter(),
    ]


all_harnesses = all_adapters


def binary_name_for(agent_id: str) -> str | None:
    """Return the executable CLI binary name for a given harness/agent ID."""
    adapter = next((a for a in all_adapters() if a.id == agent_id), None)
    return getattr(adapter, "binary_name", None) if adapter else None
