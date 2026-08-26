"""Coding agent harnesses and tool integration adapters.

AutoConduck connects to coding agent harnesses (Claude Code, OpenCode, Pi).
"""

from __future__ import annotations

from .base import BaseAdapter
from .claude_code import ClaudeCodeAdapter
from .opencode import OpenCodeAdapter
from .pi import PiAdapter
from .omp import OmpAdapter

__all__ = [
    "BaseAdapter",
    "ClaudeCodeAdapter",
    "OpenCodeAdapter",
    "PiAdapter",
    "OmpAdapter",
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
        OmpAdapter(),
    ]


all_harnesses = all_adapters


def binary_name_for(agent_id: str) -> str | None:
    """Return the executable CLI binary name for a given harness/agent ID."""
    adapter = next((a for a in all_adapters() if a.id == agent_id), None)
    return getattr(adapter, "binary_name", None) if adapter else None
