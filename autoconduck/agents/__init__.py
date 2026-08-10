from .base import BaseAdapter
from .claude_code import ClaudeCodeAdapter
from .opencode import OpenCodeAdapter
from .pi import PiAdapter

__all__ = [
    "BaseAdapter",
    "ClaudeCodeAdapter",
    "OpenCodeAdapter",
    "PiAdapter",
]

def all_adapters() -> list[BaseAdapter]:
    return [
        ClaudeCodeAdapter(),
        OpenCodeAdapter(),
        PiAdapter(),
    ]

def binary_name_for(agent_id: str) -> str | None:
    adapter = next((a for a in all_adapters() if a.id == agent_id), None)
    return getattr(adapter, "binary_name", None) if adapter else None
