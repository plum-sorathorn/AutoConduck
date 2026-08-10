from .base import BaseAdapter
from .claude_code import ClaudeCodeAdapter
from .opencode import OpenCodeAdapter
from .aider import AiderAdapter
from .continue_dev import ContinueDevAdapter
from .kilocode import KiloCodeAdapter
from .cursor import CursorAdapter
from .generic_openai import GenericOpenAIAdapter
from .pi import PiAdapter

__all__ = [
    "BaseAdapter",
    "ClaudeCodeAdapter",
    "OpenCodeAdapter",
    "AiderAdapter",
    "ContinueDevAdapter",
    "KiloCodeAdapter",
    "CursorAdapter",
    "GenericOpenAIAdapter",
    "PiAdapter",
]

def all_adapters() -> list[BaseAdapter]:
    return [
        ClaudeCodeAdapter(),
        OpenCodeAdapter(),
        AiderAdapter(),
        ContinueDevAdapter(),
        KiloCodeAdapter(),
        CursorAdapter(),
        GenericOpenAIAdapter(),
        PiAdapter(),
    ]

def binary_name_for(agent_id: str) -> str | None:
    adapter = next((a for a in all_adapters() if a.id == agent_id), None)
    return getattr(adapter, "binary_name", None) if adapter else None
