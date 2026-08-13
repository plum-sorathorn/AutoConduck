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
]

def all_adapters() -> list[BaseAdapter]:
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

def binary_name_for(agent_id: str) -> str | None:
    adapter = next((a for a in all_adapters() if a.id == agent_id), None)
    return getattr(adapter, "binary_name", None) if adapter else None
