from .base import BaseAdapter
from .claude_code import ClaudeCodeAdapter
from .opencode import OpenCodeAdapter
from .aider import AiderAdapter
from .continue_dev import ContinueDevAdapter
from .kilocode import KiloCodeAdapter
from .cursor import CursorAdapter
from .generic_openai import GenericOpenAIAdapter

__all__ = [
    "BaseAdapter",
    "ClaudeCodeAdapter",
    "OpenCodeAdapter",
    "AiderAdapter",
    "ContinueDevAdapter",
    "KiloCodeAdapter",
    "CursorAdapter",
    "GenericOpenAIAdapter",
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
    ]
