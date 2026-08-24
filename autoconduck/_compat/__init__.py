"""Compatibility layer providing fallback shims for optional binary dependencies."""

from .llama_fallback import (
    LlamaFallback,
    LlamaGrammarFallback,
    get_llama_model,
    is_llama_cpp_available,
)
from .outlines_fallback import (
    OutlinesFallback,
    generate_structured_json,
    is_outlines_available,
)
from .lancedb_fallback import (
    LanceDBFallbackConnection,
    LanceDBFallbackQuery,
    LanceDBFallbackTable,
    connect as lancedb_connect,
    is_lancedb_available,
)
from .sqlite_checkpointer import (
    CheckpointTupleFallback,
    SqliteSaverFallback,
    get_sqlite_checkpointer,
    is_sqlite_checkpointer_available,
)

__all__ = [
    "LlamaFallback",
    "LlamaGrammarFallback",
    "get_llama_model",
    "is_llama_cpp_available",
    "OutlinesFallback",
    "generate_structured_json",
    "is_outlines_available",
    "LanceDBFallbackConnection",
    "LanceDBFallbackQuery",
    "LanceDBFallbackTable",
    "lancedb_connect",
    "is_lancedb_available",
    "CheckpointTupleFallback",
    "SqliteSaverFallback",
    "get_sqlite_checkpointer",
    "is_sqlite_checkpointer_available",
]
