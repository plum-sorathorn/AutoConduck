"""Session Lifecycle & Context Guard.

Enforces:
- Immutable prefix contract: preserving upstream prompt caching across 40+ turns.
- 80% context window ceiling compaction.
- Preserves code fences (```...```) and structural markdown headers (#, ##).
- Strict SessionGuardResult contract validation.
"""
from __future__ import annotations

import copy
import logging
import re
from typing import Any
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class SessionGuardResult(BaseModel):
    messages: list[dict[str, Any]]
    compacted: bool
    original_tokens: int
    final_tokens: int
    cache_prefix_preserved: bool

    @property
    def compacted_messages(self) -> list[dict[str, Any]]:
        return self.messages


def _count_tokens_text(text: str) -> int:
    """Estimate token count based on words and characters."""
    if not text:
        return 0
    words = len(text.split())
    chars = len(text) // 4
    return max(words, chars)


def _count_tokens_messages(messages: list[dict[str, Any]]) -> int:
    """Calculate token count across messages list."""
    total = 0
    for m in messages:
        if not isinstance(m, dict):
            continue
        c = m.get("content")
        if isinstance(c, str):
            total += _count_tokens_text(c)
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict):
                    total += _count_tokens_text(str(part.get("text") or part.get("content") or ""))
                elif isinstance(part, str):
                    total += _count_tokens_text(part)
        tc = m.get("tool_calls")
        if isinstance(tc, list):
            for call in tc:
                if isinstance(call, dict):
                    fn = call.get("function", {})
                    total += _count_tokens_text(str(fn.get("arguments", "")))
    return max(1, total)


def _compact_text_content(content: str, max_chars: int = 1200) -> str:
    """Compact text while strictly preserving code blocks and markdown headers."""
    if not content or len(content) <= max_chars:
        return content

    # Find code blocks (including unclosed code blocks)
    code_blocks: list[str] = []
    code_pattern = re.compile(r"```[a-zA-Z0-9_-]*\n[\s\S]*?(?:```|$)", re.MULTILINE)
    for m in code_pattern.finditer(content):
        code_blocks.append(m.group(0))

    # Find markdown headers
    header_lines: list[str] = []
    for line in content.splitlines():
        trimmed = line.strip()
        if trimmed.startswith(("# ", "## ", "### ", "#### ", "##### ", "###### ")):
            header_lines.append(trimmed)

    # If code blocks or headers exist, assemble preserved components
    if code_blocks or header_lines:
        parts: list[str] = []
        if header_lines:
            parts.append("\n".join(header_lines[:6]))
        if code_blocks:
            for cb in code_blocks[:3]:
                # If code block is unclosed, ensure it closes cleanly if appropriate
                cb_clean = cb.strip()
                if not cb_clean.endswith("```") and cb_clean.startswith("```"):
                    cb_clean = cb_clean + "\n```"
                parts.append(cb_clean)

        parts.append(f"[... content compacted for context budget ...]")
        assembled = "\n\n".join(parts)
        return assembled

    # Otherwise head + tail compaction for raw tool outputs or long text
    head_len = min(400, max_chars // 3)
    tail_len = min(400, max_chars // 3)
    head = content[:head_len]
    tail = content[-tail_len:]
    omitted = len(content) - (head_len + tail_len)
    return f"{head}\n\n[... {omitted} characters compacted ...]\n\n{tail}"


class SessionGuard:
    """Session lifecycle manager enforcing prefix immutability and context window ceiling."""

    def __init__(self, max_turns: int = 40, max_tokens: int = 128000):
        self.max_turns = max_turns
        self.max_tokens = max_tokens

    def check_and_compact(
        self, messages: list[dict[str, Any]], max_tokens: int | None = None
    ) -> SessionGuardResult:
        """Alias method for guard_context."""
        limit = max_tokens or self.max_tokens
        return self.guard_context(messages, context_window=limit)

    def guard_context(
        self, messages: list[dict[str, Any]], context_window: int = 128000
    ) -> SessionGuardResult:
        """Guard context window, preserving prompt cache prefix and compacting if needed."""
        if not messages:
            return SessionGuardResult(
                messages=[],
                compacted=False,
                original_tokens=0,
                final_tokens=0,
                cache_prefix_preserved=True,
            )

        effective_window = context_window if context_window > 0 else 8192
        ceiling = int(effective_window * 0.80)
        original_tokens = _count_tokens_messages(messages)

        # Snapshot immutable prefix (messages 0 and 1)
        prefix_len = min(2, len(messages))
        prefix_snapshot = [copy.deepcopy(messages[i]) for i in range(prefix_len)]

        if original_tokens <= ceiling:
            # Under 80% ceiling: no compaction needed
            return SessionGuardResult(
                messages=messages,
                compacted=False,
                original_tokens=original_tokens,
                final_tokens=original_tokens,
                cache_prefix_preserved=True,
            )

        # Compaction needed: keep prefix intact, compact subsequent messages
        compacted_messages: list[dict[str, Any]] = []

        # 1. Add immutable prefix untouched
        for i in range(prefix_len):
            compacted_messages.append(copy.deepcopy(prefix_snapshot[i]))

        # 2. Compact subsequent messages
        remaining = messages[prefix_len:]
        per_msg_char_cap = max(600, (ceiling * 4) // max(1, len(remaining)))

        for m in remaining:
            m_copy = copy.deepcopy(m)
            c = m_copy.get("content")
            if isinstance(c, str) and len(c) > per_msg_char_cap:
                m_copy["content"] = _compact_text_content(c, max_chars=per_msg_char_cap)
            compacted_messages.append(m_copy)

        final_tokens = _count_tokens_messages(compacted_messages)

        # Verify prefix integrity
        prefix_preserved = True
        for i in range(prefix_len):
            if compacted_messages[i] != prefix_snapshot[i]:
                prefix_preserved = False
                compacted_messages[i] = copy.deepcopy(prefix_snapshot[i])

        return SessionGuardResult(
            messages=compacted_messages,
            compacted=True,
            original_tokens=original_tokens,
            final_tokens=final_tokens,
            cache_prefix_preserved=prefix_preserved,
        )
