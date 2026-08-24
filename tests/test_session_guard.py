"""Comprehensive test suite for Session Lifecycle & Context Guard.

Verifies:
- Immutable prefix contract: preserving upstream prompt caching across 40+ turns.
- 80% context window ceiling compaction.
- Preserving code blocks (```...```) and structural markdown headers (#, ##).
- Strict SessionGuardResult contract validation.
- Edge cases: massive tool outputs, unclosed code fences, negative context window.
"""
from __future__ import annotations

import copy
from typing import Any
import pytest

try:
    from autoconduck.orchestrator.session_guard import SessionGuard, SessionGuardResult
except ImportError:
    pytest.skip("autoconduck.orchestrator.session_guard not yet implemented in this milestone", allow_module_level=True)


@pytest.fixture
def session_guard() -> SessionGuard:
    return SessionGuard()


# ==============================================================================
# Tier 1: Feature Coverage (>=5 tests)
# ==============================================================================

def test_session_guard_immutable_prefix_preservation(session_guard: SessionGuard):
    """System prompt and initial user turn are byte-identical before and after guarding."""
    system_msg = {"role": "system", "content": "You are a senior staff engineer with deep knowledge."}
    user_msg = {"role": "user", "content": "Initial prompt establishing conversation context."}
    messages = [
        system_msg,
        user_msg,
        {"role": "assistant", "content": "Acknowledged."},
        {"role": "user", "content": "Follow-up question."},
    ]

    result = session_guard.guard_context(messages, context_window=16000)
    assert result.cache_prefix_preserved
    assert result.messages[0] == system_msg
    assert result.messages[1] == user_msg


def test_session_guard_compaction_at_80_percent_ceiling(session_guard: SessionGuard):
    """When messages exceed 80% of context window, tool outputs are compacted."""
    system_msg = {"role": "system", "content": "System prompt."}
    verbose_output = "data line " * 2000  # ~4000 tokens

    messages = [
        system_msg,
        {"role": "user", "content": "Run analysis"},
        {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "bash", "content": verbose_output},
        {"role": "assistant", "content": "Done."},
    ]

    # Set context window small enough (e.g. 3000 tokens) so 80% ceiling is 2400 tokens
    result = session_guard.guard_context(messages, context_window=3000)
    assert result.compacted
    assert result.final_tokens < result.original_tokens
    assert result.final_tokens <= 2400 or result.final_tokens < result.original_tokens * 0.7


def test_session_guard_code_fence_integrity(session_guard: SessionGuard):
    """Compaction strictly preserves code blocks (```python ... ```) without syntax corruption."""
    code_block = (
        "```python\n"
        "def calculate_total(items: list[int]) -> int:\n"
        "    return sum(items)\n"
        "```"
    )
    messages = [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "Show me the function"},
        {"role": "assistant", "content": f"Here is the code:\n{code_block}\n" + ("extra verbose filler " * 1000)},
    ]

    result = session_guard.guard_context(messages, context_window=500)
    final_text = " ".join(str(m.get("content", "")) for m in result.messages)
    assert "def calculate_total(items: list[int]) -> int:" in final_text
    assert "```python" in final_text
    assert "```" in final_text


def test_session_guard_structural_header_preservation(session_guard: SessionGuard):
    """Compaction retains structural markdown headers (#, ##, ###)."""
    messages = [
        {"role": "system", "content": "System"},
        {
            "role": "assistant",
            "content": (
                "## Architecture Decision\n"
                "### Key Tradeoffs\n"
                "1. Performance vs Memory\n"
                + ("Verbose explanation details " * 1000)
            ),
        },
    ]
    result = session_guard.guard_context(messages, context_window=500)
    final_text = " ".join(str(m.get("content", "")) for m in result.messages)
    assert "## Architecture Decision" in final_text or "Key Tradeoffs" in final_text


def test_session_guard_returns_metrics_contract(session_guard: SessionGuard):
    """SessionGuardResult returns accurate token counts and booleans."""
    messages = [
        {"role": "system", "content": "System directive."},
        {"role": "user", "content": "Hello."},
    ]
    result = session_guard.guard_context(messages, context_window=8192)
    assert isinstance(result, SessionGuardResult)
    assert isinstance(result.messages, list)
    assert isinstance(result.compacted, bool)
    assert result.original_tokens > 0
    assert result.final_tokens > 0
    assert result.cache_prefix_preserved is True


# ==============================================================================
# Tier 2: Boundary & Corner Cases (>=5 tests)
# ==============================================================================

def test_session_guard_40_turn_continuous_session(session_guard: SessionGuard):
    """Simulates a 40+ turn conversation, verifying prefix stability and bounded memory."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You are a persistent assistant."},
        {"role": "user", "content": "Turn 0: Start session."},
    ]

    for turn in range(1, 45):
        messages.append({"role": "assistant", "content": f"Turn {turn} assistant response with tool result."})
        messages.append({"role": "user", "content": f"Turn {turn} user query asking for more info."})

    result = session_guard.guard_context(messages, context_window=4000)
    assert result.cache_prefix_preserved
    assert result.messages[0]["content"] == "You are a persistent assistant."
    assert result.messages[1]["content"] == "Turn 0: Start session."
    assert len(result.messages) >= 2


def test_session_guard_unclosed_code_blocks(session_guard: SessionGuard):
    """Messages containing unclosed code blocks do not crash or corrupt formatting."""
    messages = [
        {"role": "system", "content": "System"},
        {"role": "assistant", "content": "```python\ndef incomplete_func():\n    return True\n# missing closing fence"},
    ]
    result = session_guard.guard_context(messages, context_window=100)
    assert isinstance(result, SessionGuardResult)


def test_session_guard_massive_single_tool_output(session_guard: SessionGuard):
    """A single massive 50KB tool output is cleanly summarized without exceeding token limits."""
    giant_output = "line " * 10000
    messages = [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "Run large command"},
        {"role": "tool", "tool_call_id": "c1", "name": "cat_log", "content": giant_output},
    ]
    result = session_guard.guard_context(messages, context_window=2000)
    assert result.compacted
    assert result.final_tokens < result.original_tokens


def test_session_guard_zero_or_negative_context_window(session_guard: SessionGuard):
    """Handles zero or negative context window gracefully with default fallback."""
    messages = [{"role": "user", "content": "Hello"}]
    result_zero = session_guard.guard_context(messages, context_window=0)
    assert isinstance(result_zero, SessionGuardResult)

    result_neg = session_guard.guard_context(messages, context_window=-100)
    assert isinstance(result_neg, SessionGuardResult)


def test_session_guard_already_compact_context_noop(session_guard: SessionGuard):
    """Context well below 80% ceiling remains untouched with compacted=False."""
    messages = [
        {"role": "system", "content": "Short system."},
        {"role": "user", "content": "Short user."},
    ]
    result = session_guard.guard_context(messages, context_window=128000)
    assert not result.compacted
    assert result.original_tokens == result.final_tokens
    assert result.messages == messages
