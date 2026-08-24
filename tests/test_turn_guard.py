"""Comprehensive test suite for Turn Guard (0ms tool loop classifier & stagnation escalation).

Verifies:
- Direct bypass to active tier for in-flight tool loops (<2ms latency).
- Stagnation escalation on 3+ identical tool calls.
- Stagnation escalation on 2+ consecutive tool execution errors.
- Streak counters, error tracking, and multi-format support (OpenAI & Anthropic).
- Edge cases: empty messages, malformed payloads, alternating errors.
"""
from __future__ import annotations

import json
import time
from typing import Any
import pytest

try:
    from autoconduck.server.turn_guard import TurnAction, TurnClassificationResult, TurnGuard
except ImportError:
    pytest.skip("autoconduck.server.turn_guard not yet implemented in this milestone", allow_module_level=True)


@pytest.fixture
def turn_guard() -> TurnGuard:
    return TurnGuard()


# ==============================================================================
# Tier 1: Feature Coverage (>=5 tests)
# ==============================================================================

def test_turn_guard_new_user_message_routes_to_slm(turn_guard: TurnGuard):
    """Clean initial user prompt routes to SLM Planner."""
    messages = [
        {"role": "system", "content": "You are an expert coding assistant."},
        {"role": "user", "content": "Refactor the authentication module to use JWT tokens."},
    ]
    result: TurnClassificationResult = turn_guard.classify_turn(messages)
    assert not result.is_tool_loop
    assert not result.is_stagnant
    assert result.target_action == TurnAction.SLM_PLAN
    assert result.tool_call_streak == 0
    assert result.error_streak == 0


def test_turn_guard_active_tool_loop_bypasses_to_active_tier(turn_guard: TurnGuard):
    """In-flight tool response with valid output bypasses SLM directly to active tier."""
    messages = [
        {"role": "user", "content": "Find all usages of resolve_model"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "grep", "arguments": '{"query": "resolve_model"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "grep",
            "content": "autoconduck/orchestrator/runner.py:32: def run_dynamic_orchestration(...)",
        },
    ]
    result = turn_guard.classify_turn(messages)
    assert result.is_tool_loop
    assert not result.is_stagnant
    assert result.target_action == TurnAction.DIRECT_ACTIVE_TIER
    assert result.tool_call_streak >= 1
    assert result.last_tool_name == "grep"
    assert result.error_streak == 0


def test_turn_guard_stagnation_three_identical_calls_escalates(turn_guard: TurnGuard):
    """3 consecutive identical tool calls trigger stagnation escalation to SLM."""
    call_spec = {
        "id": "call_rep",
        "type": "function",
        "function": {"name": "read_file", "arguments": '{"path": "autoconduck/config.py"}'},
    }
    tool_resp = {
        "role": "tool",
        "tool_call_id": "call_rep",
        "name": "read_file",
        "content": "# config.py content...",
    }

    messages = [
        {"role": "user", "content": "Check config"},
        {"role": "assistant", "tool_calls": [call_spec]},
        tool_resp,
        {"role": "assistant", "tool_calls": [call_spec]},
        tool_resp,
        {"role": "assistant", "tool_calls": [call_spec]},
        tool_resp,
    ]
    result = turn_guard.classify_turn(messages)
    assert result.is_stagnant
    assert result.target_action == TurnAction.ESCALATE_SLM
    assert result.stagnation_reason is not None
    assert "identical" in result.stagnation_reason.lower() or "loop" in result.stagnation_reason.lower()


def test_turn_guard_stagnation_two_consecutive_errors_escalates(turn_guard: TurnGuard):
    """2 consecutive tool execution errors trigger stagnation escalation to SLM."""
    messages = [
        {"role": "user", "content": "Fix the build"},
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "call_e1", "type": "function", "function": {"name": "bash", "arguments": '{"command": "pytest"}'}}
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_e1",
            "name": "bash",
            "content": "Error: Command failed with exit code 1. ModuleNotFoundError: 'foo'",
        },
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "call_e2", "type": "function", "function": {"name": "bash", "arguments": '{"command": "pytest tests/"}'}}
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_e2",
            "name": "bash",
            "content": "FAILED tests/test_foo.py - ModuleNotFoundError: 'foo'",
        },
    ]
    result = turn_guard.classify_turn(messages)
    assert result.is_stagnant
    assert result.target_action == TurnAction.ESCALATE_SLM
    assert result.error_streak >= 2
    assert result.stagnation_reason is not None
    assert "error" in result.stagnation_reason.lower()


def test_turn_guard_streak_counters_reset_on_progress(turn_guard: TurnGuard):
    """A successful tool call resets error streaks and progress is tracked accurately."""
    messages = [
        {"role": "user", "content": "Execute task"},
        {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "bash", "content": "Error: command not found"},
        {"role": "assistant", "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}]},
        {"role": "tool", "tool_call_id": "c2", "name": "read_file", "content": "def a(): pass"},
    ]
    result = turn_guard.classify_turn(messages)
    assert not result.is_stagnant
    assert result.target_action == TurnAction.DIRECT_ACTIVE_TIER
    assert result.error_streak == 0
    assert result.last_tool_name == "read_file"


# ==============================================================================
# Tier 2: Boundary & Corner Cases (>=5 tests)
# ==============================================================================

def test_turn_guard_empty_messages_list(turn_guard: TurnGuard):
    """Empty message sequence returns safe SLM default without throwing."""
    result = turn_guard.classify_turn([])
    assert isinstance(result, TurnClassificationResult)
    assert not result.is_tool_loop
    assert result.target_action == TurnAction.SLM_PLAN


def test_turn_guard_malformed_tool_call_payload(turn_guard: TurnGuard):
    """Malformed or non-dict tool calls are handled safely without crashing."""
    messages = [
        {"role": "user", "content": "Run tool"},
        {"role": "assistant", "tool_calls": "invalid_string_not_list"},
        {"role": "tool", "content": "Some output"},
    ]
    result = turn_guard.classify_turn(messages)
    assert isinstance(result, TurnClassificationResult)
    assert result.target_action in (TurnAction.SLM_PLAN, TurnAction.DIRECT_ACTIVE_TIER)


def test_turn_guard_sub_2ms_execution_sla(turn_guard: TurnGuard):
    """Classification latency across 200 turns must be strictly sub-2ms per call."""
    messages = [
        {"role": "user", "content": "Grep files"},
        {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "grep", "arguments": '{"q": "x"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "grep", "content": "x = 10"},
    ]
    # Warmup
    turn_guard.classify_turn(messages)

    start = time.perf_counter()
    iterations = 200
    for _ in range(iterations):
        turn_guard.classify_turn(messages)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    avg_ms = elapsed_ms / iterations

    assert avg_ms < 2.0, f"Turn Guard latency {avg_ms:.3f}ms exceeds 2.0ms SLA!"


def test_turn_guard_mixed_success_and_error_sequence(turn_guard: TurnGuard):
    """Alternating error and success tool calls do NOT accumulate to 2 consecutive errors."""
    messages = [
        {"role": "user", "content": "Task"},
        {"role": "assistant", "tool_calls": [{"id": "1", "type": "function", "function": {"name": "t1", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "1", "name": "t1", "content": "Error: failed"},
        {"role": "assistant", "tool_calls": [{"id": "2", "type": "function", "function": {"name": "t2", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "2", "name": "t2", "content": "Success: completed step"},
        {"role": "assistant", "tool_calls": [{"id": "3", "type": "function", "function": {"name": "t3", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "3", "name": "t3", "content": "Error: transient glitch"},
    ]
    result = turn_guard.classify_turn(messages)
    assert not result.is_stagnant
    assert result.error_streak == 1
    assert result.target_action == TurnAction.DIRECT_ACTIVE_TIER


def test_turn_guard_anthropic_format_tool_use_and_tool_result(turn_guard: TurnGuard):
    """Supports Anthropic-style tool_use content blocks and tool_result blocks."""
    messages = [
        {"role": "user", "content": "Edit file"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I will read the file."},
                {"type": "tool_use", "id": "tu_1", "name": "read_file", "input": {"path": "main.py"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tu_1", "content": "print('hello')", "is_error": False}
            ],
        },
    ]
    result = turn_guard.classify_turn(messages)
    assert result.is_tool_loop
    assert not result.is_stagnant
    assert result.target_action == TurnAction.DIRECT_ACTIVE_TIER
    assert result.last_tool_name == "read_file"


def test_turn_guard_system_prompt_only(turn_guard: TurnGuard):
    """System-only prompt safely routes to SLM planner."""
    messages = [{"role": "system", "content": "You are an autonomous assistant."}]
    result = turn_guard.classify_turn(messages)
    assert not result.is_tool_loop
    assert result.target_action == TurnAction.SLM_PLAN
