"""Turn Guard: Synchronous 0ms tool loop classifier & stagnation escalation.

Classifies incoming request turns in <2ms:
- Routes active in-flight tool loops directly to the active model tier (DIRECT_ACTIVE_TIER).
- Escalates to the SLM Planner upon stagnation (3+ identical calls or 2+ consecutive errors) (ESCALATE_SLM).
- Routes clean user turns to the SLM Planner (SLM_PLAN).
- Supports both OpenAI (tool_calls, role=tool) and Anthropic (tool_use, tool_result) formats.
"""
from __future__ import annotations

import json
import logging
from enum import Enum
from typing import Any
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class TurnAction(str, Enum):
    DIRECT_ACTIVE_TIER = "direct_active_tier"
    SLM_PLAN = "slm_plan"
    ESCALATE_SLM = "escalate_slm"


class TurnClassificationResult(BaseModel):
    is_tool_loop: bool
    is_stagnant: bool
    stagnation_reason: str | None = None
    target_action: TurnAction
    tool_call_streak: int = 0
    error_streak: int = 0
    last_tool_name: str | None = None


def _is_tool_error_content(content: Any, is_error_flag: bool | None = None) -> bool:
    """Check if tool execution output signifies an error."""
    if is_error_flag is True:
        return True
    if not isinstance(content, str):
        return False
    lower = content.lower()
    error_markers = [
        "error:",
        "error :",
        "failed with exit code",
        "failed tests",
        "modulenotfounderror",
        "command not found",
        "traceback (most recent call last)",
        "syntaxerror",
        "filenotfounderror",
        "permissionerror",
        "failed",
    ]
    if lower.startswith("error") or lower.startswith("failed"):
        return True
    return any(marker in lower for marker in error_markers)


class TurnGuard:
    """Synchronous fast-path classifier for agentic tool loops and stagnation."""

    def classify_turn(self, messages: list[dict[str, Any]]) -> TurnClassificationResult:
        """Classify message history for tool loop bypass or SLM escalation."""
        if not messages or not isinstance(messages, list):
            return TurnClassificationResult(
                is_tool_loop=False,
                is_stagnant=False,
                target_action=TurnAction.SLM_PLAN,
                tool_call_streak=0,
                error_streak=0,
            )

        # Inspect the last message
        last_msg = messages[-1]
        if not isinstance(last_msg, dict):
            return TurnClassificationResult(
                is_tool_loop=False,
                is_stagnant=False,
                target_action=TurnAction.SLM_PLAN,
                tool_call_streak=0,
                error_streak=0,
            )

        # Check if last message is a tool response
        is_tool_resp = False
        last_tool_name: str | None = None
        last_tool_is_error = False

        role = last_msg.get("role")
        content = last_msg.get("content")

        if role in ("tool", "function"):
            is_tool_resp = True
            last_tool_name = last_msg.get("name")
            last_tool_is_error = _is_tool_error_content(content)
        elif role == "user" and isinstance(content, list):
            # Anthropic tool_result check
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    is_tool_resp = True
                    is_err = block.get("is_error", False)
                    last_tool_is_error = is_err or _is_tool_error_content(block.get("content"))
                    break

        if not is_tool_resp:
            # Clean user prompt or system turn -> SLM Planner
            return TurnClassificationResult(
                is_tool_loop=False,
                is_stagnant=False,
                target_action=TurnAction.SLM_PLAN,
                tool_call_streak=0,
                error_streak=0,
                last_tool_name=None,
            )

        # If last_tool_name was not in the tool message, extract from previous assistant call
        all_calls: list[tuple[str, str]] = []  # list of (tool_name, args_str)
        tool_results: list[bool] = []  # list of is_error booleans

        for idx, m in enumerate(messages):
            if not isinstance(m, dict):
                continue
            m_role = m.get("role")
            m_content = m.get("content")

            # OpenAI assistant tool calls
            tool_calls = m.get("tool_calls")
            if isinstance(tool_calls, list):
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        fn = tc.get("function")
                        if isinstance(fn, dict):
                            fn_name = str(fn.get("name", ""))
                            fn_args = str(fn.get("arguments", ""))
                            all_calls.append((fn_name, fn_args))
                            last_tool_name = fn_name

            # Anthropic assistant tool_use
            if m_role == "assistant" and isinstance(m_content, list):
                for b in m_content:
                    if isinstance(b, dict) and b.get("type") == "tool_use":
                        fn_name = str(b.get("name", ""))
                        fn_args = json.dumps(b.get("input", {}), sort_keys=True)
                        all_calls.append((fn_name, fn_args))
                        last_tool_name = fn_name

            # OpenAI tool response
            if m_role in ("tool", "function"):
                is_err = _is_tool_error_content(m.get("content"))
                tool_results.append(is_err)
                if not last_tool_name and m.get("name"):
                    last_tool_name = m.get("name")

            # Anthropic user tool_result
            if m_role == "user" and isinstance(m_content, list):
                for b in m_content:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        is_err = b.get("is_error", False) or _is_tool_error_content(b.get("content"))
                        tool_results.append(is_err)

        # Calculate error streak (consecutive errors at the end)
        error_streak = 0
        for is_err in reversed(tool_results):
            if is_err:
                error_streak += 1
            else:
                break

        # Calculate identical tool call streak
        tool_call_streak = 1
        if len(all_calls) >= 2:
            last_call = all_calls[-1]
            identical_count = 1
            for c in reversed(all_calls[:-1]):
                if c == last_call:
                    identical_count += 1
                else:
                    break
            tool_call_streak = identical_count

        # Check Stagnation conditions
        # 1. 3+ identical consecutive tool calls
        if tool_call_streak >= 3:
            return TurnClassificationResult(
                is_tool_loop=True,
                is_stagnant=True,
                stagnation_reason=f"3+ identical tool calls detected (loop on {last_tool_name})",
                target_action=TurnAction.ESCALATE_SLM,
                tool_call_streak=tool_call_streak,
                error_streak=error_streak,
                last_tool_name=last_tool_name,
            )

        # 2. 2+ consecutive errors
        if error_streak >= 2:
            return TurnClassificationResult(
                is_tool_loop=True,
                is_stagnant=True,
                stagnation_reason=f"2+ consecutive tool execution errors on {last_tool_name}",
                target_action=TurnAction.ESCALATE_SLM,
                tool_call_streak=tool_call_streak,
                error_streak=error_streak,
                last_tool_name=last_tool_name,
            )

        # Active healthy tool loop
        return TurnClassificationResult(
            is_tool_loop=True,
            is_stagnant=False,
            target_action=TurnAction.DIRECT_ACTIVE_TIER,
            tool_call_streak=tool_call_streak,
            error_streak=error_streak,
            last_tool_name=last_tool_name,
        )
