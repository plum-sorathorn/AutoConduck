from dataclasses import dataclass
from typing import Literal

from .semantic_router import RouteMatch
from .complexity import (complexity_of, clean_routing_text, has_stack_trace,
    has_escalation_signal, _context_boost, _first_user_complexity, _routing_text,
    _last, _intent_drift)

STACK_TRACE_BOOST = 0.25
ESCALATION_THRESHOLD = 0.80
HYSTERESIS_FLOOR = 0.50

def _first_user_complexity_text(messages: list) -> str:
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role", "user") == "user":
            content = str(msg.get("content", ""))
            if "<system-reminder>" not in content and content.strip():
                return clean_routing_text(content)
    return ""

@dataclass(frozen=True)
class Score:
    confidence_band: Literal["fast", "slow", "ambiguous"]
    path: Literal["fast", "slow"]
    confidence: float
    complexity: float
    reason: str

def is_tool_loop(messages: list, config=None) -> bool:
    """Return True if the message sequence is an in-flight tool loop turn.

    A turn is an active tool loop if the latest non-system message is a tool
    call or tool result — the agent is mid-execution and re-routing would break
    the tool contract.

    ESCALATION EXCEPTIONS (return False, allow full scoring):
      - Explicit escalation signal in the tool result.
      - Stack trace / fatal error in the tool result.
      - Tool chain length > 12 (long-running tool loops may need more capability).
      - First user message complexity >= slow_threshold (the original task was
        complex — don't suppress slow-path routing mid-flight).
    """
    if not isinstance(messages, list) or not messages:
        return False

    last_msg = None
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            last_msg = msg
            break
        role = msg.get("role", "user")
        content = str(msg.get("content", ""))
        if role != "system" and "<system-reminder>" not in content:
            last_msg = msg
            break

    if not last_msg or not isinstance(last_msg, dict):
        return False

    role = last_msg.get("role", "user")
    is_active_tool = (
        role in ("tool", "function")
        or "tool_calls" in last_msg
        or "function_call" in last_msg
    )
    if not is_active_tool:
        content = last_msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") in (
                    "tool_use",
                    "tool_result",
                ):
                    is_active_tool = True
                    break

    if not is_active_tool:
        return False

    # Check for in-flight escalation exceptions
    last_text = str(last_msg.get("content", ""))
    if has_escalation_signal(last_text) or has_stack_trace(last_text):
        return False

    # Long tool chain soft-escalation: if >12 tool turns have fired, allow re-scoring
    tool_turn_count = sum(
        1 for m in messages
        if isinstance(m, dict) and (
            m.get("role") in ("tool", "function") or "tool_calls" in m
        )
    )
    if tool_turn_count > 12:
        return False

    cfg_sel = getattr(config, "selection", config) if config else None
    slow_threshold = float(getattr(cfg_sel, "slow_threshold", 0.75) if cfg_sel else 0.75)
    first_complexity = complexity_of(_first_user_complexity_text(messages), config)
    if first_complexity >= slow_threshold:
        # Only bypass tool-loop suppression when the *current* message also
        # carries an escalation or error signal.  Without this guard every
        # tool-loop turn in a complex session got re-scored as SLOW, triggering
        # the full orchestrator pipeline on every agent tool result.
        if has_escalation_signal(last_text) or has_stack_trace(last_text):
            return False
        # Original task was complex but current turn looks clean — keep fast.

    return True

def score(
    messages: list,
    history,
    match: RouteMatch,
    pseudo_model: str = "autoconduck",
    config=None,
) -> Score:
    cfg = config
    low = float(getattr(cfg, "ambiguous_low", 0.55) if cfg else 0.55)
    high = float(getattr(cfg, "ambiguous_high", 0.70) if cfg else 0.70)
    stack_trace_boost = float(
        getattr(cfg, "stack_trace_boost", STACK_TRACE_BOOST) if cfg else STACK_TRACE_BOOST
    )
    hysteresis_floor = float(
        getattr(cfg, "hysteresis_floor", HYSTERESIS_FLOOR) if cfg else HYSTERESIS_FLOOR
    )

    text = _routing_text(messages)
    complexity = complexity_of(text, cfg)
    trace = has_stack_trace(text)
    escalation = has_escalation_signal(text)

    # Active tool loops stay on the fast path UNLESS an escalation or stack trace
    # trigger fired (handled inside is_tool_loop) or the tool chain is very long.
    if is_tool_loop(messages, config=cfg):
        ctx = _context_boost(messages)
        first_comp = _first_user_complexity(messages, cfg)
        eff_complexity = min(1.0, max(complexity, first_comp) + ctx)
        return Score("fast", "fast", 0.0, eff_complexity, "interactive agent tool loop")

    # Apply context-aware boost (conversation depth, tool chain, intent drift)
    ctx = _context_boost(messages)
    complexity = min(1.0, complexity + ctx)

    confidence = min(
        1.0,
        max(float(match.confidence), complexity * 0.75)
        + (stack_trace_boost if trace else 0)
        + (0.30 if escalation else 0),
    )

    if trace or escalation:
        return Score(
            "slow",
            "slow",
            confidence,
            max(complexity, 0.85 if escalation else complexity),
            "agent complexity escalation" if escalation else "stack trace boost",
        )

    deescalation_threshold = float(
        getattr(getattr(cfg, "selection", None), "deescalation_threshold", 0.40) if cfg else 0.40
    )

    previous = history[-1] if isinstance(history, list) and history else history
    escalated = bool(
        getattr(previous, "complexity", 0) >= ESCALATION_THRESHOLD
        or (
            isinstance(previous, dict)
            and (
                previous.get("complexity", 0) >= ESCALATION_THRESHOLD
                or previous.get("confidence", 0) >= ESCALATION_THRESHOLD
            )
        )
    )
    # DE-ESCALATION: If session was escalated but current turn is simple (< deescalation_threshold)
    # and carries no stack trace or escalation signals, active de-escalate back to fast path.
    if escalated and complexity < deescalation_threshold and not trace and not escalation:
        return Score("fast", "fast", confidence, complexity, "de-escalated to fast path")

    if escalated:
        complexity = min(complexity, hysteresis_floor)

    multiplier = 1.0
    if pseudo_model.endswith("budget"):
        multiplier = 1.15
    elif pseudo_model.endswith("expensive"):
        multiplier = 0.85

    boundary_low, boundary_high = min(1.0, low * multiplier), min(
        1.0, high * multiplier
    )

    if boundary_low <= confidence <= boundary_high:
        return Score(
            "ambiguous",
            "fast",
            confidence,
            complexity,
            "confidence is in the ambiguous zone",
        )

    # confidence < boundary_low → message is clearly simple, fast-path it
    if confidence < boundary_low:
        return Score("fast", "fast", confidence, complexity, "below ambiguous floor")

    sel = getattr(cfg, "selection", cfg)
    slow_threshold = float(getattr(sel, "slow_threshold", 0.75) if sel else 0.75)
    slow = complexity >= slow_threshold or (
        match.route == "slow_path" and confidence >= boundary_high
    )

    return Score(
        "slow" if slow else "fast",
        "slow" if slow else "fast",
        confidence,
        complexity,
        "semantic route and complexity",
    )
