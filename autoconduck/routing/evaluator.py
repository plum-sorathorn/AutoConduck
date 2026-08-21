from dataclasses import dataclass
from typing import Literal

from .complexity import (
    complexity_of,
    clean_routing_text,
    has_stack_trace,
    has_escalation_signal,
    _context_boost,
    _first_user_complexity,
    _routing_text,
    _intent_drift,
)
from .semantic_router import RouteMatch

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



def has_stagnant_tool_loop(messages: list) -> bool:
    """Detect if the agent is caught in a stagnant or repetitive tool loop.

    Inspired by NVIDIA Switchyard's escalation router:
    1. Tool call repetition: 3+ consecutive tool calls with identical tool name and arguments.
    2. Consecutive tool errors: 2+ consecutive tool results with failure/error signatures.
    """
    if not isinstance(messages, list) or len(messages) < 2:
        return False

    # Find recent active turn messages after the latest genuine user prompt
    last_user_idx = -1
    for i, m in enumerate(messages):
        if isinstance(m, dict) and m.get("role") == "user" and "<system-reminder>" not in str(m.get("content", "")):
            c = m.get("content")
            if isinstance(c, list) and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in c):
                continue
            last_user_idx = i

    active = messages[last_user_idx + 1:] if last_user_idx >= 0 else messages[-8:]

    # 1. Check for repetitive tool call signatures
    tool_calls: list[tuple[str, str]] = []
    for msg in active:
        if not isinstance(msg, dict):
            continue
        tcs = msg.get("tool_calls") or msg.get("toolCalls")
        if isinstance(tcs, list):
            for tc in tcs:
                if isinstance(tc, dict):
                    fn = tc.get("function", {}) if isinstance(tc.get("function"), dict) else tc
                    name = str(fn.get("name") or tc.get("name") or "").strip().lower()
                    args = str(fn.get("arguments") or tc.get("arguments") or "").strip()
                    if name:
                        tool_calls.append((name, args))
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    name = str(block.get("name") or "").strip().lower()
                    inp = str(block.get("input") or "").strip()
                    if name:
                        tool_calls.append((name, inp))

    if len(tool_calls) >= 3:
        last3 = tool_calls[-3:]
        if last3[0] == last3[1] == last3[2]:
            return True

    # 2. Check for consecutive tool result errors/failures
    tool_results: list[str] = []
    for msg in active:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role in ("tool", "toolResult", "function"):
            tool_results.append(str(msg.get("content", "")))
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tool_results.append(str(block.get("content") or block.get("text") or ""))

    if len(tool_results) >= 2:
        error_keywords = (
            "traceback (most recent call last)",
            "syntaxerror:",
            "typeerror:",
            "valueerror:",
            "filenotfounderror:",
            "failed with exit code",
            "command not found",
            "permission denied",
            "patch failed",
            "patcherror:",
            "diff failed",
            "error:",
        )
        recent_errs = [
            any(kw in res.lower() for kw in error_keywords)
            for res in tool_results[-2:]
        ]
        if all(recent_errs):
            return True

    return False


def is_tool_loop(messages: list, config=None) -> bool:
    """Return True if the message sequence is an in-flight tool loop turn.

    A turn is an active tool loop if the latest non-system message is a tool
    call or tool result — the agent is mid-execution and re-routing would break
    the tool contract.

    ESCALATION EXCEPTIONS (return False, allow full scoring):
      - Explicit escalation signal in the tool result.
      - Stack trace / fatal error in the tool result.
      - Stagnant/repetitive tool loop or consecutive tool errors (Switchyard escalation).
      - Tool chain length > 12 (long-running tool loops may need more capability).
      - First user message complexity >= slow_threshold (the original task was
        complex — don't suppress slow-path routing mid-flight).
    """
    if not isinstance(messages, list) or not messages:
        return False

    recent = [
        msg for msg in messages
        if isinstance(msg, dict)
        and msg.get("role", "user") != "system"
        and "<system-reminder>" not in str(msg.get("content", ""))
    ][-4:]
    if not recent:
        return False

    signals: list[tuple[int, dict]] = []
    for index, msg in enumerate(recent):
        content = msg.get("content")
        block_signal = isinstance(content, list) and any(
            isinstance(block, dict)
            and block.get("type") in ("tool_use", "tool_result")
            for block in content
        )
        if (
            msg.get("role") in ("tool", "function", "toolResult")
            or "tool_calls" in msg
            or "function_call" in msg
            or block_signal
        ):
            signals.append((index, msg))

    # A pi tool result may be a user-role text message immediately following
    # the assistant's tool_calls message.  Do not require the final message to
    # carry the tool metadata in that representation.
    active: list[tuple[int, dict]] = []
    for index, msg in signals:
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            if index + 1 < len(recent) and recent[index + 1].get("role") in ("user", "toolResult"):
                active.append((index, msg))
            elif index == len(recent) - 1:
                active.append((index, msg))
        else:
            active.append((index, msg))
    if not active:
        return False

    # A later, unrelated user turn starts a new turn; historical tool calls
    # alone must not suppress its normal complexity evaluation.
    last_index = len(recent) - 1
    signal_index, signal_msg = active[-1]
    if recent[last_index].get("role") == "user" and not (
        signal_index + 1 == last_index
        and (
            signal_msg.get("role") == "assistant"
            and "tool_calls" in signal_msg
            or recent[last_index].get("content")
            and isinstance(recent[last_index].get("content"), list)
        )
    ):
        return False

    adjacent = recent[max(0, signal_index - 1): min(len(recent), signal_index + 3)]
    adjacent_text = "\n".join(str(item.get("content", "")) for item in adjacent)
    if signal_index + 1 < len(recent):
        adjacent_text += "\n" + str(recent[signal_index + 1].get("content", ""))

    if (
        has_escalation_signal(adjacent_text)
        or has_stack_trace(adjacent_text)
        or has_stagnant_tool_loop(messages)
    ):
        return False

    # Long tool chain soft-escalation: if >12 tool turns have fired in the *active* chain, allow re-scoring.
    # Find the latest user turn so cumulative history from past turns doesn't poison future tool loops.
    last_user_idx = -1
    for i, m in enumerate(messages):
        if isinstance(m, dict) and m.get("role") == "user" and "<system-reminder>" not in str(m.get("content", "")):
            c = m.get("content")
            if isinstance(c, list) and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in c):
                continue
            last_user_idx = i

    active_messages = messages[last_user_idx + 1:] if last_user_idx >= 0 else messages
    tool_turn_count = sum(
        1
        for m in active_messages
        if isinstance(m, dict)
        and (m.get("role") in ("tool", "function", "toolResult") or "tool_calls" in m)
    )
    if tool_turn_count > 12:
        return False

    cfg_sel = getattr(config, "selection", config) if config else None
    slow_threshold = float(
        getattr(cfg_sel, "slow_threshold", 0.75) if cfg_sel else 0.75
    )
    first_complexity = complexity_of(_first_user_complexity_text(messages), config)
    if first_complexity >= slow_threshold:
        # Only bypass tool-loop suppression when the *current* message also
        # carries an escalation or error signal.  Without this guard every
        # tool-loop turn in a complex session got re-scored as SLOW.
        if has_escalation_signal(adjacent_text) or has_stack_trace(adjacent_text):
            return False

    return True


_VERIFY_TOOLS = (
    "test",
    "verify",
    "check",
    "lint",
    "validate",
    "eval",
    "assert",
    "inspect",
    "jest",
    "pytest",
    "vitest",
    "mypy",
    "ruff",
    "pyright",
    "tsc",
    "flake8",
    "unittest",
    "ctest",
)
_RECON_TOOLS = ("read", "view", "grep", "find", "glob", "list", "search", "fetch", "cat")
_EDIT_TOOLS = ("edit", "write", "patch", "diff", "replace")
_BASH_TOOLS = ("bash", "exec", "terminal", "command", "cmd", "sh", "run_command")
_VERIFY_PAYLOAD_SIGNALS = (
    "pytest",
    "npm test",
    "pnpm test",
    "yarn test",
    "cargo test",
    "go test",
    "vitest",
    "jest",
    "ruff",
    "mypy",
    "flake8",
    "pyright",
    "tsc",
    "unittest",
    "ctest",
    "tox",
    "lint",
    "test ",
)


def _classify_tool_or_command(tool_name: str, payload: str = "") -> str:
    tn = (tool_name or "").lower()
    pl = (payload or "").lower()

    if any(v in tn for v in _VERIFY_TOOLS):
        return "verify"
    if any(r in tn for r in _RECON_TOOLS):
        return "recon"
    if any(e in tn for e in _EDIT_TOOLS):
        return "edit"
    if any(b in tn for b in _BASH_TOOLS):
        if any(vp in pl for vp in _VERIFY_PAYLOAD_SIGNALS):
            return "verify"
        return "bash"
    if pl and any(vp in pl for vp in _VERIFY_PAYLOAD_SIGNALS):
        return "verify"
    return "recon"


def detect_turn_task(messages: list) -> str | None:
    """Detect the operational task of the latest turn in O(1) time.

    Returns:
      'recon'  - reading files, grepping, searching, listing directories
      'edit'   - modifying files, applying diffs, writing code
      'verify' - running test suites, linters, validation checks
      'bash'   - executing general shell commands or scripts
      'chat'   - conversational interaction or direct user query
    """
    if not isinstance(messages, list) or not messages:
        return None

    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role in ("tool", "toolResult", "function"):
            tool_name = str(msg.get("toolName") or msg.get("name") or "").lower()
            content = str(msg.get("content") or "")
            return _classify_tool_or_command(tool_name, content)

        # Check content blocks for Anthropic style tool_result / tool_use
        content = msg.get("content")
        if isinstance(content, list):
            for block in reversed(content):
                if isinstance(block, dict):
                    btype = block.get("type")
                    if btype == "tool_result":
                        tool_name = str(block.get("name") or block.get("tool_name") or "").lower()
                        bcontent = str(block.get("content") or "")
                        return _classify_tool_or_command(tool_name, bcontent)
                    if btype == "tool_use":
                        tool_name = str(block.get("name") or "").lower()
                        input_payload = str(block.get("input") or "")
                        return _classify_tool_or_command(tool_name, input_payload)

        # Check tool_calls in assistant message
        tool_calls = msg.get("tool_calls") or msg.get("toolCalls")
        if tool_calls and isinstance(tool_calls, list):
            for tc in reversed(tool_calls):
                fn_name = ""
                args = ""
                if isinstance(tc, dict):
                    fn_name = str(tc.get("function", {}).get("name") or tc.get("name") or "").lower()
                    args = str(tc.get("function", {}).get("arguments") or tc.get("arguments") or "")
                if fn_name:
                    return _classify_tool_or_command(fn_name, args)
        if role == "user":
            break
    return None


def score(
    messages: list,
    history,
    match: RouteMatch,
    pseudo_model: str = "autoconduck",
    config=None,
) -> Score:
    cfg = config
    low = float(getattr(cfg, "ambiguous_low", 0.60) if cfg else 0.60)
    high = float(getattr(cfg, "ambiguous_high", 0.75) if cfg else 0.75)
    stack_trace_boost = float(
        getattr(cfg, "stack_trace_boost", STACK_TRACE_BOOST)
        if cfg
        else STACK_TRACE_BOOST
    )
    hysteresis_floor = float(
        getattr(cfg, "hysteresis_floor", HYSTERESIS_FLOOR) if cfg else HYSTERESIS_FLOOR
    )

    text = _routing_text(messages)
    complexity = complexity_of(text, cfg)
    trace = has_stack_trace(text)
    escalation = has_escalation_signal(text)
    stagnant = has_stagnant_tool_loop(messages)

    # Active tool loops stay on the fast path UNLESS an escalation, stack trace,
    # or stagnation trigger fired (handled inside is_tool_loop) or the tool chain is very long.
    if is_tool_loop(messages, config=cfg):
        ctx = _context_boost(messages, config=cfg)
        first_comp = _first_user_complexity(messages, cfg)
        eff_complexity = min(1.0, max(complexity, first_comp) + ctx)
        return Score("fast", "fast", 0.0, eff_complexity, "interactive agent tool loop")

    # Apply context-aware boost (conversation depth, tool chain, intent drift)
    ctx = _context_boost(messages, config=cfg)
    complexity = min(1.0, complexity + ctx)

    confidence = min(
        1.0,
        max(float(match.confidence), complexity * 0.75)
        + (stack_trace_boost if trace else 0)
        + (0.30 if (escalation or stagnant) else 0),
    )

    if trace or escalation or stagnant:
        reason = (
            "stagnant tool loop escalation"
            if stagnant
            else ("agent complexity escalation" if escalation else "stack trace boost")
        )
        return Score(
            "slow",
            "slow",
            confidence,
            max(complexity, 0.85 if (escalation or stagnant) else complexity),
            reason,
        )

    deescalation_threshold = float(
        getattr(getattr(cfg, "selection", None), "deescalation_threshold", 0.40)
        if cfg
        else 0.40
    )

    sel = getattr(cfg, "selection", cfg) if cfg else None
    window_size = int(getattr(sel, "hysteresis_window_size", 5) if sel else 5)
    decay = float(getattr(sel, "hysteresis_decay", 0.85) if sel else 0.85)

    recent_history = (
        history[-window_size:]
        if isinstance(history, list)
        else ([history] if history else [])
    )
    decayed_max_complexity = 0.0
    for idx, item in enumerate(reversed(recent_history)):
        c = (
            item.get("complexity", 0.0)
            if isinstance(item, dict)
            else getattr(item, "complexity", 0.0)
        )
        conf = (
            item.get("confidence", 0.0)
            if isinstance(item, dict)
            else getattr(item, "confidence", 0.0)
        )
        effective_item_comp = max(
            float(c or 0.0),
            float(conf or 0.0) if float(conf or 0.0) >= ESCALATION_THRESHOLD else 0.0,
        )
        decayed_val = effective_item_comp * (decay ** idx)
        if decayed_val > decayed_max_complexity:
            decayed_max_complexity = decayed_val

    escalated = decayed_max_complexity >= ESCALATION_THRESHOLD

    # DE-ESCALATION: If session was escalated but current turn is simple (< deescalation_threshold)
    # and carries no stack trace or escalation signals, active de-escalate back to fast path.
    if (
        escalated
        and complexity < deescalation_threshold
        and not trace
        and not escalation
    ):
        return Score(
            "fast", "fast", confidence, complexity, "de-escalated to fast path"
        )

    if escalated:
        complexity = min(complexity, hysteresis_floor)

    slow_threshold = float(getattr(sel, "slow_threshold", 0.75) if sel else 0.75)
    if complexity >= slow_threshold:
        return Score("slow", "slow", confidence, complexity, "complexity threshold")

    multiplier = 1.0
    if pseudo_model.endswith("budget"):
        multiplier = 1.15
    elif pseudo_model.endswith("expensive"):
        multiplier = 0.85

    boundary_low, boundary_high = (
        min(1.0, low * multiplier),
        min(1.0, high * multiplier),
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
