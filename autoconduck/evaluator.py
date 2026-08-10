from dataclasses import dataclass
from typing import Literal
import re
from .semantic_router import RouteMatch

STACK_TRACE_BOOST = 0.25
ESCALATION_THRESHOLD = 0.80
HYSTERESIS_FLOOR = 0.50
_SYSTEM_REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>", re.IGNORECASE | re.DOTALL)


def clean_routing_text(text: object) -> str:
    """Remove Claude Code's injected reminders before measuring user intent."""
    return _SYSTEM_REMINDER.sub("", str(text or "")).strip()

@dataclass(frozen=True)
class Score:
    confidence_band: Literal["fast", "slow", "ambiguous"]
    path: Literal["fast", "slow"]
    confidence: float
    complexity: float
    reason: str

def has_stack_trace(text: str) -> bool:
    patterns = (r"Traceback \(most recent call last\)", r"File \"[^\"]+\", line \d+", r"\b(?:Error|Exception|CompilerError|TypeError|ValueError|SyntaxError):", r"\b\w+(?:\.\w+)*:\d+:\d+\b", r"\b(?:fatal|unhandled|undefined reference|build failed)\b")
    return any(re.search(p, text, re.I) for p in patterns)

def complexity_of(text: str, config=None) -> float:
    # All regex calls here operate on the same text string; caller should dedup/shared-cache
    # where possible (e.g., dispatcher.clean_routing_text already strips reminders before score).
    t = str(text or "")
    length = min(1.0, len(t) / 1200)
    refs = min(1.0, len(re.findall(r"@\w+|#\d+|`[^`]*`|https?://\S+", t)) / 3)
    structural = min(1.0, (len(re.findall(r"(?m)^\s*(?:[-*]|\d+[.)])\s+|```|^\s*##\s+", t)) + len(re.findall(r"refactor|migrate|redesign|architecture|feature|entire|whole|all files|multiple files|integration|codebase", t, re.I))) / 3)
    files = min(1.0, len(re.findall(r"[\w./\\-]+\.\w{1,4}\b", t)) / 3)
    hard = r"architecture|refactor|migrate|race condition|concurrency|security|distributed|optimize|algorithm"
    easy = r"typo|rename|format|comment|lint|simple"
    keyword_domain = (max(-3, min(3, len(re.findall(hard, t, re.I)) - len(re.findall(easy, t, re.I)))) + 3) / 6
    edits = bool(re.search(r"\b(?:fix|implement|add|refactor|write|build)\b", t, re.I))
    reads = bool(re.search(r"\b(?:explain|what|why|describe|review|summarize)\b", t, re.I))
    edit_intent = 1.0 if edits and not reads else 0.0 if reads and not edits else 0.5
    numbered = len(re.findall(r"(?m)^\s*\d+[.)]\s+", t))
    markers = len(re.findall(r"\b(?:then|next|after that|also|finally)\b", t, re.I)) + max(0, numbered - 1)
    multi_step = min(1.0, markers / 3)
    weights = getattr(getattr(config, "selection", None), "complexity_weights", None) or {"length": .15, "refs": .10, "structural": .25, "files": .10, "keyword_domain": .15, "edit_intent": .15, "multi_step": .10}
    value = sum(weights[k] * v for k, v in {"length": length, "refs": refs, "structural": structural, "files": files, "keyword_domain": keyword_domain, "edit_intent": edit_intent, "multi_step": multi_step}.items())
    return min(1.0, value + (STACK_TRACE_BOOST if has_stack_trace(t) else 0))

def is_tool_loop(messages: list) -> bool:
    """Return True if the message sequence represents an in-flight tool loop turn.

    A turn is an active tool loop if the latest active message in the sequence is a tool call
    or tool result, indicating the agent is currently executing tool actions. If the latest
    message is a new user prompt, it is evaluated for task complexity instead of being forced
    to the fast path.
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
    if role in ("tool", "function"):
        return True
    if "tool_calls" in last_msg or "function_call" in last_msg:
        return True
    content = last_msg.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") in ("tool_use", "tool_result"):
                return True
    return False

def _last(messages: list) -> str:
    if not isinstance(messages, list) or not messages:
        return ""
    for item in reversed(messages):
        if not isinstance(item, dict) or item.get("role", "user") == "user":
            content = item.get("content", "") if isinstance(item, dict) else getattr(item, "content", item)
            if "<system-reminder>" not in content:
                return str(content or "")
            return clean_routing_text(content)
    return ""

def score(messages: list, history, match: RouteMatch, pseudo_model: str = "autoconduck", config=None) -> Score:
    # Hoist all getattr(cfg, ...) calls to top — avoids repeated attr lookups inside conditionals.
    cfg = config
    low = float(getattr(cfg, "ambiguous_low", 0.55) if cfg else 0.55)
    high = float(getattr(cfg, "ambiguous_high", 0.70) if cfg else 0.70)
    stack_trace_boost = float(getattr(cfg, "stack_trace_boost", STACK_TRACE_BOOST) if cfg else STACK_TRACE_BOOST)
    hysteresis_floor = float(getattr(cfg, "hysteresis_floor", HYSTERESIS_FLOOR) if cfg else HYSTERESIS_FLOOR)

    text = _last(messages)
    complexity = complexity_of(text, cfg)
    trace = has_stack_trace(text)
    confidence = min(1.0, max(float(match.confidence), complexity * 0.75) + (stack_trace_boost if trace else 0))

    # Active tool loops in CLI agents (Claude Code, OpenCode, Pi) must always stay on the fast path
    if is_tool_loop(messages):
        return Score("fast", "fast", confidence, complexity, "interactive agent tool loop")

    previous = history[-1] if isinstance(history, list) and history else history
    escalated = bool(getattr(previous, "complexity", 0) >= ESCALATION_THRESHOLD or (isinstance(previous, dict) and (previous.get("complexity", 0) >= ESCALATION_THRESHOLD or previous.get("confidence", 0) >= ESCALATION_THRESHOLD)))
    if escalated and not trace:
        complexity = min(complexity, hysteresis_floor)
    multiplier = 1.0
    if pseudo_model.endswith("budget"):
        multiplier = 1.15
    elif pseudo_model.endswith("expensive"):
        multiplier = 0.85
    boundary_low, boundary_high = min(1.0, low * multiplier), min(1.0, high * multiplier)
    if confidence < boundary_low or (boundary_low <= confidence <= boundary_high):
        return Score("ambiguous", "fast", confidence, complexity, "confidence is in the ambiguous zone")
    sel = getattr(cfg, "selection", cfg)
    slow_threshold = float(getattr(sel, "slow_threshold", 0.75) if sel else 0.75)
    slow = complexity >= slow_threshold or (match.route == "slow_path" and confidence >= boundary_high)
    return Score("slow" if slow else "fast", "slow" if slow else "fast", confidence, complexity, "stack trace boost" if trace else "semantic route and complexity")
