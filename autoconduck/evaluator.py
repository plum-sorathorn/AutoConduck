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

def complexity_of(text: str) -> float:
    # All regex calls here operate on the same text string; caller should dedup/shared-cache
    # where possible (e.g., dispatcher.clean_routing_text already strips reminders before score).
    t = str(text or "")
    length = min(len(t) / 500.0, 1.0)
    identifiers = len(set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", t)))
    refs = min(identifiers / 25.0, 1.0)
    structural = len(re.findall(r"refactor|migrate|redesign|architecture|feature|entire|whole|all files|multiple files|integration|codebase", t, re.I))
    files = len(re.findall(r"\b\S+\.(?:py|js|ts|go|rs|java|sql|yaml|json)\b", t, re.I))
    return min(1.0, 0.20 * length + 0.15 * refs + 0.50 * min(structural / 3, 1) + 0.15 * min(files / 3, 1))

def _last(messages: list) -> str:
    if not messages: return ""
    item = messages[-1]
    content = item.get("content", "") if isinstance(item, dict) else getattr(item, "content", item)
    # Dispatcher already calls clean_routing_text on the text before scoring; skip re-clean unless present.
    if "<system-reminder>" not in content:
        return str(content or "")
    return clean_routing_text(content)

def score(messages: list, history, match: RouteMatch, pseudo_model: str = "autoconduck", config=None) -> Score:
    # Hoist all getattr(cfg, ...) calls to top — avoids repeated attr lookups inside conditionals.
    cfg = config
    low = float(getattr(cfg, "ambiguous_low", 0.55) if cfg else 0.55)
    high = float(getattr(cfg, "ambiguous_high", 0.70) if cfg else 0.70)
    stack_trace_boost = float(getattr(cfg, "stack_trace_boost", STACK_TRACE_BOOST) if cfg else STACK_TRACE_BOOST)
    hysteresis_floor = float(getattr(cfg, "hysteresis_floor", HYSTERESIS_FLOOR) if cfg else HYSTERESIS_FLOOR)

    text = _last(messages)
    complexity = complexity_of(text)
    trace = has_stack_trace(text)
    confidence = min(1.0, max(float(match.confidence), complexity * 0.75) + (stack_trace_boost if trace else 0))
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
    slow = complexity >= 0.6 or (match.route == "slow_path" and confidence >= boundary_high)
    return Score("slow" if slow else "fast", "slow" if slow else "fast", confidence, complexity, "stack trace boost" if trace else "semantic route and complexity")
