from __future__ import annotations

import re
import time
from typing import Any, Callable, Optional

from pydantic import BaseModel

REGEX_FAST = re.compile(r"^(fix|format|typo|rename|docstring|check syntax|where is|grep)\b", re.I)
SLOW_KEYWORDS = [
    "refactor application",
    "build feature",
    "architecture",
    "backtesting",
    "migrate",
    "rewrite entire",
    "monorepo",
    "codebase-wide",
]
AMBIGUOUS_LOW, AMBIGUOUS_HIGH = 0.40, 0.55
FAST_PROMPT_MAX_LEN = 120


class Decision(BaseModel):
    path: str  # fast | slow | ambiguous
    reason: str
    T_i: float | None = None
    elapsed_ms: float = 0.0
    tier_hint: str | None = None


def _last_user_text(messages: list[Any]) -> str:
    if not messages:
        return ""
    # find last user message; fallback to last message
    for m in reversed(messages):
        role = getattr(m, "role", None)
        if isinstance(m, dict):
            role = m.get("role")
        if role == "user":
            c = getattr(m, "content", None)
            if isinstance(m, dict):
                c = m.get("content", "")
            if isinstance(c, list):
                return " ".join(str(x.get("text", x) if isinstance(x, dict) else str(x)) for x in c)
            return str(c or "")
    # no user found -> last
    m = messages[-1]
    c = getattr(m, "content", None)
    if isinstance(m, dict):
        c = m.get("content", "")
    if isinstance(c, list):
        return " ".join(str(x.get("text", x) if isinstance(x, dict) else str(x)) for x in c)
    return str(c or "")


def _attachment_count(request: Any) -> int:
    # attachments: count tool messages or heuristic file refs > threshold
    # Also support explicit attribute
    if hasattr(request, "attachments"):
        try:
            return len(getattr(request, "attachments"))
        except Exception:
            pass
    # fallback: count messages that look like file attachments (content includes "```" or file paths)
    # For spec: treat as 0 unless caller provides attachment_count via messages metadata
    # We also check for additional field "attachments" in dict form
    if isinstance(request, dict):
        return len(request.get("attachments", []))
    return 0


def classify(
    request: Any,
    turn_state: Any | None = None,
    score_fn: Callable | None = None,
) -> Decision:
    """
    Pure synchronous classifier.
    request: ChatRequest or dict with .messages / ["messages"]
    turn_state: TurnState | None
    score_fn: injectable for tests, signature (last_message, turn_state) -> float
    """
    t0 = time.perf_counter()
    try:
        messages = getattr(request, "messages", None)
        if messages is None and isinstance(request, dict):
            messages = request.get("messages", [])
        if messages is None:
            messages = []

        prompt = _last_user_text(messages)
        elapsed_base = (time.perf_counter() - t0) * 1000

        # Tier 1: fast-path regex override
        if len(prompt) < FAST_PROMPT_MAX_LEN and REGEX_FAST.search(prompt.strip()):
            return Decision(
                path="fast",
                reason="tier1_regex",
                T_i=None,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

        # Tier 2: slow-path override
        ac = _attachment_count(request)
        lower = prompt.lower()
        if ac > 3 or any(kw in lower for kw in SLOW_KEYWORDS):
            return Decision(
                path="slow",
                reason="tier2_override",
                T_i=None,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

        # Tier 3: score band
        # lazy import to avoid cycle
        if score_fn is None:
            from .evaluator import score as _score

            score_fn = _score

        last_msg = messages[-1] if messages else {"content": ""}
        try:
            T_i = float(score_fn(last_msg, turn_state))
        except Exception:
            T_i = 0.5

        T_i = max(0.0, min(1.0, T_i))
        if T_i < AMBIGUOUS_LOW:
            return Decision(path="fast", reason="tier3_low", T_i=T_i, elapsed_ms=(time.perf_counter() - t0) * 1000)
        if T_i > AMBIGUOUS_HIGH:
            return Decision(path="slow", reason="tier3_high", T_i=T_i, elapsed_ms=(time.perf_counter() - t0) * 1000)
        return Decision(path="ambiguous", reason="tier3_band", T_i=T_i, elapsed_ms=(time.perf_counter() - t0) * 1000)
    except Exception:
        return Decision(path="fast", reason="gatekeeper_error", T_i=0.5, elapsed_ms=(time.perf_counter() - t0) * 1000)
