from __future__ import annotations

import math
import re
from typing import Optional

try:
    import tiktoken

    _enc = tiktoken.get_encoding("cl100k_base")

    def _encode_len(text: str) -> int:
        try:
            return len(_enc.encode(text))
        except Exception:
            return max(1, len(text) // 4)

except ImportError:

    def _encode_len(text: str) -> int:
        return max(1, len(text) // 4)


from .config import Config, ModelEntry
from . import state as state_mod

EMA_ALPHA = 0.1
DEGRADED_ERROR_RATE = 0.20
DEGRADED_WINDOW_SECONDS = 300
DEGRADED_MIN_SAMPLES = 5

INTENT_KEYWORDS: dict[str, list[str]] = {
    "fix": ["fix", "bug", "error", "traceback", "exception"],
    "refactor": ["refactor", "rewrite", "migrate", "restructure"],
    "architecture": ["architecture", "design system", "monorepo", "codebase-wide"],
    "build_feature": ["build feature", "implement feature", "create feature"],
    "format": ["format", "lint", "prettier", "typo"],
}


def _detect_intent(text: str) -> str:
    lower = text.lower()
    for intent, kws in INTENT_KEYWORDS.items():
        if any(kw in lower for kw in kws):
            return intent
    return "default"


def transform(T: float, pseudo: str) -> float:
    if pseudo == "autoconduck-budget":
        return T * 0.6
    if pseudo == "autoconduck-expensive":
        return min(1.0, T * 1.4 + 0.1)
    return T


def estimate_tokens(messages: list, config: Config, intent: str | None = None) -> tuple[int, int]:
    # t_in: exact via tiktoken over all messages
    total_text = ""
    for m in messages:
        c = getattr(m, "content", None)
        if c is None and isinstance(m, dict):
            c = m.get("content", "")
        if isinstance(c, list):
            c = " ".join(str(x.get("text", x) if isinstance(x, dict) else str(x)) for x in c)
        total_text += str(c or "") + "\n"
    t_in = _encode_len(total_text)
    # t_out: intent table + EMA correction
    if intent is None:
        intent = _detect_intent(total_text)
    base = config.intent_tokens.get(intent, config.intent_tokens.get("default", 800))
    ema_val = state_mod.get_ema().get(intent)
    # EMA is already smoothing; use ema value if available else base
    # Blend: use EMA value directly (it started at default)
    t_out = int(ema_val) if ema_val else base
    return t_in, t_out


def scaled_cost(cost: float) -> float:
    return math.log(1 + cost)


def select(
    T_prime: float,
    models: list[ModelEntry],
    t_in: int,
    t_out: int,
) -> ModelEntry:
    enabled = [m for m in models if m.enabled]
    if not enabled:
        raise ValueError("no enabled models")

    # tier buckets
    if T_prime < 0.33:
        tier_pool = [m for m in enabled if m.tier == "budget"]
    elif T_prime <= 0.75:
        tier_pool = [m for m in enabled if m.tier in ("balanced", "budget")]
        # prefer balanced but allow budget; if no balanced, use budget
        balanced = [m for m in tier_pool if m.tier == "balanced"]
        if balanced:
            tier_pool = balanced
    else:
        tier_pool = [m for m in enabled if m.tier in ("expensive", "reasoning")]
        if not tier_pool:
            tier_pool = [m for m in enabled if m.tier == "balanced"]

    if not tier_pool:
        tier_pool = enabled

    # compute costs
    def cost_of(m: ModelEntry) -> float:
        # price per 1K -> per token
        c = (m.price_in / 1000.0) * t_in + (m.price_out / 1000.0) * t_out
        return scaled_cost(c)

    # filter degraded
    non_degraded = [m for m in tier_pool if not state_mod.is_degraded(m.id)]
    pool = non_degraded if non_degraded else tier_pool
    # cheapest by scaled cost
    pool_sorted = sorted(pool, key=cost_of)
    return pool_sorted[0]


def record_usage(model_id: str, actual_in: int, actual_out: int, intent: str = "default") -> None:
    state_mod.record_usage(model_id, actual_in, actual_out, intent)


def record_error(model_id: str) -> None:
    state_mod.record_error(model_id)


def is_degraded(model_id: str) -> bool:
    return state_mod.is_degraded(model_id)
