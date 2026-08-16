from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .presets_data import FALLBACK_PATH
from .presets_fallback import FALLBACK_PRESETS

_litellm_costs_cache: dict[str, dict] | None = None

def _load_fallback() -> dict[str, dict]:
    if FALLBACK_PATH.exists():
        try:
            return json.loads(FALLBACK_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    # The checked-in provider fallback is the authoritative backup when the
    # generated pricing snapshot is absent or unreadable.
    return {
        row["id"]: dict(row)
        for rows in FALLBACK_PRESETS.values()
        for row in rows
        if isinstance(row, dict) and row.get("id")
    }


def _ingest_litellm_costs(enrich_pricing: bool = True) -> dict[str, dict]:
    global _litellm_costs_cache
    if _litellm_costs_cache is not None:
        return _litellm_costs_cache
    # Gate the expensive litellm.model_cost iteration behind a flag so that
    # fast paths like ``--claude`` never pay this cost.  It defaults to True
    # for backward compatibility; callers that don't need enriched pricing
    # (e.g. agent launcher, onboarding screens) should pass False.
    if not enrich_pricing:
        _litellm_costs_cache = {}
        return _litellm_costs_cache
    try:
        import litellm  # type: ignore

        cost = getattr(litellm, "model_cost", {}) or {}
        out: dict[str, dict] = {}
        for k, v in cost.items():
            if isinstance(v, dict) and "input_cost_per_token" in v:
                # litellm stores per token; convert to per 1M tokens
                out[k] = {
                    "price_in": float(v.get("input_cost_per_token", 0)) * 1_000_000,
                    "price_out": float(v.get("output_cost_per_token", 0)) * 1_000_000,
                }
        _litellm_costs_cache = out
        return out
    except Exception:
        return {}


def _catalog_provider(model_id: str, raw: dict[str, Any] | None = None) -> str:
    provider = (raw or {}).get("litellm_provider") or (raw or {}).get("provider")
    if provider:
        return str(provider)
    return model_id.split("/", 1)[0] if "/" in model_id else "openai"


_CATALOG_QUALIFIERS = (
    "us.",
    "eu.",
    "apac.",
    "bedrock.",
    "azure.",
    "vertex_ai.",
    "anthropic.",
    "meta.",
    "amazon.",
    "mistral.",
    "cohere.",
    "ai21.",
)


def clean_model_id(model_id: str) -> str:
    cleaned = model_id.rsplit("/", 1)[-1]
    while True:
        lower = cleaned.lower()
        qualifier = next((q for q in _CATALOG_QUALIFIERS if lower.startswith(q)), None)
        if not qualifier:
            return cleaned
        cleaned = cleaned[len(qualifier) :]


def _catalog_id_is_unqualified(model_id: str) -> bool:
    cleaned = model_id.rsplit("/", 1)[-1].lower()
    return "/" not in model_id and not any(
        cleaned.startswith(q) for q in _CATALOG_QUALIFIERS
    )
