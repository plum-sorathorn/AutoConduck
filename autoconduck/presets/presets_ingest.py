from __future__ import annotations

import json
from typing import Any

from .presets_data import FALLBACK_PATH
from .presets_fallback import FALLBACK_PRESETS

_litellm_costs_cache: dict[str, dict] | None = None
_benchmark_scores_cache: dict[str, float] | None = None


def _ingest_benchmark_scores(enrich_pricing: bool = True) -> dict[str, float]:
    """
    Fetch external benchmark scores (e.g., Aider leaderboard or LMSYS Elo) to automatically 
    assign capability scores to models, avoiding manual scoring.
    """
    global _benchmark_scores_cache
    if _benchmark_scores_cache is not None:
        return _benchmark_scores_cache
        
    if not enrich_pricing:
        _benchmark_scores_cache = {}
        return _benchmark_scores_cache
        
    try:
        # In a production environment, this would pull from a stable CDN or HuggingFace Dataset
        # containing normalized MMLU/Elo scores mapped to model IDs.
        # For now, we seed it with an empty cache to be populated by the update script.
        _benchmark_scores_cache = {}
        return _benchmark_scores_cache
    except Exception:
        return {}

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


# Provider-specific model ID normalization tables.
#
# Some gateways (DevPass / LLMGateway) use a different canonical model ID than the
# upstream vendor.  For example the xAI API uses ``grok-4.5`` while DevPass expects
# ``grok-4-5`` (dash instead of dot between major and minor version numbers).
#
# Mapping keys are the *raw* model ID from the gateway catalog. Values are the
# correct string to send to that provider's ``/v1/chat/completions`` endpoint.
def _id_mapping():
    """Return a frozen dict mapping raw gateway IDs to their correct dispatch form."""
    return {
        # --- DevPass / LLMGateway (both share api.llmgateway.io) ---
        # Grok series -- X.AI uses dots but DevPass gateways use dashes
        "grok-4.5": "grok-4-5",
        "grok-4_5": "grok-4-5",  # URL-encoded variant
        # Other potential future mismatches -- extend as needed
    }


_GW_PROVIDER_KEYS = ("devpass", "llmgateway")


def normalize_model_id_for_provider(model_id: str, provider: str) -> str:
    """Apply provider-specific model-ID normalisation where the gateway differs
    from the vendor-supplied identifier.

    Many providers (OpenAI, Anthropic, Google, xAI, etc.) ship directly-compatible
    IDs so the function is a no-op for those cases.  It only transforms IDs when a
    gateway such as DevPass or LLMGateway requires a different syntax than the
    vendor's own API.

    Examples
    --------
    >>> normalize_model_id_for_provider("grok-4.5", "devpass")
    'grok-4-5'
    >>> normalize_model_id_for_provider("gpt-4o", "openai")
    'gpt-4o'
    """
    if provider not in _GW_PROVIDER_KEYS:
        return model_id

    mappings = _id_mapping()
    if model_id in mappings:
        return mappings[model_id]

    # Only rewrite Grok vendor-style dotted versions (grok-4.5 -> grok-4-5).
    # Other dotted IDs on these gateways (gpt-5.6, qwen3.7-flash, glm-4.6) are
    # already the live API strings.
    import re

    if re.match(r"(?i)^grok-", model_id):
        return re.sub(r"(?<=-[0-9])\.(?=[0-9]+(?:-|$))", "-", model_id, count=1)
    return model_id


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
