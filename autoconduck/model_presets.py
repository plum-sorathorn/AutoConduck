from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import ModelEntry

# Bundled fallback pricing (populated in pricing_fallback.json)
FALLBACK_PATH = Path(__file__).parent / "pricing_fallback.json"

# Preset groups

PRESETS: dict[str, list[dict[str, Any]]] = {
    "anthropic": [
        {"id": "claude-3-5-sonnet-20241022", "provider": "anthropic", "tier": "reasoning", "price_in": 3.0, "price_out": 15.0, "api_key_env": "ANTHROPIC_API_KEY"},
        {"id": "claude-3-5-haiku-20241022", "provider": "anthropic", "tier": "budget", "price_in": 0.25, "price_out": 1.25, "api_key_env": "ANTHROPIC_API_KEY"},
        {"id": "claude-3-opus-20240229", "provider": "anthropic", "tier": "expensive", "price_in": 15.0, "price_out": 75.0, "api_key_env": "ANTHROPIC_API_KEY"},
    ],
    "openai": [
        {"id": "gpt-4o", "provider": "openai", "tier": "balanced", "price_in": 2.5, "price_out": 10.0, "api_key_env": "OPENAI_API_KEY"},
        {"id": "gpt-4o-mini", "provider": "openai", "tier": "budget", "price_in": 0.15, "price_out": 0.6, "api_key_env": "OPENAI_API_KEY"},
        {"id": "o1-preview", "provider": "openai", "tier": "reasoning", "price_in": 15.0, "price_out": 60.0, "api_key_env": "OPENAI_API_KEY"},
        {"id": "o1-mini", "provider": "openai", "tier": "budget", "price_in": 3.0, "price_out": 12.0, "api_key_env": "OPENAI_API_KEY"},
    ],
    "google": [
        {"id": "gemini-1.5-pro", "provider": "google", "tier": "balanced", "price_in": 1.25, "price_out": 5.0, "api_key_env": "GOOGLE_API_KEY"},
        {"id": "gemini-1.5-flash", "provider": "google", "tier": "budget", "price_in": 0.075, "price_out": 0.30, "api_key_env": "GOOGLE_API_KEY"},
    ],
    "mistral": [
        {"id": "mistral-large-latest", "provider": "mistral", "tier": "balanced", "price_in": 2.0, "price_out": 6.0, "api_key_env": "MISTRAL_API_KEY"},
    ],
}

PRESET_ORDER = ["custom", "openai", "anthropic", "google", "mistral"]

def default_preset_models(key: str) -> list[dict[str, Any]]:
    return list(PRESETS.get(key, []))


def _load_fallback() -> dict[str, dict]:
    if FALLBACK_PATH.exists():
        try:
            return json.loads(FALLBACK_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _ingest_litellm_costs() -> dict[str, dict]:
    try:
        import litellm  # type: ignore

        cost = getattr(litellm, "model_cost", {}) or {}
        out: dict[str, dict] = {}
        for k, v in cost.items():
            if isinstance(v, dict) and "input_cost_per_token" in v:
                # litellm stores per token; convert to per 1K
                out[k] = {
                    "price_in": float(v.get("input_cost_per_token", 0)) * 1000,
                    "price_out": float(v.get("output_cost_per_token", 0)) * 1000,
                }
        return out
    except Exception:
        return {}


def discover_models(
    preset_keys: list[str] | None = None,
    custom_models: list[dict[str, Any]] | None = None,
    use_litellm: bool = True,
    overrides: dict[str, list[ModelEntry]] | None = None,
) -> list[ModelEntry]:
    """
    Build normalized ModelEntry list from presets + custom + pricing registry.
    """
    entries: list[ModelEntry] = []
    fallback = _load_fallback()
    litellm_costs = _ingest_litellm_costs() if use_litellm else {}

    keys = preset_keys or []
    for k in keys:
        raw_models = custom_models if k == "custom" else (overrides[k] if overrides and k in overrides else PRESETS.get(k, []))
        for raw_entry in raw_models or []:
            raw = raw_entry.model_dump(mode="python") if isinstance(raw_entry, ModelEntry) else raw_entry
            mid = raw["id"]
            # enrich price from litellm if available
            price_in = raw.get("price_in")
            price_out = raw.get("price_out")
            if mid in litellm_costs:
                price_in = litellm_costs[mid].get("price_in", price_in)
                price_out = litellm_costs[mid].get("price_out", price_out)
            elif mid in fallback:
                price_in = fallback[mid].get("price_in", price_in)
                price_out = fallback[mid].get("price_out", price_out)
            entries.append(
                ModelEntry(
                    id=mid,
                    provider=raw.get("provider", "openai"),
                    api_key_env=raw.get("api_key_env", "OPENAI_API_KEY"),
                    tier=raw.get("tier", "balanced"),  # type: ignore
                    price_in=float(price_in or 0),
                    price_out=float(price_out or 0),
                    enabled=True,
                )
            )

    for raw in ([] if "custom" in keys else (custom_models or [])):
        mid = raw.get("id") or raw.get("model")
        if not mid:
            continue
        # price lookup
        pi = raw.get("price_in")
        po = raw.get("price_out")
        if mid in litellm_costs and pi is None:
            pi = litellm_costs[mid].get("price_in")
            po = litellm_costs[mid].get("price_out")
        entries.append(
            ModelEntry(
                id=str(mid),
                provider=str(raw.get("provider", "openai")),
                api_key_env=str(raw.get("api_key_env", "OPENAI_API_KEY")),
                tier=str(raw.get("tier", "balanced")),  # type: ignore
                price_in=float(pi or 0.001),
                price_out=float(po or 0.002),
                enabled=bool(raw.get("enabled", True)),
            )
        )

    # de-duplicate by id (last wins)
    seen: dict[str, ModelEntry] = {}
    for e in entries:
        seen[e.id] = e
    return list(seen.values())


def normalize_entries(raw_list: list[dict[str, Any]]) -> list[ModelEntry]:
    return [ModelEntry.model_validate(r) for r in raw_list]

def resolve_models(cfg: Any) -> list[ModelEntry]:
    models = discover_models(getattr(cfg, "selected_presets", []), getattr(cfg, "custom_models", []), overrides=getattr(cfg, "preset_overrides", {}))
    cfg.model_list = [model.model_dump() for model in models]
    return models
