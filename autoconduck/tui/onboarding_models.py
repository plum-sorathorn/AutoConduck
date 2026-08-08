"""Pure data helpers used by the onboarding screens."""
from __future__ import annotations
from typing import Any

DEFAULT_PRICE_IN = 0.001
DEFAULT_PRICE_OUT = 0.002

def normalize_provider_models(provider: dict[str, Any], key: str) -> list[dict[str, Any]]:
    """Normalize a gateway/provider preset into the preset model shape."""
    models = provider.get("models", [])
    if isinstance(models, dict):
        models = list(models)
    return [{"id": str(model.get("id", model) if isinstance(model, dict) else model),
             "provider": str(model.get("provider", key) if isinstance(model, dict) else key),
             "tier": str(model.get("tier", "balanced") if isinstance(model, dict) else "balanced"),
             "price_in": float(model.get("price_in", DEFAULT_PRICE_IN) if isinstance(model, dict) else DEFAULT_PRICE_IN),
             "price_out": float(model.get("price_out", DEFAULT_PRICE_OUT) if isinstance(model, dict) else DEFAULT_PRICE_OUT),
             "api_key_env": str(model.get("api_key_env", provider.get("api_key_env", "")) if isinstance(model, dict) else provider.get("api_key_env", "")),
             **({"base_url": provider["base_url"]} if provider.get("base_url") else {})}
            for model in models if str(model.get("id", model) if isinstance(model, dict) else model)]

def models_for_provider(key: str, presets: dict[str, list[dict[str, Any]]], devpass: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    return normalize_provider_models(devpass or {}, key) if key == "devpass" else [dict(row) for row in presets.get(key, [])]

def overrides_for_toggle(key: str, models: list[dict[str, Any]], enabled: set[str]) -> list[dict[str, Any]]:
    return [dict(model) for model in models if model["id"] in enabled]

def upsert_custom_models(existing: list[dict[str, Any]], provider: str, base_url: str, api_key_env: str, model_ids: list[str]) -> list[dict[str, Any]]:
    result = [row for row in existing if row.get("provider") != provider]
    result.extend({"id": model_id, "provider": provider, "api_key_env": api_key_env,
                   "base_url": base_url, "price_in": DEFAULT_PRICE_IN, "price_out": DEFAULT_PRICE_OUT}
                  for model_id in dict.fromkeys(x.strip() for x in model_ids) if model_id.strip())
    return result

def remove_custom_provider(existing: list[dict[str, Any]], provider: str) -> list[dict[str, Any]]:
    return [row for row in existing if row.get("provider") != provider]
