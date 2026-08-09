"""Pure data helpers used by the onboarding screens."""
from __future__ import annotations
from typing import Any

DEFAULT_PRICE_IN = 0.001
DEFAULT_PRICE_OUT = 0.002

def models_for_provider(key: str, presets: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [dict(row) for row in presets.get(key, [])]

def overrides_for_toggle(key: str, models: list[dict[str, Any]], enabled: set[str]) -> list[dict[str, Any]]:
    return [dict(model) for model in models if model["id"] in enabled]

def default_enabled_ids(models: list[dict[str, Any]], existing_overrides: list[dict[str, Any]] | None = None) -> set[str]:
    model_ids = {model["id"] for model in models}
    if existing_overrides:
        return {row["id"] for row in existing_overrides if row.get("id") in model_ids}
    return model_ids if len(models) <= 5 else set()

def upsert_custom_models(existing: list[dict[str, Any]], provider: str, base_url: str, api_key_env: str, model_ids: list[str]) -> list[dict[str, Any]]:
    result = [row for row in existing if row.get("provider") != provider]
    result.extend({"id": model_id, "provider": provider, "api_key": api_key_env,
                   "base_url": base_url, "price_in": DEFAULT_PRICE_IN, "price_out": DEFAULT_PRICE_OUT, "enabled": True}
                  for model_id in dict.fromkeys(x.strip() for x in model_ids) if model_id.strip())
    return result

def remove_custom_provider(existing: list[dict[str, Any]], provider: str) -> list[dict[str, Any]]:
    return [row for row in existing if row.get("provider") != provider]

