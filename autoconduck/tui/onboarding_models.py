"""Pure data helpers used by the onboarding screens."""
from __future__ import annotations
import re
from copy import deepcopy
from typing import Any

from ..model_presets import _ingest_litellm_costs, clean_model_id

DEFAULT_PRICE_IN = 0.001
DEFAULT_PRICE_OUT = 0.002
def apply_api_key(entries: list[dict], value: str) -> list[dict]:
    """Apply either an environment variable name or literal key immutably."""
    result = deepcopy(entries)
    value = value.strip()
    if not value:
        return result
    if value.startswith("env:"):
        return [{**entry_without_key, "api_key_env": value[4:]}
                for entry in result
                for entry_without_key in [{key: item for key, item in entry.items() if key != "api_key"}]]
    from ..auth import set_provider_key
    providers = {str(entry.get("provider") or str(entry.get("id", "")).split("/", 1)[0] or "openai") for entry in result}
    for provider in providers:
        set_provider_key(provider, value)
    return [{**entry_without_env}
            for entry in result
            for entry_without_env in [{key: item for key, item in entry.items() if key != "api_key_env"}]]

def search_match(term: str, *fields: str) -> bool:
    """Return whether a separator-insensitive term occurs in any field."""
    normalize = lambda value: re.sub(r"[^a-z0-9]", "", value.lower())
    normalized_term = normalize(term)
    return not normalized_term or any(normalized_term in normalize(field) for field in fields)

def models_for_provider(key: str, presets: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [dict(row) for row in presets.get(key, [])]

def overrides_for_toggle(key: str, models: list[dict[str, Any]], enabled: set[str], existing_overrides: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    stored = {row.get("id"): row for row in (existing_overrides or [])}
    result = []
    for model in models:
        if model["id"] not in enabled:
            continue
        fresh = dict(model)
        old = stored.get(model["id"], {})
        for field in ("api_key", "api_key_env"):
            if field in old:
                fresh[field] = old[field]
        result.append(fresh)
    return result


def default_enabled_ids(models: list[dict[str, Any]], existing_overrides: list[dict[str, Any]] | None = None) -> set[str]:
    model_ids = {model["id"] for model in models}
    if existing_overrides:
        return {row["id"] for row in existing_overrides if row.get("id") in model_ids}
    return model_ids if len(models) <= 6 else set()

def upsert_custom_models(existing: list[dict[str, Any]], provider: str, base_url: str, api_key_env: str, model_ids: list[str], anthropic_base_url: str = "") -> list[dict[str, Any]]:
    result = [row for row in existing if row.get("provider") != provider]
    costs = _ingest_litellm_costs()
    by_clean_id = {clean_model_id(model_id): values for model_id, values in costs.items()}
    for model_id in dict.fromkeys(x.strip() for x in model_ids):
        if model_id:
            prices = by_clean_id.get(clean_model_id(model_id), {})
            if api_key_env.startswith("env:"):
                auth = {"api_key_env": api_key_env[4:]}
            else:
                from ..auth import set_provider_key
                set_provider_key(provider, api_key_env)
                auth = {}
            entry = {"id": model_id, "provider": provider, **auth,
                     "base_url": base_url, "price_in": prices.get("price_in", DEFAULT_PRICE_IN),
                     "price_out": prices.get("price_out", DEFAULT_PRICE_OUT), "enabled": True}
            if anthropic_base_url and anthropic_base_url.strip():
                entry["anthropic_base_url"] = anthropic_base_url.strip()
            result.append(entry)
    return result

def remove_custom_provider(existing: list[dict[str, Any]], provider: str) -> list[dict[str, Any]]:
    return [row for row in existing if row.get("provider") != provider]
