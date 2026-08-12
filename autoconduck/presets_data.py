from __future__ import annotations

from pathlib import Path
from typing import Any

FALLBACK_PATH = Path(__file__).parent / "pricing_fallback.json"

CATALOG_SHORTLIST = (
    "gpt-4o",
    "gpt-4o-mini",
    "o1-mini",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-3-opus-20240229",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "mistral-large-latest",
    "llama-3.1-70b-instruct",
)

# Preset groups

from .presets_fallback import FALLBACK_PRESETS

PRESETS: dict[str, list[dict[str, Any]]] = {
    "anthropic": [
        {
            "id": "claude-sonnet-5",
            "provider": "anthropic",
            "tier": "balanced",
            "price_in": 2.0,
            "price_out": 10.0,
            "api_key_env": "ANTHROPIC_API_KEY",
        },
        {
            "id": "claude-sonnet-4-6",
            "provider": "anthropic",
            "tier": "balanced",
            "price_in": 3.0,
            "price_out": 15.0,
            "api_key_env": "ANTHROPIC_API_KEY",
        },
        {
            "id": "claude-opus-5",
            "provider": "anthropic",
            "tier": "expensive",
            "price_in": 5.0,
            "price_out": 25.0,
            "api_key_env": "ANTHROPIC_API_KEY",
        },
        {
            "id": "claude-opus-4-8",
            "provider": "anthropic",
            "tier": "expensive",
            "price_in": 5.0,
            "price_out": 25.0,
            "api_key_env": "ANTHROPIC_API_KEY",
        },
        {
            "id": "claude-haiku-4-5",
            "provider": "anthropic",
            "tier": "budget",
            "price_in": 1.0,
            "price_out": 5.0,
            "api_key_env": "ANTHROPIC_API_KEY",
        },
    ],
    "openai": [
        {
            "id": "gpt-5.6-luna",
            "provider": "openai",
            "tier": "budget",
            "price_in": 0.2,
            "price_out": 1.2,
            "api_key_env": "OPENAI_API_KEY",
        },
        {
            "id": "gpt-5.6",
            "provider": "openai",
            "tier": "expensive",
            "price_in": 5.0,
            "price_out": 30.0,
            "api_key_env": "OPENAI_API_KEY",
        },
        {
            "id": "gpt-5.2",
            "provider": "openai",
            "tier": "balanced",
            "price_in": 1.75,
            "price_out": 14.0,
            "api_key_env": "OPENAI_API_KEY",
        },
        {
            "id": "gpt-5-mini",
            "provider": "openai",
            "tier": "budget",
            "price_in": 0.25,
            "price_out": 2.0,
            "api_key_env": "OPENAI_API_KEY",
        },
        {
            "id": "gpt-5-nano",
            "provider": "openai",
            "tier": "budget",
            "price_in": 0.05,
            "price_out": 0.4,
            "api_key_env": "OPENAI_API_KEY",
        },
        {
            "id": "gpt-5-pro",
            "provider": "openai",
            "tier": "expensive",
            "price_in": 15.0,
            "price_out": 120.0,
            "api_key_env": "OPENAI_API_KEY",
        },
        {
            "id": "gpt-4.1",
            "provider": "openai",
            "tier": "balanced",
            "price_in": 2.0,
            "price_out": 8.0,
            "api_key_env": "OPENAI_API_KEY",
        },
    ],
    "google": [
        {
            "id": "gemini-3.5-flash",
            "provider": "google",
            "tier": "balanced",
            "price_in": 1.5,
            "price_out": 9.0,
            "api_key_env": "GOOGLE_API_KEY",
        },
        {
            "id": "gemini-3.5-flash-lite",
            "provider": "google",
            "tier": "budget",
            "price_in": 0.3,
            "price_out": 2.5,
            "api_key_env": "GOOGLE_API_KEY",
        },
        {
            "id": "gemini-3.1-pro-preview",
            "provider": "google",
            "tier": "balanced",
            "price_in": 2.0,
            "price_out": 12.0,
            "api_key_env": "GOOGLE_API_KEY",
        },
        {
            "id": "gemini-2.5-pro",
            "provider": "google",
            "tier": "balanced",
            "price_in": 1.25,
            "price_out": 10.0,
            "api_key_env": "GOOGLE_API_KEY",
        },
        {
            "id": "gemini-2.5-flash",
            "provider": "google",
            "tier": "budget",
            "price_in": 0.3,
            "price_out": 2.5,
            "api_key_env": "GOOGLE_API_KEY",
        },
    ],
    "mistral": [
        {
            "id": "mistral-large-latest",
            "provider": "mistral",
            "tier": "balanced",
            "price_in": 2.0,
            "price_out": 6.0,
            "api_key_env": "MISTRAL_API_KEY",
        },
    ],
}
PRESETS.update(FALLBACK_PRESETS)

PRESET_ORDER = ["custom", "openai", "anthropic", "google", "mistral", "llmgateway", "opencodego"]

# Compatibility exports from the original ``model_presets`` module.  Keep
# these derived from PRESETS so the catalog cannot silently drift from the
# preset data when a provider adds or removes a model.
DEFAULT_MODELS = [row["id"] for rows in PRESETS.values() for row in rows]
MODEL_IDS = list(DEFAULT_MODELS)
