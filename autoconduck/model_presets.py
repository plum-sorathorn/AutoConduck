from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import ModelEntry

# Bundled fallback pricing (populated in pricing_fallback.json)
FALLBACK_PATH = Path(__file__).parent / "pricing_fallback.json"
_litellm_costs_cache: dict[str, dict] | None = None
_catalog_cache: list[dict[str, Any]] | None = None

# A deliberately small, useful set for UIs which do not want to render the
# entire LiteLLM registry.
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
    "llmgateway": [
        {
            "id": "qwen3.7-flash",
            "provider": "llmgateway",
            "tier": "budget",
            "price_in": 0.03,
            "price_out": 0.13,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "muse-spark-1.2",
            "provider": "llmgateway",
            "tier": "balanced",
            "price_in": 1.25,
            "price_out": 4.25,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "muse-spark-1.1",
            "provider": "llmgateway",
            "tier": "balanced",
            "price_in": 1.25,
            "price_out": 4.25,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "gpt-5.6-luna",
            "provider": "llmgateway",
            "tier": "budget",
            "price_in": 0.2,
            "price_out": 1.2,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "gpt-5.6-terra",
            "provider": "llmgateway",
            "tier": "balanced",
            "price_in": 2.0,
            "price_out": 12.0,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "gpt-5.6-sol",
            "provider": "llmgateway",
            "tier": "expensive",
            "price_in": 5.0,
            "price_out": 30.0,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "claude-sonnet-5",
            "provider": "llmgateway",
            "tier": "balanced",
            "price_in": 2.0,
            "price_out": 10.0,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "claude-opus-5",
            "provider": "llmgateway",
            "tier": "expensive",
            "price_in": 5.0,
            "price_out": 25.0,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "gemini-3.6-flash",
            "provider": "llmgateway",
            "tier": "balanced",
            "price_in": 1.5,
            "price_out": 7.5,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "gemini-3.5-flash-lite",
            "provider": "llmgateway",
            "tier": "budget",
            "price_in": 0.3,
            "price_out": 2.5,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "kimi-k3",
            "provider": "llmgateway",
            "tier": "expensive",
            "price_in": 3.0,
            "price_out": 15.0,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "kimi-k3-fast",
            "provider": "llmgateway",
            "tier": "expensive",
            "price_in": 4.5,
            "price_out": 22.5,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "grok-4.5",
            "provider": "llmgateway",
            "tier": "balanced",
            "price_in": 2.0,
            "price_out": 6.0,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "hy3",
            "provider": "llmgateway",
            "tier": "budget",
            "price_in": 0.14,
            "price_out": 0.58,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "qwen3.6-flash",
            "provider": "llmgateway",
            "tier": "budget",
            "price_in": 0.17,
            "price_out": 0.99,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "qwen3.8-max",
            "provider": "llmgateway",
            "tier": "balanced",
            "price_in": 1.81,
            "price_out": 5.45,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "qwen3.7-max",
            "provider": "llmgateway",
            "tier": "balanced",
            "price_in": 1.25,
            "price_out": 3.75,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "qwen3.7-plus",
            "provider": "llmgateway",
            "tier": "budget",
            "price_in": 0.4,
            "price_out": 1.6,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "qwen3.6-plus",
            "provider": "llmgateway",
            "tier": "balanced",
            "price_in": 0.5,
            "price_out": 3.0,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "qwen3.6-max-preview",
            "provider": "llmgateway",
            "tier": "balanced",
            "price_in": 1.3,
            "price_out": 7.8,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "qwen3-max",
            "provider": "llmgateway",
            "tier": "balanced",
            "price_in": 0.84,
            "price_out": 3.38,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "qwen-flash",
            "provider": "llmgateway",
            "tier": "budget",
            "price_in": 0.02,
            "price_out": 0.22,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "qwen-plus",
            "provider": "llmgateway",
            "tier": "budget",
            "price_in": 0.11,
            "price_out": 0.29,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "qwen-plus-latest",
            "provider": "llmgateway",
            "tier": "budget",
            "price_in": 0.11,
            "price_out": 0.29,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "qwen3-coder-next",
            "provider": "llmgateway",
            "tier": "budget",
            "price_in": 0.11,
            "price_out": 0.68,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "qwen3-coder-flash",
            "provider": "llmgateway",
            "tier": "budget",
            "price_in": 0.14,
            "price_out": 0.57,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "qwen3-coder-plus",
            "provider": "llmgateway",
            "tier": "expensive",
            "price_in": 6.0,
            "price_out": 60.0,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "qwen3-coder-480b-a35b-instruct",
            "provider": "llmgateway",
            "tier": "budget",
            "price_in": 0.22,
            "price_out": 1.8,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "qwen3-vl-flash",
            "provider": "llmgateway",
            "tier": "budget",
            "price_in": 0.02,
            "price_out": 0.21,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "qwen3-vl-235b-a22b-instruct",
            "provider": "llmgateway",
            "tier": "budget",
            "price_in": 0.2,
            "price_out": 0.88,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "claude-opus-4-1-20250805",
            "provider": "llmgateway",
            "tier": "expensive",
            "price_in": 15.0,
            "price_out": 75.0,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "claude-opus-4-5-20251101",
            "provider": "llmgateway",
            "tier": "expensive",
            "price_in": 5.0,
            "price_out": 25.0,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "claude-opus-4-6",
            "provider": "llmgateway",
            "tier": "expensive",
            "price_in": 5.0,
            "price_out": 25.0,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "claude-opus-4-7",
            "provider": "llmgateway",
            "tier": "expensive",
            "price_in": 5.0,
            "price_out": 25.0,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "claude-opus-4-8",
            "provider": "llmgateway",
            "tier": "expensive",
            "price_in": 5.0,
            "price_out": 25.0,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "claude-sonnet-4-5",
            "provider": "llmgateway",
            "tier": "expensive",
            "price_in": 3.0,
            "price_out": 15.0,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "claude-sonnet-4-5-20250929",
            "provider": "llmgateway",
            "tier": "expensive",
            "price_in": 3.0,
            "price_out": 15.0,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "claude-sonnet-4-6",
            "provider": "llmgateway",
            "tier": "expensive",
            "price_in": 3.0,
            "price_out": 15.0,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "claude-haiku-4-5",
            "provider": "llmgateway",
            "tier": "balanced",
            "price_in": 1.0,
            "price_out": 5.0,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "claude-haiku-4-5-20251001",
            "provider": "llmgateway",
            "tier": "balanced",
            "price_in": 1.0,
            "price_out": 5.0,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "claude-fable-5",
            "provider": "llmgateway",
            "tier": "expensive",
            "price_in": 10.0,
            "price_out": 50.0,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "claude-3-opus",
            "provider": "llmgateway",
            "tier": "expensive",
            "price_in": 15.0,
            "price_out": 75.0,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "gpt-5.5",
            "provider": "llmgateway",
            "tier": "expensive",
            "price_in": 5.0,
            "price_out": 30.0,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "gpt-5.4",
            "provider": "llmgateway",
            "tier": "expensive",
            "price_in": 2.5,
            "price_out": 15.0,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "gpt-5.4-mini",
            "provider": "llmgateway",
            "tier": "balanced",
            "price_in": 0.75,
            "price_out": 4.5,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "gpt-5.4-nano",
            "provider": "llmgateway",
            "tier": "budget",
            "price_in": 0.2,
            "price_out": 1.25,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "gpt-5.3-codex",
            "provider": "llmgateway",
            "tier": "balanced",
            "price_in": 1.75,
            "price_out": 14.0,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "gpt-5.2-codex",
            "provider": "llmgateway",
            "tier": "balanced",
            "price_in": 1.75,
            "price_out": 14.0,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "gpt-5.2",
            "provider": "llmgateway",
            "tier": "balanced",
            "price_in": 1.75,
            "price_out": 14.0,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "gpt-5.1-codex-mini",
            "provider": "llmgateway",
            "tier": "budget",
            "price_in": 0.25,
            "price_out": 2.0,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "gpt-5.1",
            "provider": "llmgateway",
            "tier": "balanced",
            "price_in": 1.25,
            "price_out": 10.0,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "gpt-5",
            "provider": "llmgateway",
            "tier": "balanced",
            "price_in": 1.25,
            "price_out": 10.0,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "gpt-5-mini",
            "provider": "llmgateway",
            "tier": "budget",
            "price_in": 0.25,
            "price_out": 2.0,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "gpt-5-nano",
            "provider": "llmgateway",
            "tier": "budget",
            "price_in": 0.05,
            "price_out": 0.4,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "gpt-4o",
            "provider": "llmgateway",
            "tier": "balanced",
            "price_in": 2.5,
            "price_out": 10.0,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "gpt-4o-mini",
            "provider": "llmgateway",
            "tier": "budget",
            "price_in": 0.15,
            "price_out": 0.6,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "gpt-4.1",
            "provider": "llmgateway",
            "tier": "balanced",
            "price_in": 2.0,
            "price_out": 8.0,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "gpt-4.1-mini",
            "provider": "llmgateway",
            "tier": "budget",
            "price_in": 0.4,
            "price_out": 1.6,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "gpt-4.1-nano",
            "provider": "llmgateway",
            "tier": "budget",
            "price_in": 0.1,
            "price_out": 0.4,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "o4-mini",
            "provider": "llmgateway",
            "tier": "balanced",
            "price_in": 1.1,
            "price_out": 4.4,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "gpt-oss-120b",
            "provider": "llmgateway",
            "tier": "budget",
            "price_in": 0.05,
            "price_out": 0.25,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "grok-4",
            "provider": "llmgateway",
            "tier": "expensive",
            "price_in": 3.0,
            "price_out": 15.0,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "grok-4-3",
            "provider": "llmgateway",
            "tier": "balanced",
            "price_in": 1.25,
            "price_out": 2.5,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "grok-4-20-beta-0309-reasoning",
            "provider": "llmgateway",
            "tier": "balanced",
            "price_in": 2.0,
            "price_out": 6.0,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "grok-4-20-beta-0309-non-reasoning",
            "provider": "llmgateway",
            "tier": "balanced",
            "price_in": 2.0,
            "price_out": 6.0,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "grok-build-0-1",
            "provider": "llmgateway",
            "tier": "balanced",
            "price_in": 1.0,
            "price_out": 2.0,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "gemini-2.5-flash",
            "provider": "llmgateway",
            "tier": "budget",
            "price_in": 0.3,
            "price_out": 2.5,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "gemini-2.5-flash-lite",
            "provider": "llmgateway",
            "tier": "budget",
            "price_in": 0.1,
            "price_out": 0.4,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "gemini-2.5-pro",
            "provider": "llmgateway",
            "tier": "balanced",
            "price_in": 1.25,
            "price_out": 10.0,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "gemini-3-flash-preview",
            "provider": "llmgateway",
            "tier": "balanced",
            "price_in": 0.5,
            "price_out": 3.0,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "gemini-3.1-flash-lite",
            "provider": "llmgateway",
            "tier": "budget",
            "price_in": 0.25,
            "price_out": 1.5,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "gemini-3.1-pro-preview",
            "provider": "llmgateway",
            "tier": "balanced",
            "price_in": 2.0,
            "price_out": 12.0,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "gemini-3.5-flash",
            "provider": "llmgateway",
            "tier": "balanced",
            "price_in": 1.5,
            "price_out": 9.0,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "gemini-pro-latest",
            "provider": "llmgateway",
            "tier": "balanced",
            "price_in": 2.0,
            "price_out": 12.0,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "kimi-k2.5",
            "provider": "llmgateway",
            "tier": "budget",
            "price_in": 0.4,
            "price_out": 1.98,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "kimi-k2.6",
            "provider": "llmgateway",
            "tier": "budget",
            "price_in": 0.22,
            "price_out": 1.14,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "kimi-k2.7-code",
            "provider": "llmgateway",
            "tier": "balanced",
            "price_in": 0.95,
            "price_out": 4.0,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "kimi-k2.7-code-highspeed",
            "provider": "llmgateway",
            "tier": "balanced",
            "price_in": 1.9,
            "price_out": 8.0,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "kimi-k2-thinking",
            "provider": "llmgateway",
            "tier": "balanced",
            "price_in": 0.6,
            "price_out": 2.5,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "glm-4.6",
            "provider": "llmgateway",
            "tier": "balanced",
            "price_in": 0.55,
            "price_out": 2.2,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "glm-4.6v",
            "provider": "llmgateway",
            "tier": "budget",
            "price_in": 0.3,
            "price_out": 0.9,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "glm-4.6v-flashx",
            "provider": "llmgateway",
            "tier": "budget",
            "price_in": 0.04,
            "price_out": 0.4,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "glm-4.7",
            "provider": "llmgateway",
            "tier": "budget",
            "price_in": 0.38,
            "price_out": 1.98,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "glm-4.5v",
            "provider": "llmgateway",
            "tier": "balanced",
            "price_in": 0.6,
            "price_out": 1.8,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "glm-5",
            "provider": "llmgateway",
            "tier": "balanced",
            "price_in": 0.57,
            "price_out": 2.58,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "glm-5.1",
            "provider": "llmgateway",
            "tier": "balanced",
            "price_in": 0.93,
            "price_out": 2.93,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "glm-5.2",
            "provider": "llmgateway",
            "tier": "balanced",
            "price_in": 0.55,
            "price_out": 1.93,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "glm-4-32b-0414-128k",
            "provider": "llmgateway",
            "tier": "budget",
            "price_in": 0.1,
            "price_out": 0.1,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "minimax-m2",
            "provider": "llmgateway",
            "tier": "budget",
            "price_in": 0.2,
            "price_out": 1.0,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "minimax-m2.1",
            "provider": "llmgateway",
            "tier": "budget",
            "price_in": 0.27,
            "price_out": 1.1,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "minimax-m2.5",
            "provider": "llmgateway",
            "tier": "budget",
            "price_in": 0.3,
            "price_out": 1.2,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "minimax-m2.5-highspeed",
            "provider": "llmgateway",
            "tier": "balanced",
            "price_in": 0.6,
            "price_out": 2.4,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "minimax-m2.7",
            "provider": "llmgateway",
            "tier": "budget",
            "price_in": 0.08,
            "price_out": 0.32,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "minimax-m2.7-highspeed",
            "provider": "llmgateway",
            "tier": "balanced",
            "price_in": 0.6,
            "price_out": 2.4,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "minimax-m3",
            "provider": "llmgateway",
            "tier": "budget",
            "price_in": 0.3,
            "price_out": 1.2,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "deepseek-v3.1",
            "provider": "llmgateway",
            "tier": "balanced",
            "price_in": 0.56,
            "price_out": 1.68,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "deepseek-v3.2",
            "provider": "llmgateway",
            "tier": "budget",
            "price_in": 0.26,
            "price_out": 0.38,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "deepseek-v4-flash",
            "provider": "llmgateway",
            "tier": "budget",
            "price_in": 0.14,
            "price_out": 0.28,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "deepseek-v4-pro",
            "provider": "llmgateway",
            "tier": "budget",
            "price_in": 0.43,
            "price_out": 0.87,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "mimo-v2.5",
            "provider": "llmgateway",
            "tier": "budget",
            "price_in": 0.14,
            "price_out": 0.28,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "mimo-v2.5-pro",
            "provider": "llmgateway",
            "tier": "budget",
            "price_in": 0.43,
            "price_out": 0.87,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "seed-1-6-250615",
            "provider": "llmgateway",
            "tier": "budget",
            "price_in": 0.25,
            "price_out": 2.0,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "seed-1-6-250915",
            "provider": "llmgateway",
            "tier": "budget",
            "price_in": 0.25,
            "price_out": 2.0,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "seed-1-6-flash-250715",
            "provider": "llmgateway",
            "tier": "budget",
            "price_in": 0.07,
            "price_out": 0.3,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "seed-1-8-251228",
            "provider": "llmgateway",
            "tier": "budget",
            "price_in": 0.25,
            "price_out": 2.0,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "fugu-ultra",
            "provider": "llmgateway",
            "tier": "expensive",
            "price_in": 5.0,
            "price_out": 30.0,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "nemotron-3-ultra-550b",
            "provider": "llmgateway",
            "tier": "balanced",
            "price_in": 0.5,
            "price_out": 2.5,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
        {
            "id": "gemma-4-31b-it",
            "provider": "llmgateway",
            "tier": "budget",
            "price_in": 0.1,
            "price_out": 0.3,
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        },
    ],
    # OpenCode Go — https://opencode.ai/zen/go/v1/models
    # Prices are per 1M tokens as provided by the user's pricing table.
    # base_url points to the OpenCode Go v1 endpoint; api_key_env resolves to the
    # user's OpenCode account key.  All models are OpenAI-compatible.
    "opencodego": [
        {
            "id": "grok-4.5",
            "provider": "opencodego",
            "tier": "balanced",
            "price_in": 2.00,
            "price_out": 6.00,
            "api_key_env": "OPENCODE_API_KEY",
            "base_url": "https://opencode.ai/zen/go/v1",
        },
        {
            "id": "gpt-5.6-luna",
            "provider": "opencodego",
            "tier": "budget",
            "price_in": 0.20,
            "price_out": 1.20,
            "api_key_env": "OPENCODE_API_KEY",
            "base_url": "https://opencode.ai/zen/go/v1",
        },
        {
            "id": "glm-5.2",
            "provider": "opencodego",
            "tier": "balanced",
            "price_in": 1.40,
            "price_out": 4.40,
            "api_key_env": "OPENCODE_API_KEY",
            "base_url": "https://opencode.ai/zen/go/v1",
        },
        {
            "id": "glm-5.1",
            "provider": "opencodego",
            "tier": "balanced",
            "price_in": 1.40,
            "price_out": 4.40,
            "api_key_env": "OPENCODE_API_KEY",
            "base_url": "https://opencode.ai/zen/go/v1",
        },
        {
            "id": "kimi-k3",
            "provider": "opencodego",
            "tier": "expensive",
            "price_in": 3.00,
            "price_out": 15.00,
            "api_key_env": "OPENCODE_API_KEY",
            "base_url": "https://opencode.ai/zen/go/v1",
        },
        {
            "id": "kimi-k2.7-code",
            "provider": "opencodego",
            "tier": "balanced",
            "price_in": 0.95,
            "price_out": 4.00,
            "api_key_env": "OPENCODE_API_KEY",
            "base_url": "https://opencode.ai/zen/go/v1",
        },
        {
            "id": "kimi-k2.6",
            "provider": "opencodego",
            "tier": "budget",
            "price_in": 0.95,
            "price_out": 4.00,
            "api_key_env": "OPENCODE_API_KEY",
            "base_url": "https://opencode.ai/zen/go/v1",
        },
        {
            "id": "mimo-v2.5",
            "provider": "opencodego",
            "tier": "budget",
            "price_in": 0.14,
            "price_out": 0.28,
            "api_key_env": "OPENCODE_API_KEY",
            "base_url": "https://opencode.ai/zen/go/v1",
        },
        {
            "id": "mimo-v2.5-pro",
            "provider": "opencodego",
            "tier": "budget",
            "price_in": 0.435,
            "price_out": 0.87,
            "api_key_env": "OPENCODE_API_KEY",
            "base_url": "https://opencode.ai/zen/go/v1",
        },
        {
            "id": "minimax-m3",
            "provider": "opencodego",
            "tier": "budget",
            "price_in": 0.30,
            "price_out": 1.20,
            "api_key_env": "OPENCODE_API_KEY",
            "base_url": "https://opencode.ai/zen/go/v1",
        },
        {
            "id": "minimax-m2.7",
            "provider": "opencodego",
            "tier": "budget",
            "price_in": 0.30,
            "price_out": 1.20,
            "api_key_env": "OPENCODE_API_KEY",
            "base_url": "https://opencode.ai/zen/go/v1",
        },
        {
            "id": "minimax-m2.5",
            "provider": "opencodego",
            "tier": "budget",
            "price_in": 0.30,
            "price_out": 1.20,
            "api_key_env": "OPENCODE_API_KEY",
            "base_url": "https://opencode.ai/zen/go/v1",
        },
        {
            "id": "qwen3.8-max",
            "provider": "opencodego",
            "tier": "balanced",
            "price_in": 2.00,
            "price_out": 6.00,
            "api_key_env": "OPENCODE_API_KEY",
            "base_url": "https://opencode.ai/zen/go/v1",
        },
        {
            "id": "qwen3.7-max",
            "provider": "opencodego",
            "tier": "balanced",
            "price_in": 2.50,
            "price_out": 7.50,
            "api_key_env": "OPENCODE_API_KEY",
            "base_url": "https://opencode.ai/zen/go/v1",
        },
        {
            "id": "qwen3.7-plus",
            "provider": "opencodego",
            "tier": "budget",
            "price_in": 0.40,
            "price_out": 1.60,
            "api_key_env": "OPENCODE_API_KEY",
            "base_url": "https://opencode.ai/zen/go/v1",
        },
        {
            "id": "qwen3.6-plus",
            "provider": "opencodego",
            "tier": "budget",
            "price_in": 0.50,
            "price_out": 3.00,
            "api_key_env": "OPENCODE_API_KEY",
            "base_url": "https://opencode.ai/zen/go/v1",
        },
        {
            "id": "deepseek-v4-pro",
            "provider": "opencodego",
            "tier": "budget",
            "price_in": 0.435,
            "price_out": 0.87,
            "api_key_env": "OPENCODE_API_KEY",
            "base_url": "https://opencode.ai/zen/go/v1",
        },
        {
            "id": "deepseek-v4-flash",
            "provider": "opencodego",
            "tier": "budget",
            "price_in": 0.14,
            "price_out": 0.28,
            "api_key_env": "OPENCODE_API_KEY",
            "base_url": "https://opencode.ai/zen/go/v1",
        },
        {
            "id": "hy3",
            "provider": "opencodego",
            "tier": "budget",
            "price_in": 0.14,
            "price_out": 0.58,
            "api_key_env": "OPENCODE_API_KEY",
            "base_url": "https://opencode.ai/zen/go/v1",
        },
        {
            "id": "glm-5",
            "provider": "opencodego",
            "tier": "balanced",
            "price_in": 0.57,
            "price_out": 2.58,
            "api_key_env": "OPENCODE_API_KEY",
            "base_url": "https://opencode.ai/zen/go/v1",
        },
        {
            "id": "kimi-k2.5",
            "provider": "opencodego",
            "tier": "budget",
            "price_in": 0.4,
            "price_out": 1.98,
            "api_key_env": "OPENCODE_API_KEY",
            "base_url": "https://opencode.ai/zen/go/v1",
        },
    ],
}

PRESET_ORDER = ["custom", "openai", "anthropic", "google", "mistral", "llmgateway", "opencodego"]


def default_preset_models(key: str) -> list[dict[str, Any]]:
    return list(PRESETS.get(key, []))


def _load_fallback() -> dict[str, dict]:
    if FALLBACK_PATH.exists():
        try:
            return json.loads(FALLBACK_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


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


def curated_model_catalog() -> list[dict[str, Any]]:
    """Return deduplicated chat models with prices in USD per million tokens."""
    global _catalog_cache
    if _catalog_cache is not None:
        return [dict(row) for row in _catalog_cache]

    rows: dict[str, dict[str, Any]] = {}
    # _ingest_litellm_costs is lazy and cached; its values are per 1M tokens.
    costs = _ingest_litellm_costs()
    try:
        import litellm  # type: ignore

        source = getattr(litellm, "model_cost", {}) or {}
    except Exception:
        source = {}
    for model_id, values in costs.items():
        raw = source.get(model_id, {})
        if not isinstance(raw, dict) or "output_cost_per_token" not in raw:
            continue
        if (
            any(key in raw for key in ("mode", "image_generation", "embedding"))
            and raw.get("mode") != "chat"
        ):
            continue
        bare = clean_model_id(model_id)
        candidate = {
            "id": bare,
            "provider": _catalog_provider(model_id, raw),
            "price_in": values.get("price_in", 0),
            "price_out": values.get("price_out", 0),
        }
        # Prefer an unqualified model over provider-qualified aliases.
        if bare not in rows or _catalog_id_is_unqualified(model_id):
            rows[bare] = candidate

    fallback = _load_fallback()
    for group, preset_rows in PRESETS.items():
        for row in preset_rows:
            rows.setdefault(
                row["id"],
                {
                    "id": row["id"],
                    "provider": row.get("provider", group),
                    "price_in": row.get("price_in", 0),
                    "price_out": row.get("price_out", 0),
                },
            )
    for model_id, row in fallback.items():
        if isinstance(row, dict):
            rows.setdefault(
                model_id,
                {
                    "id": model_id,
                    "provider": row.get("provider", "openai"),
                    "price_in": float(row.get("price_in", 0)),
                    "price_out": float(row.get("price_out", 0)),
                },
            )
    _catalog_cache = sorted(rows.values(), key=lambda row: (row["provider"], row["id"]))
    return [dict(row) for row in _catalog_cache]


def catalog_for_provider(provider: str) -> list[dict[str, Any]]:
    return [
        row
        for row in curated_model_catalog()
        if row["provider"].lower() == provider.lower()
    ]


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
    costs_by_clean = {
        clean_model_id(key): values for key, values in litellm_costs.items()
    }

    keys = list(PRESETS) if preset_keys is None else preset_keys
    for k in keys:
        raw_models = (
            custom_models
            if k == "custom"
            else (overrides[k] if overrides and k in overrides else PRESETS.get(k, []))
        )
        for raw_entry in raw_models or []:
            raw = (
                raw_entry.model_dump(mode="python")
                if isinstance(raw_entry, ModelEntry)
                else raw_entry
            )
            mid = raw["id"]
            # enrich price from litellm if available
            price_in = raw.get("price_in")
            price_out = raw.get("price_out")
            cost = costs_by_clean.get(clean_model_id(str(mid)))
            if cost:
                price_in = cost.get("price_in", price_in)
                price_out = cost.get("price_out", price_out)
            elif mid in fallback:
                price_in = fallback[mid].get("price_in", price_in)
                price_out = fallback[mid].get("price_out", price_out)
            entries.append(
                ModelEntry(
                    id=mid,
                    provider=raw.get("provider", "openai"),
                    api_key_env=raw.get("api_key_env", "OPENAI_API_KEY"),
                    api_key=raw.get("api_key"),
                    base_url=raw.get("base_url"),
                    tier=raw.get("tier", "balanced"),
                    price_in=float(price_in or 0),
                    price_out=float(price_out or 0),
                    enabled=bool(raw.get("enabled", True)),
                )
            )

    for raw in [] if "custom" in keys else (custom_models or []):
        mid = raw.get("id") or raw.get("model")
        if not mid:
            continue
        # price lookup
        pi = raw.get("price_in")
        po = raw.get("price_out")
        cost = costs_by_clean.get(clean_model_id(str(mid)))
        if cost and pi is None:
            pi = cost.get("price_in")
            po = cost.get("price_out")
        entries.append(
            ModelEntry(
                id=str(mid),
                provider=str(raw.get("provider", "openai")),
                api_key_env=str(raw.get("api_key_env", "OPENAI_API_KEY")),
                api_key=raw.get("api_key"),
                base_url=raw.get("base_url"),
                tier=str(raw.get("tier", "balanced")),
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


def resolve_models(cfg: Any, use_litellm: bool = True) -> list[ModelEntry]:
    models = discover_models(
        getattr(cfg, "selected_presets", []),
        getattr(cfg, "custom_models", []),
        use_litellm=use_litellm,
        overrides=getattr(cfg, "preset_overrides", {}),
    )
    cfg.model_list = [model.model_dump() for model in models]
    return models
