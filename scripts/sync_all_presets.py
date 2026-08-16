"""Unified updater for all model presets and the catalog documentation."""
from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_LITELLM_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
DEVPASS_MODELS_URL = "https://devpass.llmgateway.io/models"
LLMGATEWAY_MODELS_URL = "https://api.llmgateway.io/v1/models"

ENV_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "groq": "GROQ_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "devpass": "DEVPASS_API_KEY",
    "llmgateway": "LLMGATEWAY_API_KEY",
}


def fetch_upstream_litellm_costs() -> dict[str, dict[str, Any]]:
    """Fetch the latest upstream LiteLLM model database."""
    try:
        req = urllib.request.Request(UPSTREAM_LITELLM_URL, headers={"User-Agent": "autoconduck/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            out = {}
            for k, v in data.items():
                if isinstance(v, dict) and "input_cost_per_token" in v:
                    out[k] = {
                        "price_in": float(v.get("input_cost_per_token", 0)) * 1_000_000,
                        "price_out": float(v.get("output_cost_per_token", 0)) * 1_000_000,
                        "provider": v.get("litellm_provider", "openai"),
                        "mode": v.get("mode", "chat"),
                    }
            return out
    except Exception as exc:
        print(f"Warning: could not fetch upstream LiteLLM database ({exc}); using installed package.")
        from autoconduck.model_presets import _ingest_litellm_costs
        return _ingest_litellm_costs()


def fetch_devpass_models(costs: dict[str, dict]) -> list[dict[str, Any]]:
    """Fetch models from https://devpass.llmgateway.io/models."""
    all_models = {}
    page = 1
    while True:
        url = f"{DEVPASS_MODELS_URL}?page={page}" if page > 1 else DEVPASS_MODELS_URL
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "autoconduck/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                text = resp.read().decode("utf-8")
        except Exception as e:
            print(f"Error fetching DevPass page {page}: {e}")
            break

        model_matches = re.findall(
            r'\\?"id\\?":\\?"([a-zA-Z0-9.\-_]+)\\?",\\?"createdAt\\?":.*?\\?"name\\?":\\?"([^"\\]+)\\?".*?\\?"premium\\?":(true|false)',
            text,
        )
        for mid, name, premium in model_matches:
            if len(mid) == 20 and mid.isalnum() and not any(c in mid for c in ".-_"):
                continue
            if mid not in all_models:
                all_models[mid] = {"id": mid, "name": name, "premium": premium == "true"}

        if f'href="/models?page={page + 1}"' not in text and f'href=\\"/models?page={page + 1}\\"' not in text:
            break
        page += 1
        if page > 25:
            break

    devpass_entries = []
    for mid, info in sorted(all_models.items()):
        if any(x in mid for x in ("embedding", "image", "video", "tts", "stt", "transcribe", "reranker", "audio")):
            continue
        prices = costs.get(mid, {})
        p_in = prices.get("price_in", 0.0)
        p_out = prices.get("price_out", 0.0)
        tier = "expensive" if info["premium"] or p_out >= 20.0 else "budget" if p_out < 3.0 else "balanced"
        devpass_entries.append({
            "id": mid,
            "provider": "devpass",
            "tier": tier,
            "price_in": round(p_in, 4),
            "price_out": round(p_out, 4),
            "api_key_env": "DEVPASS_API_KEY",
            "base_url": "https://api.llmgateway.io",
        })
    return devpass_entries


def fetch_llmgateway_models(costs: dict[str, dict]) -> list[dict[str, Any]]:
    """Fetch models from https://api.llmgateway.io/v1/models."""
    try:
        req = urllib.request.Request(LLMGATEWAY_MODELS_URL, headers={"User-Agent": "autoconduck/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"Error fetching LLMGateway models: {e}")
        return []

    entries = []
    for m in data.get("data", []):
        mid = m.get("id", "")
        if not mid or mid in ("custom", "auto"):
            continue
        if any(x in mid for x in ("embedding", "image", "video", "tts", "stt", "transcribe", "reranker", "audio")):
            continue
        prices = costs.get(mid, {})
        p_in = prices.get("price_in", 0.0)
        p_out = prices.get("price_out", 0.0)
        tier = "expensive" if p_out >= 20.0 else "budget" if p_out < 3.0 else "balanced"
        entries.append({
            "id": mid,
            "provider": "llmgateway",
            "tier": tier,
            "price_in": round(p_in, 4),
            "price_out": round(p_out, 4),
            "api_key_env": "LLMGATEWAY_API_KEY",
            "base_url": "https://api.llmgateway.io",
        })
    return sorted(entries, key=lambda x: x["id"])


def build_provider_presets(costs: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Curate top models for standard providers directly from the upstream registry."""
    # Preferred curated model shortlist per provider
    provider_curations = {
        "anthropic": [
            "claude-3-7-sonnet",
            "claude-3-5-sonnet",
            "claude-3-5-haiku",
            "claude-3-opus",
            "claude-sonnet-4-6",
            "claude-sonnet-5",
            "claude-opus-5",
            "claude-haiku-4-5",
        ],
        "openai": [
            "gpt-4.5-preview",
            "gpt-4o",
            "gpt-4o-mini",
            "o1",
            "o3-mini",
            "o4-mini",
            "gpt-5.6",
            "gpt-5.2",
            "gpt-5-mini",
            "gpt-5.1-codex",
            "gpt-4.1",
        ],
        "google": [
            "gemini-3.7-flash",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-3.5-flash",
            "gemini-3.5-flash-lite",
            "gemini-3.1-pro-preview",
        ],
        "mistral": [
            "codestral-2508",
            "devstral-2512",
            "mistral-large-latest",
            "mistral-small-latest",
            "ministral-8b-latest",
        ],
        "deepseek": [
            "deepseek-chat",
            "deepseek-reasoner",
            "deepseek-v3.2",
            "deepseek-v4-flash",
            "deepseek-v4-pro",
        ],
    }

    presets = {}
    for provider, candidates in provider_curations.items():
        rows = []
        for mid in candidates:
            info = costs.get(mid) or costs.get(f"{provider}/{mid}") or {}
            p_in = info.get("price_in", 0.0)
            p_out = info.get("price_out", 0.0)
            tier = "expensive" if p_out >= 20.0 else "budget" if p_out < 3.0 else "balanced"
            rows.append({
                "id": mid,
                "provider": provider,
                "tier": tier,
                "price_in": round(p_in, 4),
                "price_out": round(p_out, 4),
                "api_key_env": ENV_KEYS.get(provider, f"{provider.upper()}_API_KEY"),
            })
        presets[provider] = rows

    return presets


def sync_all():
    print("1. Fetching upstream LiteLLM model & pricing database...")
    costs = fetch_upstream_litellm_costs()
    print(f"   Fetched pricing for {len(costs)} models.")

    print("2. Building curated presets for Anthropic, OpenAI, Google, Mistral, DeepSeek...")
    presets = build_provider_presets(costs)

    print("3. Fetching DevPass models from https://devpass.llmgateway.io/models...")
    devpass_models = fetch_devpass_models(costs)
    presets["devpass"] = devpass_models
    print(f"   Synced {len(devpass_models)} DevPass models.")

    print("4. Fetching LLMGateway models from https://api.llmgateway.io/v1/models...")
    llmgateway_models = fetch_llmgateway_models(costs)
    presets["llmgateway"] = llmgateway_models
    print(f"   Synced {len(llmgateway_models)} LLMGateway models.")

    return presets


if __name__ == "__main__":
    presets = sync_all()
    print("\nSummary of synced presets:")
    for k, v in presets.items():
        print(f"  - {k}: {len(v)} models")
