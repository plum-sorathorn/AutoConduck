"""Sync the curated DevPass preset from https://devpass.llmgateway.io/models."""
import urllib.request
import re
import json
from pathlib import Path

from autoconduck.model_presets import _ingest_litellm_costs, clean_model_id

def fetch_devpass_catalog() -> list[dict]:
    all_models = {}
    page = 1
    while True:
        url = f"https://devpass.llmgateway.io/models?page={page}" if page > 1 else "https://devpass.llmgateway.io/models"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "autoconduck/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                text = resp.read().decode("utf-8")
        except Exception as e:
            print(f"Error fetching page {page}: {e}")
            break

        # Match model records from the embedded page state
        model_matches = re.findall(
            r'\\?"id\\?":\\?"([a-zA-Z0-9.\-_]+)\\?",\\?"createdAt\\?":.*?\\?"name\\?":\\?"([^"\\]+)\\?".*?\\?"premium\\?":(true|false)',
            text,
        )
        page_models = 0
        for mid, name, premium in model_matches:
            # Skip random 20-character database IDs
            if len(mid) == 20 and mid.isalnum() and not any(c in mid for c in ".-_"):
                continue
            if mid not in all_models:
                all_models[mid] = {
                    "id": mid,
                    "name": name,
                    "premium": premium == "true",
                }
                page_models += 1

        if f'href="/models?page={page + 1}"' not in text and f'href=\\"/models?page={page + 1}\\"' not in text:
            break
        page += 1
        if page > 25:
            break

    costs = _ingest_litellm_costs()
    by_clean = {clean_model_id(k): v for k, v in costs.items()}

    devpass_entries = []
    for mid, info in sorted(all_models.items()):
        # Filter out non-chat / media / embedding models
        if any(x in mid for x in ("embedding", "image", "video", "tts", "stt", "transcribe", "reranker", "audio")):
            continue
        prices = by_clean.get(clean_model_id(mid), {})
        p_in = prices.get("price_in", 0.0)
        p_out = prices.get("price_out", 0.0)
        tier = "expensive" if info["premium"] or p_out >= 20.0 else "budget" if p_out < 3.0 else "balanced"
        devpass_entries.append({
            "id": mid,
            "provider": "devpass",
            "tier": tier,
            "price_in": p_in,
            "price_out": p_out,
            "api_key_env": "DEVPASS_API_KEY",
            "base_url": "https://api.llmgateway.io",
        })

    return devpass_entries

if __name__ == "__main__":
    entries = fetch_devpass_catalog()
    print(f"Successfully synced {len(entries)} DevPass models from https://devpass.llmgateway.io/models")
    print("Sample models:")
    for e in entries[:10]:
        print(" ", e)
