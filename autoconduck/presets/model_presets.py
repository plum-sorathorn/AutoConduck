from __future__ import annotations

from typing import Any

from autoconduck.config import ModelEntry

from . import presets_ingest as _ingest_module
from .presets_data import *
from .presets_ingest import (
    _catalog_id_is_unqualified,
    _catalog_provider,
    _load_fallback,
    clean_model_id,
    normalize_model_id_for_provider,
)

_litellm_costs_cache: dict[str, dict] | None = None
_catalog_cache: list[dict[str, Any]] | None = None

# This module remains the compatibility facade for the pre-split public API.
# In particular, MODEL_IDS and DEFAULT_MODELS intentionally come from the
# data module rather than being copied here.
__all__ = [
    "PRESETS",
    "MODEL_IDS",
    "DEFAULT_MODELS",
    "default_preset_models",
    "_load_fallback",
    "_ingest_litellm_costs",
    "_catalog_provider",
    "clean_model_id",
    "normalize_model_id_for_provider",
    "_catalog_id_is_unqualified",
    "curated_model_catalog",
    "catalog_for_provider",
    "discover_models",
    "normalize_entries",
    "resolve_models",
]


def _ingest_litellm_costs(enrich_pricing: bool = True) -> dict[str, dict]:
    global _litellm_costs_cache
    if _litellm_costs_cache is None:
        _ingest_module._litellm_costs_cache = None
    result = _ingest_module._ingest_litellm_costs(enrich_pricing)
    _litellm_costs_cache = result
    return result


def default_preset_models(key: str) -> list[dict[str, Any]]:
    return list(PRESETS.get(key, []))


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
            mid = normalize_model_id_for_provider(
                str(raw["id"]), str(raw.get("provider", "openai"))
            )
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
                    price_in=float(price_in or 0),
                    price_out=float(price_out or 0),
                    enabled=bool(raw.get("enabled", True)),
                )
            )

    for raw in [] if "custom" in keys else (custom_models or []):
        mid = raw.get("id") or raw.get("model")
        if not mid:
            continue
        mid = normalize_model_id_for_provider(
            str(mid), str(raw.get("provider", "openai"))
        )
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
