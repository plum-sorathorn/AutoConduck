"""Model qualification, endpoint resolution, and API key management."""

from __future__ import annotations

import logging
import os
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from autoconduck.config.models import ModelEntry

_legacy_key_warning = False


def provider_for(entry: dict[str, Any], cfg: Any = None) -> str:
    """Derive the stable provider identity used by auth.yaml."""
    try:
        if entry.get("provider"):
            return str(entry["provider"])
        base = entry.get("api_base") or entry.get("base_url")
        for endpoint in getattr(cfg, "custom_models", []) or []:
            if base and base == endpoint.get("base_url"):
                return str(endpoint.get("provider") or endpoint.get("display_name"))
        raw = entry.get("id") or entry.get("model_name") or entry.get("model")
        params = entry.get("litellm_params")
        if not raw and isinstance(params, dict):
            raw = params.get("model")
        qualified = qualify_model(str(raw or ""))
        return qualified.split("/", 1)[0] if "/" in qualified else "openai"
    except Exception:
        return "openai"


def resolve_api_key(entry: dict[str, Any], provider: str | None = None) -> str:
    """Resolve the API key for a model entry from auth.yaml, direct entry, or env."""
    global _legacy_key_warning
    try:
        from autoconduck.auth.auth import get_provider_key

        auth_key = get_provider_key(provider or provider_for(entry))
        if auth_key is not None:
            return auth_key
    except Exception:
        pass
    if entry.get("api_key"):
        if not _legacy_key_warning:
            logging.getLogger("autoconduck").warning(
                "Literal API keys in config.yaml are deprecated; use auth.yaml"
            )
            _legacy_key_warning = True
        return str(entry["api_key"])
    name = entry.get("api_key_env")
    if name:
        return os.environ.get(name, "")
    return ""


def qualify_model(model_id: str) -> str:
    """Return a LiteLLM provider-qualified model name."""
    value = str(model_id or "")
    if "/" in value:
        provider = value.split("/", 1)[0]
        try:
            from litellm import provider_list

            if provider in provider_list:
                return value
        except Exception:
            pass
    return value if value.startswith("openai/") else f"openai/{value}"


def normalize_api_base(base_url: str) -> str:
    """Return an OpenAI-compatible endpoint URL with the required /v1 path."""
    value = str(base_url or "").rstrip("/")
    value = _repair_base_url_scheme(value)
    if not value:
        return value
    parts = urlsplit(value)
    if parts.path.rstrip("/").split("/")[-1].lower() == "v1":
        return value
    return urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            parts.path.rstrip("/") + "/v1",
            parts.query,
            parts.fragment,
        )
    )


def _repair_base_url_scheme(base_url: str) -> str:
    """Repair a malformed/missing URL scheme so base URLs are always usable."""
    value = str(base_url or "").strip()
    if not value:
        return value
    if "://" in value:
        scheme, _, rest = value.partition("://")
        if scheme.lower() == "ttps":
            return "https://" + rest
        if scheme.lower() in ("http", "https"):
            return value
        return value
    return "https://" + value


def _normalize_model_entries(config_dict: dict[str, Any]) -> None:
    for field in ("custom_models", "model_list"):
        for entry in config_dict.get(field, []) or []:
            if not isinstance(entry, dict):
                continue
            for key in ("base_url", "api_base"):
                if entry.get(key) is not None:
                    entry[key] = _repair_base_url_scheme(entry[key])


def _configured_model_sources(cfg: Any):
    """Yield model pools in precedence order, including Pi's optional pool."""
    if hasattr(cfg, "models") and isinstance(cfg.models, dict) and cfg.models:
        for entry in cfg.models.values():
            yield entry.model_dump() if isinstance(entry, ModelEntry) else entry
    yield from (getattr(cfg, "model_list", []) or [])
    yield from (getattr(cfg, "custom_models", []) or [])
    pi = getattr(cfg, "pi", None)
    if pi is not None and getattr(pi, "enabled", True):
        entries = getattr(pi, "model_entries", []) or []
        if entries:
            yield from (
                entry.model_dump() if isinstance(entry, ModelEntry) else entry
                for entry in entries
            )
        elif getattr(pi, "model", None):
            yield {
                "id": pi.model,
                "provider": pi.provider,
                "api_key_env": pi.api_key_env,
                "api_key": pi.api_key,
                "base_url": pi.base_url,
            }


def resolve_orchestrator_model(cfg: Any = None) -> str:
    """Select the first enabled configured model for orchestration calls."""
    if cfg is None:
        try:
            from autoconduck.config.manager import get_config

            cfg = get_config()
        except Exception:
            cfg = None
    for entry in _configured_model_sources(cfg):
        if not isinstance(entry, dict) or entry.get("enabled", True) is False:
            continue
        model = entry.get("id") or entry.get("model_name") or entry.get("model")
        if not model and isinstance(entry.get("litellm_params"), dict):
            model = entry["litellm_params"].get("model")
        if model:
            return str(model)
    try:
        from autoconduck.auth.auth import load_auth

        auth = load_auth()
        if auth:
            from autoconduck.presets.presets_fallback import _FALLBACK_PRESETS

            for prov in auth:
                presets = _FALLBACK_PRESETS.get(prov, [])
                for p in presets:
                    if isinstance(p, dict) and p.get("id"):
                        return str(p["id"])
    except Exception:
        pass
    return "gpt-4o"


def select_model_by_tier(tier: str, cfg: Any = None) -> str:
    """Select a configured model by tier."""
    try:
        from autoconduck.routing.pricing import select_for_tier

        return select_for_tier(tier, config=cfg) or resolve_orchestrator_model(cfg)
    except Exception:
        return resolve_orchestrator_model(cfg)


def orchestrator_litellm_params(cfg: Any = None) -> dict[str, str]:
    """Build LiteLLM kwargs for the configured orchestration model."""
    from autoconduck.server.messages_api import litellm_params_for
    from autoconduck.config.manager import get_config

    model = resolve_orchestrator_model(cfg)
    return litellm_params_for(model, cfg or get_config())
