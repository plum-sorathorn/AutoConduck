"""Provider credentials stored separately from the model configuration."""
from __future__ import annotations

import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from autoconduck.config import backups_dir, home_dir, provider_for

log = logging.getLogger("autoconduck")


def auth_path() -> Path:
    structured = home_dir() / "auth" / "auth.yaml"
    if structured.exists():
        return structured
    return home_dir() / "auth.yaml"


def load_auth() -> dict[str, str]:
    try:
        auth_fn = getattr(sys.modules.get("autoconduck.auth"), "auth_path", auth_path)
        if not auth_fn().exists():
            return {}
        auth_fn = getattr(sys.modules.get("autoconduck.auth"), "auth_path", auth_path)
        data = yaml.safe_load(auth_fn().read_text(encoding="utf-8")) or {}
        providers = data.get("providers", {}) if isinstance(data, dict) else {}
        return {str(k): str(v) for k, v in providers.items() if isinstance(v, (str, int, float))}
    except Exception as exc:
        log.warning("Unable to read auth file: %s", exc)
        return {}


def save_auth(mapping: dict[str, str]) -> None:
    auth_fn = getattr(sys.modules.get("autoconduck.auth"), "auth_path", auth_path)
    path = auth_fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"providers": mapping}, sort_keys=True), encoding="utf-8")
    if os.name != "nt":
        path.chmod(0o600)


def get_provider_key(provider: str) -> str | None:
    load_fn = getattr(sys.modules.get("autoconduck.auth"), "load_auth", load_auth)
    value = load_fn().get(provider)
    if value is None:
        return None
    if value.startswith("env:"):
        return os.environ.get(value[4:], "")
    return value


def set_provider_key(provider: str, value: str) -> None:
    load_fn = getattr(sys.modules.get("autoconduck.auth"), "load_auth", load_auth)
    mapping = load_fn()
    mapping[provider] = value
    save_auth(mapping)


def migrate_from_config(cfg: Any) -> int:
    """Move literal model keys to auth.yaml, preserving env references."""
    try:
        entries = [*getattr(cfg, "model_list", []), *getattr(cfg, "custom_models", [])]
        literals = [(entry, provider_for(entry, cfg)) for entry in entries
                    if isinstance(entry, dict) and entry.get("api_key")]
        if not literals:
            return 0
        load_fn = getattr(sys.modules.get("autoconduck.auth"), "load_auth", load_auth)
        mapping = load_fn()
        changed = False
        for entry, provider in literals:
            if provider not in mapping:
                mapping[provider] = str(entry["api_key"])
            if "api_key" in entry:
                del entry["api_key"]
                changed = True
        if mapping:
            save_auth(mapping)
        if changed:
            source = home_dir() / "config.yaml"
            if source.exists():
                backup = backups_dir("config")
                backup.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
                shutil.copy2(source, backup / f"{stamp}.bak")
            from autoconduck.config import save_config
            save_config(cfg)
        return len(literals)
    except Exception as exc:
        log.warning("Unable to migrate API keys to auth file: %s", exc)
        return 0
