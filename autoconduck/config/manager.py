"""Configuration file I/O, validation, caching, and backups."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any
import yaml

from autoconduck.config.models import Config
from autoconduck.config.paths import backups_dir, config_path
from autoconduck.config.resolver import (
    _configured_model_sources,
    _normalize_model_entries,
    resolve_api_key,
)

_config: Config | None = None
_config_digest: bytes | None = None
_config_path: Path | None = None


def load_config(path: str | Path | None = None) -> Config:
    """Load configuration from YAML file and apply environment variable overrides."""
    p = config_path(path)
    data = {}
    if p.exists():
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if "AUTOCONDUCK_PORT" in os.environ:
        data["port"] = int(os.environ["AUTOCONDUCK_PORT"])
    if "AUTOCONDUCK_LOG_LEVEL" in os.environ:
        data["log_level"] = os.environ["AUTOCONDUCK_LOG_LEVEL"]
    config = Config(**data)
    _normalize_model_entries(
        {
            "custom_models": config.custom_models,
            "model_list": config.model_list,
        }
    )
    if any(
        isinstance(e, dict) and e.get("api_key")
        for s in (config.model_list, config.custom_models)
        for e in s
    ):
        from autoconduck.auth.auth import migrate_from_config

        migrate_from_config(config)
    for entry in _configured_model_sources(config):
        if (
            isinstance(entry, dict)
            and entry.get("enabled", True)
            and not resolve_api_key(entry)
        ):
            logging.getLogger("autoconduck").warning(
                "No API key is configured for model %s (set auth.yaml or api_key_env)",
                entry.get("id") or entry.get("model_name") or "<unknown>",
            )
    if not any(
        isinstance(entry, dict) and entry.get("enabled", True)
        for entry in _configured_model_sources(config)
    ):
        logging.getLogger("autoconduck").warning(
            "No models are configured in %s - add a preset or model_list or every request will fall back to a hardcoded default and may fail auth.",
            p,
        )
    if getattr(getattr(config, "selection", None), "phase_bands", None):
        validate_phase_bands(config)
    return config


def validate_phase_bands(config: Config) -> list[str]:
    """Validate configured phase bands against available models in the pool if explicitly specified."""
    warnings: list[str] = []
    try:
        from autoconduck.routing.pricing import pool_ids, scaled_cost

        models = pool_ids(config)
        if not models:
            return warnings
        phase_bands = getattr(
            getattr(config, "selection", None), "phase_bands", None
        )
        if not phase_bands:
            return warnings
        costs = {m: scaled_cost(m, config) for m in models}
        for name, band in phase_bands.items():
            if not isinstance(band, (list, tuple)) or len(band) < 2:
                continue
            lo, hi = float(band[0]), float(band[1])
            in_band = [m for m, c in costs.items() if lo <= c <= hi]
            if not in_band:
                cost_summary = ", ".join(
                    f"{m}: {c:.2f}"
                    for m, c in sorted(costs.items(), key=lambda x: x[1])
                )
                msg = (
                    f"Phase band '{name}' [{lo:.2f}, {hi:.2f}] contains 0 configured models "
                    f"(available pool scaled costs: {cost_summary}). "
                    f"Orchestration will fall back to closest available model."
                )
                warnings.append(msg)
                logging.getLogger("autoconduck").warning(msg)
    except Exception:
        pass
    return warnings


def get_config() -> Config:
    """Return cached singleton Config instance, reloading if modified on disk."""
    global _config, _config_digest, _config_path
    path = config_path().resolve()
    try:
        digest = path.read_bytes()
    except FileNotFoundError:
        digest = None
    if _config is None or _config_path != path or _config_digest != digest:
        _config = load_config(path)
        _config_path = path
        _config_digest = digest
    return _config


def save_config(cfg: Config, path: str | Path | None = None) -> None:
    """Save configuration to disk with atomic replacement."""
    global _config, _config_digest, _config_path
    p = config_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = cfg.model_dump()
    _normalize_model_entries(data)
    temp_path = p.with_name(f"{p.name}.{os.getpid()}.tmp")
    try:
        temp_path.write_text(yaml.safe_dump(data), encoding="utf-8")
        temp_path.replace(p)
    except Exception:
        p.write_text(yaml.safe_dump(data), encoding="utf-8")
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
    _config = None
    _config_digest = None
    _config_path = None


def backup_config(path: str | Path | None = None) -> Path | None:
    """Make a plain, timestamped backup of config.yaml before managed edits."""
    source = config_path(path)
    if not source.exists():
        return None

    import datetime

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    target = backups_dir("config") / f"{stamp}.bak"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target
