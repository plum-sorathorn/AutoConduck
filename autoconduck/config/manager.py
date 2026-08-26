"""Configuration file I/O, validation, caching, and backups."""

from __future__ import annotations

import datetime
import logging
import os
import shutil
import threading
import time
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

_config_lock = threading.RLock()
_config: Config | None = None
_config_digest: bytes | None = None
_config_path: Path | None = None


def _has_configured_models(cfg: Config | None) -> bool:
    """Return True if the config has at least one active configured model entry."""
    if cfg is None:
        return False
    return any(
        isinstance(entry, dict) and entry.get("enabled", True)
        for entry in _configured_model_sources(cfg)
    )


def _find_latest_backup(path: str | Path | None = None) -> Path | None:
    """Find the most recent non-empty backup of config.yaml."""
    bdir = backups_dir("config")
    if not bdir.exists():
        return None
    baks = sorted(bdir.glob("*.bak"), reverse=True)
    for b in baks:
        if b.is_file() and b.stat().st_size > 0:
            return b
    return None


def load_config(
    path: str | Path | None = None,
    fallback_to_backup: bool = True,
) -> Config:
    """Load configuration from YAML file and apply environment variable overrides."""
    p = config_path(path)
    data: dict[str, Any] = {}
    read_success = False

    if p.exists():
        for attempt in range(3):
            try:
                raw = p.read_text(encoding="utf-8")
                if raw.strip():
                    parsed = yaml.safe_load(raw)
                    if isinstance(parsed, dict):
                        data = parsed
                        read_success = True
                        break
            except Exception as exc:
                if attempt < 2:
                    time.sleep(0.02 * (attempt + 1))
                else:
                    logging.getLogger("autoconduck").warning(
                        "Failed reading config from %s: %s", p, exc
                    )

    # If the file is missing/empty/corrupted, attempt recovery from latest valid backup
    if not read_success and fallback_to_backup and (path is None or Path(path) == config_path()):
        latest_bak = _find_latest_backup(path)
        if latest_bak:
            try:
                raw_bak = latest_bak.read_text(encoding="utf-8")
                parsed_bak = yaml.safe_load(raw_bak)
                if isinstance(parsed_bak, dict) and parsed_bak:
                    logging.getLogger("autoconduck").warning(
                        "Config file %s was missing or empty; restored configuration from backup %s",
                        p,
                        latest_bak,
                    )
                    data = parsed_bak
                    read_success = True
                    try:
                        p.parent.mkdir(parents=True, exist_ok=True)
                        p.write_text(raw_bak, encoding="utf-8")
                    except Exception:
                        pass
            except Exception:
                pass

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

        migrate_from_config(config, path=p)
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
    if not _has_configured_models(config):
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
    with _config_lock:
        path = config_path().resolve()
        digest: bytes | None = None
        for attempt in range(3):
            try:
                if path.exists():
                    digest = path.read_bytes()
                break
            except Exception:
                if attempt < 2:
                    time.sleep(0.02 * (attempt + 1))

        # If disk read returned empty digest/None but we already have an active valid config:
        # Preserve the active in-memory configuration to prevent transient drops or defaults.
        if (digest is None or len(digest.strip()) == 0) and _has_configured_models(_config) and _config_path == path:
            logging.getLogger("autoconduck").warning(
                "Config file %s temporarily unavailable or empty; retaining active in-memory configuration",
                path,
            )
            return _config

        if _config is None or _config_path != path or _config_digest != digest:
            new_config = load_config(path)
            # If the reload yielded 0 models but the previously loaded in-memory config had models
            # and the file was unreadable or empty, retain the active configuration.
            if not _has_configured_models(new_config) and _has_configured_models(_config) and (digest is None or len(digest.strip()) == 0):
                logging.getLogger("autoconduck").warning(
                    "Config reloaded from %s yielded no models; retaining active in-memory configuration",
                    path,
                )
                return _config
            _config = new_config
            _config_path = path
            _config_digest = digest
        return _config


def save_config(cfg: Config, path: str | Path | None = None, force_empty: bool = False) -> None:
    """Save configuration to disk with atomic replacement, backup, and lock."""
    global _config, _config_digest, _config_path
    with _config_lock:
        p = config_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists() and p.stat().st_size > 0:
            if not _has_configured_models(cfg) and not force_empty:
                try:
                    existing_raw = p.read_text(encoding="utf-8")
                    existing_parsed = yaml.safe_load(existing_raw)
                    if isinstance(existing_parsed, dict):
                        existing_cfg = Config(**existing_parsed)
                        if _has_configured_models(existing_cfg):
                            logging.getLogger("autoconduck").warning(
                                "Refusing to overwrite valid config containing models with an empty model list at %s",
                                p,
                            )
                            return
                except Exception:
                    pass
            backup_config(p)

        data = cfg.model_dump()
        _normalize_model_entries(data)
        serialized = yaml.safe_dump(data)
        raw_bytes = serialized.encode("utf-8")

        temp_path = p.with_name(
            f"{p.name}.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.tmp"
        )
        saved = False
        try:
            temp_path.write_text(serialized, encoding="utf-8")
            for attempt in range(5):
                try:
                    temp_path.replace(p)
                    saved = True
                    break
                except OSError:
                    time.sleep(0.03 * (attempt + 1))
        except Exception:
            pass

        if not saved:
            try:
                p.write_text(serialized, encoding="utf-8")
                saved = True
            except Exception as exc:
                logging.getLogger("autoconduck").error("Failed to save config to %s: %s", p, exc)
            finally:
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except OSError:
                        pass

        _config = cfg
        _config_digest = raw_bytes
        _config_path = p.resolve()


def backup_config(path: str | Path | None = None) -> Path | None:
    """Make a plain, timestamped backup of config.yaml before managed edits."""
    source = config_path(path)
    if not source.exists() or source.stat().st_size == 0:
        return None

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    target = backups_dir("config") / f"{stamp}.bak"
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(source, target)
        return target
    except Exception:
        return None
