from __future__ import annotations

import os
import yaml
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

DEFAULT_PORT = 11434
CONFIG_VERSION = 1

Tier = Literal["budget", "balanced", "expensive", "reasoning"]


class ModelEntry(BaseModel):
    id: str
    provider: str = "openai"
    api_key_env: str = Field(default="OPENAI_API_KEY", description="env var name, never raw key")
    tier: Tier = "balanced"
    price_in: float = Field(default=0.001, description="per 1K tokens input")
    price_out: float = Field(default=0.002, description="per 1K tokens output")
    enabled: bool = True
    context_window: int = 8192
    supports_streaming: bool = True

    @field_validator("price_in", "price_out")
    @classmethod
    def _non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("price must be >=0")
        return v


class PseudoModelConfig(BaseModel):
    transform: str = "x"


class Config(BaseModel):
    version: int = CONFIG_VERSION
    port: int = DEFAULT_PORT
    cache_enabled: bool = False
    log_level: Literal["debug", "info", "warning", "error"] = "info"
    backups_retention: int = 5
    models: list[ModelEntry] = Field(default_factory=list)
    pseudo_models: dict[str, PseudoModelConfig] = Field(
        default_factory=lambda: {
            "autoconduck": PseudoModelConfig(transform="x"),
            "autoconduck-budget": PseudoModelConfig(transform="x*0.6"),
            "autoconduck-expensive": PseudoModelConfig(transform="min(1,x*1.4+0.1)"),
        }
    )
    # intent -> predicted output tokens
    intent_tokens: dict[str, int] = Field(
        default_factory=lambda: {
            "fix": 400,
            "format": 200,
            "refactor": 2500,
            "architecture": 3500,
            "default": 800,
            "build_feature": 3000,
            "migrate": 3200,
        }
    )
    max_workers: int = 4
    max_in_flight: int = 32


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def home_dir() -> Path:
    return Path(os.environ.get("AUTOCONDUCK_HOME", Path.home() / ".autoconduck"))


def config_path() -> Path:
    return home_dir() / "config.yaml"


def state_path() -> Path:
    return home_dir() / "state.json"


def logs_path() -> Path:
    return home_dir() / "logs" / "routing.jsonl"


def backups_dir(agent: str) -> Path:
    return home_dir() / "backups" / agent


def ensure_home() -> None:
    home_dir().mkdir(parents=True, exist_ok=True)
    (home_dir() / "backups").mkdir(parents=True, exist_ok=True)
    (home_dir() / "logs").mkdir(parents=True, exist_ok=True)
    (home_dir() / "cache").mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_config_singleton: Config | None = None


def get_config(reload: bool = False) -> Config:
    global _config_singleton
    if _config_singleton is None or reload:
        _config_singleton = load_config()
    return _config_singleton


def load_config(path: Path | None = None) -> Config:
    p = path or config_path()
    # file layer
    data: dict = {}
    if p.exists():
        try:
            raw = p.read_text(encoding="utf-8")
            data = yaml.safe_load(raw) or {}
        except Exception as e:
            raise ValueError(f"config corrupt at {p}: {e}") from e
    # env overrides (AUTOCONDUCK_PORT, AUTOCONDUCK_CACHE_ENABLED, etc.)
    env_port = os.environ.get("AUTOCONDUCK_PORT")
    if env_port is not None:
        try:
            data["port"] = int(env_port)
        except ValueError:
            pass
    env_cache = os.environ.get("AUTOCONDUCK_CACHE_ENABLED")
    if env_cache is not None:
        data["cache_enabled"] = env_cache.lower() in ("1", "true", "yes")
    env_log = os.environ.get("AUTOCONDUCK_LOG_LEVEL")
    if env_log:
        data["log_level"] = env_log.lower()

    if not data:
        # return defaults (onboarding will populate models)
        return Config()

    # Validate via pydantic; allow partial
    try:
        return Config.model_validate(data)
    except Exception as e:
        raise ValueError(f"config validation failed: {e}") from e


def save_config(cfg: Config, path: Path | None = None) -> Path:
    ensure_home()
    p = path or config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    # never write raw api keys; ModelEntry already holds env var names
    data = cfg.model_dump(mode="python")
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    global _config_singleton
    _config_singleton = cfg
    return p


def apply_cli_overrides(cfg: Config, port: int | None = None, cache_enabled: bool | None = None) -> Config:
    if port is not None:
        cfg.port = port
    if cache_enabled is not None:
        cfg.cache_enabled = cache_enabled
    return cfg
