import os
from pathlib import Path
from pydantic import BaseModel, Field
import yaml

class ModelEntry(BaseModel):
    id: str
    provider: str = "openai"
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None
    tier: str = "balanced"
    price_in: float = 0.0
    price_out: float = 0.0
    enabled: bool = True

class Config(BaseModel):
    host: str = "127.0.0.1"; port: int = 11434; log_level: str = "INFO"
    ambiguous_low: float = 0.55; ambiguous_high: float = 0.70; hysteresis_floor: float = 0.50; escalation_threshold: float = 0.80; stack_trace_boost: float = 0.25
    ema_alpha: float = 0.1; degraded_error_rate: float = 0.20; degraded_window_s: int = 300; pseudo_model: str = "autoconduck"
    model_list: list[dict] = Field(default_factory=list); routing_log: bool = True
    selected_presets: list[str] = Field(default_factory=list)
    custom_models: list[dict] = Field(default_factory=list)
    preset_overrides: dict[str, list[dict]] = Field(default_factory=dict)
    shims: dict[str, str] = Field(default_factory=dict)
    managed_server: bool = False
def home_dir() -> Path: return Path(os.environ.get("AUTOCONDUCK_HOME", Path.home() / ".autoconduck"))
def backups_dir(agent: str | None = None) -> Path:
    path = home_dir() / "backups"
    return path / agent if agent else path
def logs_path() -> Path: return home_dir() / "autoconduck.log"
def run_dir() -> Path: return home_dir() / "run"
def load_config(path=None) -> Config:
    p = Path(path) if path else home_dir() / "config.yaml"; data = {}
    if p.exists(): data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if "AUTOCONDUCK_PORT" in os.environ: data["port"] = int(os.environ["AUTOCONDUCK_PORT"])
    if "AUTOCONDUCK_LOG_LEVEL" in os.environ: data["log_level"] = os.environ["AUTOCONDUCK_LOG_LEVEL"]
    return Config(**data)
_config = None
_config_digest = None
_config_path = None
def get_config() -> Config:
    global _config, _config_digest, _config_path
    path = (home_dir() / "config.yaml").resolve()
    try:
        digest = path.read_bytes()
    except FileNotFoundError:
        digest = None
    if _config is None or _config_path != path or _config_digest != digest:
        _config = load_config(path)
        _config_path = path
        _config_digest = digest
    return _config
def save_config(cfg, path=None):
    global _config, _config_digest, _config_path
    p = Path(path) if path else home_dir() / "config.yaml"; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(yaml.safe_dump(cfg.model_dump()), encoding="utf-8")
    _config = None
    _config_digest = None
    _config_path = None
