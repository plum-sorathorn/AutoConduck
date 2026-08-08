import os
from pathlib import Path
from pydantic import BaseModel, Field
import yaml

class Config(BaseModel):
    host: str = "127.0.0.1"; port: int = 11434; log_level: str = "INFO"
    ambiguous_low: float = 0.55; ambiguous_high: float = 0.70; hysteresis_floor: float = 0.50; escalation_threshold: float = 0.80; stack_trace_boost: float = 0.25
    ema_alpha: float = 0.1; degraded_error_rate: float = 0.20; degraded_window_s: int = 300; pseudo_model: str = "autoconduck"
    model_list: list[dict] = Field(default_factory=list); routing_log: bool = True
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
def get_config() -> Config:
    global _config
    if _config is None: _config = load_config()
    return _config
def save_config(cfg, path=None):
    p = Path(path) if path else home_dir() / "config.yaml"; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(yaml.safe_dump(cfg.model_dump()), encoding="utf-8")
