import logging
import os
from urllib.parse import urlsplit, urlunsplit
from pathlib import Path
from pydantic import BaseModel, Field
import yaml

class ModelEntry(BaseModel):
    id: str
    provider: str = "openai"
    api_key_env: str = "OPENAI_API_KEY"
    api_key: str | None = None
    base_url: str | None = None
    tier: str = "balanced"
    price_in: float = 0.0
    price_out: float = 0.0
    enabled: bool = True

class SelectionConfig(BaseModel):
    value_to_cost_gamma: float = 1.0
    pseudo_bias_budget: float = -0.20
    pseudo_bias_expensive: float = 0.20
    pseudo_bias_enabled: bool = True
    ema_min_samples: int = 3
    closeness_epsilon: float = 0.02
    expose_value_in_stats: bool = True
    phase_bands: dict[str, list[float]] = Field(default_factory=lambda: {"planner": [0.55, 0.85], "subagent": [0.10, 0.55], "executor": [0.35, 0.70]})
    complexity_weights: dict[str, float] = Field(default_factory=lambda: {"length": .15, "refs": .10, "structural": .25, "files": .10, "keyword_domain": .15, "edit_intent": .15, "multi_step": .10})

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
    selection: SelectionConfig = Field(default_factory=SelectionConfig)

def resolve_api_key(entry: dict) -> str:
    if entry.get("api_key"):
        return str(entry["api_key"])
    name = entry.get("api_key_env")
    if name:
        return os.environ.get(name, "")
    return ""

def qualify_model(model_id: str) -> str:
    """Return a LiteLLM provider-qualified model name."""
    value = str(model_id or "")
    return value if "/" in value else f"openai/{value}"

def normalize_api_base(base_url: str) -> str:
    """Return an OpenAI-compatible endpoint URL with the required /v1 path."""
    value = str(base_url or "").rstrip("/")
    if not value:
        return value
    parts = urlsplit(value)
    if parts.path.rstrip("/").split("/")[-1].lower() == "v1":
        return value
    return urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/") + "/v1", parts.query, parts.fragment))

def resolve_orchestrator_model(cfg=None) -> str:
    """Select the first enabled configured model for orchestration calls."""
    if cfg is None:
        try:
            cfg = get_config()
        except Exception:
            cfg = None
    for source in (getattr(cfg, "model_list", []) or [], getattr(cfg, "custom_models", []) or []):
        for entry in source:
            if not isinstance(entry, dict) or entry.get("enabled", True) is False:
                continue
            model = entry.get("id") or entry.get("model_name") or entry.get("model")
            if not model and isinstance(entry.get("litellm_params"), dict):
                model = entry["litellm_params"].get("model")
            if model:
                return str(model)
    return "gpt-4o"

def select_model_by_tier(tier: str, cfg=None) -> str:
    """Select a configured model by relative price, with the legacy fallback."""
    try:
        from autoconduck.pricing import select_model_by_tier as _select
        return _select(tier, cfg or get_config()) or resolve_orchestrator_model(cfg)
    except Exception:
        return resolve_orchestrator_model(cfg)

def orchestrator_litellm_params(cfg=None) -> dict[str, str]:
    """Build LiteLLM kwargs for the configured orchestration model."""
    model = resolve_orchestrator_model(cfg)
    for source in (getattr(cfg, "model_list", []) or [], getattr(cfg, "custom_models", []) or []):
        for entry in source:
            if not isinstance(entry, dict):
                continue
            raw = entry.get("id") or entry.get("model_name") or entry.get("model")
            params = entry.get("litellm_params") if isinstance(entry.get("litellm_params"), dict) else entry
            if str(raw or "").removeprefix("openai/") != str(model).removeprefix("openai/"):
                continue
            result = {"model": qualify_model(model)}
            if params.get("base_url") or params.get("api_base"):
                result["api_base"] = normalize_api_base(params.get("base_url") or params["api_base"])
            if params.get("api_key_env") or params.get("api_key"):
                api_key = resolve_api_key(params)
                if api_key:
                    result["api_key"] = api_key
            return result
    return {"model": qualify_model(model)}
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
    config = Config(**data)
    for source in (config.model_list, config.custom_models):
        for entry in source:
            if (isinstance(entry, dict) and entry.get("api_key_env")
                    and not entry.get("api_key")
                    and not os.environ.get(entry["api_key_env"])):
                logging.getLogger("autoconduck").warning(
                    "API key environment variable %s is not set for model %s",
                    entry["api_key_env"], entry.get("id") or entry.get("model_name") or "<unknown>",
                )
    return config
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
