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
    anthropic_base_url: str | None = None
    api_base: str | None = None
    tier: str = "balanced"
    price_in: float = 0.0
    price_out: float = 0.0
    enabled: bool = True
    max_usd_per_min: float | None = None


class SelectionConfig(BaseModel):
    """Selection controls; pool entries may set quality_score and max_usd_per_min."""

    value_to_cost_gamma: float = 1.0
    pseudo_bias_budget: float = -0.20
    pseudo_bias_expensive: float = 0.20
    pseudo_bias_enabled: bool = True
    ema_min_samples: int = 3
    closeness_epsilon: float = 0.02
    expose_value_in_stats: bool = True
    phase_bands: dict[str, list[float]] = Field(
        default_factory=lambda: {
            "planner": [0.55, 0.85],
            "subagent": [0.10, 0.55],
            "executor": [0.35, 0.70],
        }
    )
    complexity_weights: dict[str, float] = Field(
        default_factory=lambda: {
            # Layer 1 — surface signals
            "length":              0.08,
            "structural":          0.12,
            "scope_breadth":       0.12,
            "code_density":        0.05,
            # Layer 2 — domain-agnostic semantic signals
            "abstraction_level":   0.12,
            "uncertainty_hedge":   0.08,
            "cross_domain":        0.12,
            "task_novelty":        0.08,
            "imperative_strength": 0.15,
            "multi_step":          0.08,
            # weights sum to 1.00
        }
    )
    ema_alpha: float = 0.1
    quality_min_success_rate: float = 0.5
    spend_guard_enabled: bool = True
    spend_guard_max_usd_per_min: float = 0.20
    spend_guard_window_s: int = 300
    # Tiebreaker: disabled by default — the extra LLM call almost always just
    # falls back to the complexity-based decision anyway.  Enable manually for
    # installations that want the extra signal.
    tiebreaker_enabled: bool = False
    tiebreaker_min_complexity: float = 0.45
    budget_tiebreaker_min_complexity: float = 0.65
    slow_threshold: float = 0.75
    # Minimum complexity required before the full LangGraph orchestrator is
    # invoked.  Requests below this threshold are treated as fast-path even
    # when the dispatcher returns path=slow, saving 3-5 LLM calls per turn.
    min_orchestrator_complexity: float = 0.62
    # Maximum scaled cost permitted for file reading / recon / read analyst subagents.
    # Expensive models (scaled_cost > 0.55) will be excluded from file read tasks.
    max_file_read_scaled_cost: float = 0.55
    # Threshold below which an escalated session de-escalates back to fast path.
    deescalation_threshold: float = 0.40
    # Enable compiled fast-path mini graph execution.
    enable_fast_path_graph: bool = True
    # Executor-subagent fan-out doubles LLM call count for multi-subtask plans;
    # keep disabled by default and only enable for truly complex batch tasks.
    enable_executor_subagents: bool = False


class ClaudeCodeSettings(BaseModel):
    # Add "Bash" here to auto-approve shell commands. It is intentionally not
    # enabled by default because blanket shell approval is a security relaxation.
    allowed_tools: list[str] = [
        "Task",
        "Skill",
        "Git",
        "Read",
        "Write",
        "Edit",
        "WebFetch",
    ]
    default_mode: str | None = None
    enable_all_project_mcp_servers: bool = False


class PiSettings(BaseModel):
    """Pi coding-agent integration settings."""

    enabled: bool = True
    model: str | None = None
    provider: str = "autoconduck"
    api_key_env: str = "PI_API_KEY"
    api_key: str | None = None
    base_url: str | None = None
    context_window: int = 1000000
    model_entries: list[ModelEntry] = Field(default_factory=list)


class Config(BaseModel):
    host: str = "127.0.0.1"
    port: int = 11434
    log_level: str = "INFO"
    # Tightened from 0.55/0.70 — reduces the ambiguous zone so fewer messages
    # trigger the expensive tiebreaker/orchestrator resolution path.
    ambiguous_low: float = 0.60
    ambiguous_high: float = 0.75
    hysteresis_floor: float = 0.50
    escalation_threshold: float = 0.80
    stack_trace_boost: float = 0.25
    ema_alpha: float = 0.1
    degraded_error_rate: float = 0.20
    degraded_window_s: int = 300
    fast_path_digest_enabled: bool = True
    fast_path_digest_max_files: int = 4
    fast_path_digest_max_bytes: int = 8192
    fast_path_digest_max_lines: int = 40
    fast_path_digest_timeout_ms: int = 150
    fast_path_digest_max_total_bytes: int = 12000
    fast_path_digest_min_files: int = 2
    pseudo_model: str = "autoconduck"
    model_list: list[dict] = Field(default_factory=list)
    routing_log: bool = True
    selected_presets: list[str] = Field(default_factory=list)
    custom_models: list[dict] = Field(default_factory=list)
    preset_overrides: dict[str, list[dict]] = Field(default_factory=dict)
    shims: dict[str, str] = Field(default_factory=dict)
    managed_server: bool = False
    # When True, `conduck start --[agent]` spawns the AutoConduck proxy in a
    # separate terminal window and runs the agent in the calling terminal.
    launch_in_new_terminal: bool = False
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    claude_code: ClaudeCodeSettings = Field(default_factory=ClaudeCodeSettings)
    pi: PiSettings = Field(default_factory=PiSettings)


_legacy_key_warning = False


def provider_for(entry: dict, cfg=None) -> str:
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


def resolve_api_key(entry: dict, provider=None) -> str:
    global _legacy_key_warning
    try:
        from .auth import get_provider_key

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
    """Repair a malformed/missing URL scheme so base URLs are always usable.

    Fixes the classic ``ttps://`` typo and bare hostnames without changing
    values that already carry a valid HTTP(S) scheme.
    """
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


def _normalize_model_entries(config_dict):
    for field in ("custom_models", "model_list"):
        for entry in config_dict.get(field, []) or []:
            if not isinstance(entry, dict):
                continue
            for key in ("base_url", "api_base"):
                if entry.get(key) is not None:
                    entry[key] = _repair_base_url_scheme(entry[key])


def _configured_model_sources(cfg):
    """Yield model pools in precedence order, including Pi's optional pool."""
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


def resolve_orchestrator_model(cfg=None) -> str:
    """Select the first enabled configured model for orchestration calls."""
    if cfg is None:
        try:
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
    return "gpt-4o"


def select_model_by_tier(tier: str, cfg=None) -> str:
    """Select a configured model by relative price, with the legacy fallback."""
    try:
        from autoconduck.routing.pricing import select_model_by_tier as _select

        return _select(tier, cfg or get_config()) or resolve_orchestrator_model(cfg)
    except Exception:
        return resolve_orchestrator_model(cfg)


def orchestrator_litellm_params(cfg=None) -> dict[str, str]:
    """Build LiteLLM kwargs for the configured orchestration model."""
    from .messages_api import litellm_params_for

    model = resolve_orchestrator_model(cfg)
    return litellm_params_for(model, cfg or get_config())


def home_dir() -> Path:
    return Path(os.environ.get("AUTOCONDUCK_HOME", Path.home() / ".autoconduck"))


def backups_dir(agent: str | None = None) -> Path:
    path = home_dir() / "backups"
    return path / agent if agent else path


def logs_path() -> Path:
    return home_dir() / "autoconduck.log"


def run_dir() -> Path:
    return home_dir() / "run"


def load_config(path=None) -> Config:
    p = Path(path) if path else home_dir() / "config.yaml"
    data = {}
    if p.exists():
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if "AUTOCONDUCK_PORT" in os.environ:
        data["port"] = int(os.environ["AUTOCONDUCK_PORT"])
    if "AUTOCONDUCK_LOG_LEVEL" in os.environ:
        data["log_level"] = os.environ["AUTOCONDUCK_LOG_LEVEL"]
    config = Config(**data)
    _normalize_model_entries({
        "custom_models": config.custom_models,
        "model_list": config.model_list,
    })
    if any(
        isinstance(e, dict) and e.get("api_key")
        for s in (config.model_list, config.custom_models)
        for e in s
    ):
        from .auth import migrate_from_config

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
    p = Path(path) if path else home_dir() / "config.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    data = cfg.model_dump()
    _normalize_model_entries(data)
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    _config = None
    _config_digest = None
    _config_path = None


def backup_config(path=None) -> Path | None:
    """Make a plain, timestamped backup of config.yaml before managed edits."""
    source = Path(path) if path else home_dir() / "config.yaml"
    if not source.exists():
        return None
    import shutil

    stamp = __import__("datetime").datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    target = backups_dir("config") / f"{stamp}.bak"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target
