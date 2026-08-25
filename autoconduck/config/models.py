"""Configuration data models and schemas."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field, field_validator


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
    cost_input: float = 0.0
    cost_output: float = 0.0
    context_window: int = 128000
    supports_tools: bool = True
    is_reasoning: bool = False
    enabled: bool = True
    max_usd_per_min: float | None = None
    capability_score: float = 0.0


class SelectionConfig(BaseModel):
    phase_role_cards: bool = True
    dump_prompts: bool = False
    planner_model_override: str | None = None
    planner_response_format: str = "json_object"
    planner_retry_cheaper: bool = True
    progress_verbosity: str = "verbose"
    """Selection controls; pool entries may set quality_score and max_usd_per_min."""

    # 0.3.0 SLM Architecture & Dynamic DAG Tunables
    slm_model_path: str = "models/qwen2.5-coder-0.5b-instruct-q4.onnx"
    slm_circuit_breaker_timeout_ms: int = 100
    cheap_fast_max_cost: float = 0.50
    balanced_max_cost: float = 4.00
    session_guard_compaction_ratio: float = 0.80
    rag_max_tokens: int = 250
    rag_db_path: str = "~/.autoconduck/rag_db"

    value_to_cost_gamma: float = 1.0
    pseudo_bias_budget: float = -0.20
    pseudo_bias_expensive: float = 0.20
    pseudo_bias_enabled: bool = True
    ema_min_samples: int = 3
    closeness_epsilon: float = 0.02
    expose_value_in_stats: bool = True
    phase_bands: dict[str, list[float]] = Field(default_factory=dict)
    complexity_weights: dict[str, float] = Field(
        default_factory=lambda: {
            # Layer 1 — surface signals
            "length": 0.08,
            "structural": 0.12,
            "scope_breadth": 0.12,
            "code_density": 0.05,
            # Layer 2 — domain-agnostic semantic signals
            "abstraction_level": 0.12,
            "uncertainty_hedge": 0.08,
            "cross_domain": 0.12,
            "task_novelty": 0.08,
            "imperative_strength": 0.15,
            "multi_step": 0.08,
            # weights sum to 1.00
        }
    )
    ema_alpha: float = 0.1
    quality_min_success_rate: float = 0.5
    spend_guard_enabled: bool = True
    spend_guard_max_usd_per_min: float = 0.20
    spend_guard_window_s: int = 300
    tiebreaker_enabled: bool = False
    tiebreaker_min_complexity: float = 0.45
    budget_tiebreaker_min_complexity: float = 0.65
    slow_threshold: float = 0.75
    min_orchestrator_complexity: float = 0.72
    subagent_timeout_s: float = 120.0
    subagent_max_tokens: int = 4096
    max_file_read_scaled_cost: float = 0.55
    fast_path_max_scaled_cost: float = 0.50
    deescalation_threshold: float = 0.40
    enable_fast_path_graph: bool = True
    enable_executor_subagents: bool = False
    executor_enable_tools: bool = True
    executor_max_tool_rounds: int = 10
    executor_tool_time_budget_s: float = 180.0
    executor_max_read_bytes: int = 200_000
    executor_enable_bash: bool = False
    slow_stream_progress: bool = True
    default_target_bias: float = 0.0
    enable_per_turn_task_routing: bool = True
    recon_task_band: list[float] = [0.05, 0.35]
    edit_task_band: list[float] = [0.30, 0.65]
    verify_task_band: list[float] = [0.15, 0.50]
    bash_task_band: list[float] = [0.20, 0.55]
    recon_max_complexity: float = 0.20
    edit_min_complexity: float = 0.45
    verify_complexity_band: list[float] = [0.20, 0.50]
    latency_sensitivity: float = 0.0
    intent_drift_enabled: bool = True
    intent_drift_threshold: float = 0.70
    hysteresis_window_size: int = 5
    hysteresis_decay: float = 0.85
    non_english_fallback_complexity: float = 0.45

    @field_validator("progress_verbosity", mode="before")
    @classmethod
    def _valid_progress_verbosity(cls, value):
        return value if value in {"off", "terse", "verbose"} else "terse"


class ClaudeCodeSettings(BaseModel):
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
    models: dict[str, ModelEntry] = Field(default_factory=dict)
    model_list: list[dict] = Field(default_factory=list)
    routing_log: bool = True
    selected_presets: list[str] = Field(default_factory=list)
    custom_models: list[dict] = Field(default_factory=list)
    preset_overrides: dict[str, list[dict]] = Field(default_factory=dict)
    shims: dict[str, str] = Field(default_factory=dict)
    managed_server: bool = False
    launch_in_new_terminal: bool = False
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    claude_code: ClaudeCodeSettings = Field(default_factory=ClaudeCodeSettings)
    pi: PiSettings = Field(default_factory=PiSettings)
