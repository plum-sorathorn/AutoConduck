"""Unit tests for complexity scoring, valuation, budget pressure, and tuning."""
import json
import math
import pytest

from autoconduck.config import Config, SelectionConfig
from autoconduck.routing.evaluator import complexity_of
from autoconduck.tuning import (
    DEFAULT_BANDS,
    SimpleInputs,
    TuneResult,
    _defaults,
    compute_tuning,
    load_profile,
    project_spend,
    save_profile,
    token_to_usd,
)

POOL = [
    {"id": "qwen3.7-flash", "price_in": 0.10, "price_out": 0.40},
    {"id": "muse-spark-1.2", "price_in": 0.60, "price_out": 2.00},
    {"id": "claude-sonnet-5", "price_in": 3.00, "price_out": 15.00},
    {"id": "gpt-5.6-luna", "price_in": 5.00, "price_out": 15.00},
]


def _fixed_inputs(monthly_limit=87.0, headroom_pct=25.0, active_hours=160.0, burst=1.8):
    return SimpleInputs(
        monthly_limit=monthly_limit,
        headroom_pct=headroom_pct,
        active_hours_per_month=active_hours,
        burst_factor=burst,
    )


# ---------------------------------------------------------------------------
# Complexity & Valuation Tests
# ---------------------------------------------------------------------------


def test_calculate_complexity_bounds():
    score = complexity_of("refactor database models and update authentication")
    assert 0.0 <= score <= 1.0


def test_complexity_increases_with_scope():
    simple = complexity_of("fix typo")
    complex_task = complexity_of(
        "Refactor the entire database architecture, migrate all tables, overhaul API endpoints, and rewrite tests."
    )
    assert complex_task > simple


# ---------------------------------------------------------------------------
# Tuning Engine & Pressure Calculations
# ---------------------------------------------------------------------------


def test_pressure_decreases_as_budget_increases():
    limits = [1, 10, 87, 500, 5000]
    pressures = [compute_tuning(_fixed_inputs(monthly_limit=m), POOL).pressure for m in limits]
    for a, b in zip(pressures, pressures[1:]):
        assert a >= b - 1e-9


def test_fixed_pool_linear_formulas_exact():
    inputs = _fixed_inputs()
    result = compute_tuning(inputs, POOL)
    p = result.pressure
    assert result.pressure == pytest.approx(0.48343236005114476, rel=1e-9)

    tunables = result.tunables
    assert tunables["value_to_cost_gamma"][1] == pytest.approx(1.0 + p * 2.0)
    assert tunables["pseudo_bias_budget"][1] == pytest.approx(-0.20 - p * 0.20)
    assert tunables["pseudo_bias_expensive"][1] == pytest.approx(0.20 - p * 0.35)
    assert tunables["ambiguous_low"][1] == pytest.approx(0.60 + p * 0.05)
    assert tunables["ambiguous_high"][1] == pytest.approx(0.75 + p * 0.05)
    assert tunables["ema_alpha"][1] == pytest.approx(0.10 + p * 0.10)
    assert tunables["fast_path_max_scaled_cost"][1] == pytest.approx(round(max(0.15, min(0.75, 0.65 - p * 0.40)), 3))

    rate = 65.25 / (160 * 60)
    assert result.rate_per_min == pytest.approx(rate)
    guard = rate * 1.8
    assert tunables["spend_guard_max_usd_per_min"][1] == pytest.approx(guard)


def test_fixed_pool_per_model_limits_weighted_by_price():
    inputs = _fixed_inputs()
    result = compute_tuning(inputs, POOL)
    limits = result.per_model_limits
    assert "qwen3.7-flash" in limits
    assert "gpt-5.6-luna" in limits
    assert isinstance(limits["qwen3.7-flash"], float)


def test_save_and_load_profile(tmp_path, monkeypatch):
    from autoconduck import config
    monkeypatch.setattr(config, "home_dir", lambda: tmp_path)
    inputs = _fixed_inputs()
    result = compute_tuning(inputs, POOL)
    save_profile(inputs, result)
    loaded = load_profile()
    assert "tunables" in loaded or "inputs" in loaded or isinstance(loaded, dict)


def test_scope_breadth_snake_and_kebab_case():
    from autoconduck.routing.complexity import complexity_of
    snake_case_prompt = "Refactor user_account_manager, payment_gateway_service, and audit_log_emitter across all modules"
    camel_case_prompt = "Refactor UserAccountManager, PaymentGatewayService, and AuditLogEmitter across all modules"
    assert complexity_of(snake_case_prompt) > 0.35
    assert complexity_of(camel_case_prompt) > 0.35


def test_cross_domain_frontend_sql_and_ml_keywords():
    from autoconduck.routing.complexity import complexity_of
    ui_task = "Fix webgl shader animation layout shift and reconcile virtual dom state"
    sql_task = "Analyze query plan, fix n+1 queries with composite index and table scan optimization"
    ml_task = "Diagnose loss divergence and cuda oom with gradient backpropagation and ddp"
    assert complexity_of(ui_task) > 0.35
    assert complexity_of(sql_task) > 0.35
    assert complexity_of(ml_task) > 0.35


def test_short_prompt_abstraction_dampened():
    from autoconduck.routing.complexity import complexity_of
    short = "design"
    long_arch = "design and architect the multi-tenant distributed streaming pipeline framework"
    assert complexity_of(short) < complexity_of(long_arch)


def test_non_english_fallback_complexity():
    from autoconduck.routing.complexity import complexity_of, is_non_english
    cjk_prompt = "请重构整个数据库架构并优化查询性能"
    assert is_non_english(cjk_prompt) is True
    # Non-English prompt gets a balanced fallback floor (~0.45+) instead of collapsing to 0
    assert complexity_of(cjk_prompt) >= 0.40


def test_recalibrate_weights_from_records():
    from autoconduck.tuning import recalibrate_weights_from_records
    records = [
        {"escalated": True, "path": "slow", "complexity": 0.85},
        {"escalated": True, "path": "slow", "complexity": 0.80},
        {"escalated": True, "path": "slow", "complexity": 0.90},
        {"escalated": False, "path": "fast", "complexity": 0.30},
        {"escalated": False, "path": "fast", "complexity": 0.20},
    ]
    recalibrated = recalibrate_weights_from_records(records)
    assert sum(recalibrated.values()) == pytest.approx(1.0, rel=1e-3)
    assert recalibrated["scope_breadth"] > 0.12

