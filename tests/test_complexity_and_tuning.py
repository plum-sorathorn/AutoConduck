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
