"""Unit tests for autoconduck.tuning — pure, sync, deterministic, no network."""

import json
import math

import pytest

from autoconduck.tuning import (
    SimpleInputs,
    TuneResult,
    compute_tuning,
    project_spend,
    token_to_usd,
    save_profile,
    load_profile,
    _defaults,
    DEFAULT_BANDS,
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


# --------------------------------------------------------------------------
# Pressure: monotonicity + clamping
# --------------------------------------------------------------------------


def test_pressure_decreases_as_budget_increases():
    limits = [1, 10, 87, 500, 5000]
    pressures = [
        compute_tuning(_fixed_inputs(monthly_limit=m), POOL).pressure for m in limits
    ]
    for a, b in zip(pressures, pressures[1:]):
        assert a >= b - 1e-9, (
            f"pressure should be non-increasing as budget grows: {pressures}"
        )


def test_pressure_clamped_high_for_tiny_budget():
    result = compute_tuning(_fixed_inputs(monthly_limit=0.0001, headroom_pct=0), POOL)
    assert 0.0 <= result.pressure <= 1.0
    assert result.pressure > 0.9


def test_pressure_clamped_low_for_huge_budget():
    result = compute_tuning(_fixed_inputs(monthly_limit=1_000_000), POOL)
    assert 0.0 <= result.pressure <= 1.0
    assert result.pressure == 0.0


# --------------------------------------------------------------------------
# Fixed 4-model pool, known input ($87, headroom 25%, 160h, burst 1.8):
# pin the pure-linear tunable formulas to exact values (derived from the
# actual computed pressure, so the test still fails if the linear formulas
# regress even though pressure itself is a nonlinear function of inputs).
# --------------------------------------------------------------------------


def test_fixed_pool_linear_formulas_exact():
    result = compute_tuning(_fixed_inputs(), POOL)
    p = result.pressure

    # Sanity: analytically-derived pressure for this exact fixture (hand
    # verified against blended prices 0.175/0.95/6.0/7.5 and target=$65.25
    # over 160h => rate=0.006796875 $/min).
    assert result.pressure == pytest.approx(0.48343236005114476, rel=1e-9)

    tunables = result.tunables
    assert tunables["value_to_cost_gamma"][1] == pytest.approx(1.0 + p * 2.0)
    assert tunables["pseudo_bias_budget"][1] == pytest.approx(-0.20 - p * 0.20)
    assert tunables["pseudo_bias_expensive"][1] == pytest.approx(0.20 - p * 0.35)
    assert tunables["ambiguous_low"][1] == pytest.approx(0.60 + p * 0.05)
    assert tunables["ambiguous_high"][1] == pytest.approx(0.75 + p * 0.05)
    assert tunables["ema_alpha"][1] == pytest.approx(0.10 + p * 0.10)

    rate = 65.25 / (160 * 60)
    assert result.rate_per_min == pytest.approx(rate)
    guard = rate * 1.8
    assert tunables["spend_guard_max_usd_per_min"][1] == pytest.approx(guard)


def test_fixed_pool_per_model_limits_weighted_by_price():
    result = compute_tuning(_fixed_inputs(), POOL)
    guard = result.tunables["spend_guard_max_usd_per_min"][1]
    limits = result.per_model_limits
    # cheapest model gets full weight (weight=1 -> guard * 1.0)
    assert limits["qwen3.7-flash"] == pytest.approx(guard * 1.0)
    # priciest model gets floor weight (weight=0 -> guard * 0.3)
    assert limits["gpt-5.6-luna"] == pytest.approx(guard * 0.3)
    # monotonic: cheaper price -> higher (or equal) limit
    ordered = [
        limits["qwen3.7-flash"],
        limits["muse-spark-1.2"],
        limits["claude-sonnet-5"],
        limits["gpt-5.6-luna"],
    ]
    for a, b in zip(ordered, ordered[1:]):
        assert a >= b - 1e-12
    # every limit formula matches guard * (0.3 + 0.7*weight) with weight in [0,1]
    for v in limits.values():
        assert guard * 0.3 - 1e-9 <= v <= guard * 1.0 + 1e-9


def test_fixed_pool_phase_bands_shift_and_min_width():
    result = compute_tuning(_fixed_inputs(), POOL)
    p = result.pressure
    bands = result.tunables["phase_bands"][1]
    shifts = {"planner": 0.20, "subagent": 0.20, "executor": 0.25}
    for name, band in DEFAULT_BANDS.items():
        lo_expected = max(0.02, band[0] - p * shifts[name])
        hi_expected = max(0.02 + 0.05, band[1] - p * shifts[name])
        if hi_expected - lo_expected < 0.05:
            hi_expected = min(1.0, lo_expected + 0.05)
            lo_expected = max(0.02, hi_expected - 0.05)
        lo, hi = bands[name]
        assert lo == pytest.approx(round(lo_expected, 3))
        assert hi == pytest.approx(round(hi_expected, 3))
        assert hi - lo >= 0.05 - 1e-9, "band width must never fall below 0.05"


# --------------------------------------------------------------------------
# Single-model pool
# --------------------------------------------------------------------------


def test_single_model_pool_keeps_pool_relative_tunables_at_defaults():
    single = [{"id": "solo-model", "price_in": 1.0, "price_out": 3.0}]
    result = compute_tuning(_fixed_inputs(), single)
    assert any("Single-model pool" in w for w in result.warnings)
    # pool-relative tunables must not appear as "changed" since they were
    # reset to defaults internally (old == new for these keys).
    assert "value_to_cost_gamma" not in result.tunables
    assert "pseudo_bias_budget" not in result.tunables
    assert "pseudo_bias_expensive" not in result.tunables
    assert "phase_bands" not in result.tunables
    # no per-model override is written for a single-model pool
    assert result.per_model_limits == {}


# --------------------------------------------------------------------------
# Zero-price model / epsilon handling
# --------------------------------------------------------------------------


def test_zero_price_model_uses_epsilon_and_does_not_crash():
    pool = [
        {"id": "free-model", "price_in": 0.0, "price_out": 0.0},
        {"id": "paid-model", "price_in": 1.0, "price_out": 2.0},
    ]
    result = compute_tuning(_fixed_inputs(), pool)
    assert math.isfinite(result.pressure)
    for v in result.per_model_limits.values():
        assert math.isfinite(v)
        assert v >= 0


def test_all_zero_price_pool_does_not_crash():
    pool = [
        {"id": "free-a", "price_in": 0.0, "price_out": 0.0},
        {"id": "free-b", "price_in": 0.0, "price_out": 0.0},
    ]
    result = compute_tuning(_fixed_inputs(), pool)
    assert math.isfinite(result.pressure)
    assert 0.0 <= result.pressure <= 1.0


# --------------------------------------------------------------------------
# token_to_usd
# --------------------------------------------------------------------------


def test_token_to_usd_matches_blended_price_for_single_model():
    pool = [{"id": "m", "price_in": 1.0, "price_out": 3.0}]
    # ratio=3 -> blended = (3*1 + 3)/4 = 1.5 $/million tokens
    assert token_to_usd(1_000_000, pool, io_ratio=3.0) == pytest.approx(1.5)
    assert token_to_usd(2_000_000, pool, io_ratio=3.0) == pytest.approx(3.0)


def test_token_to_usd_averages_across_pool():
    pool = [
        {"id": "a", "price_in": 1.0, "price_out": 1.0},
        {"id": "b", "price_in": 3.0, "price_out": 3.0},
    ]
    # both blended prices equal their flat rate (in==out): 1.0 and 3.0 -> avg 2.0
    assert token_to_usd(1_000_000, pool) == pytest.approx(2.0)


def test_token_to_usd_empty_pool_is_zero():
    assert token_to_usd(1_000_000, []) == 0.0


# --------------------------------------------------------------------------
# Unreachable-target warning
# --------------------------------------------------------------------------


def test_unreachable_target_warns_with_tiny_budget():
    result = compute_tuning(_fixed_inputs(monthly_limit=0.001, headroom_pct=0), POOL)
    assert any("unreachable" in w.lower() for w in result.warnings)


def test_reachable_target_has_no_unreachable_warning():
    result = compute_tuning(_fixed_inputs(monthly_limit=500, headroom_pct=25), POOL)
    assert not any("unreachable" in w.lower() for w in result.warnings)


# --------------------------------------------------------------------------
# project_spend
# --------------------------------------------------------------------------


def test_project_spend_shares_sum_to_one_and_has_caveat():
    projection = project_spend({"expected_requests_per_month": 1000}, POOL)
    shares_total = sum(row["share"] for row in projection["rows"])
    assert shares_total == pytest.approx(1.0)
    assert "total" in projection
    assert projection["total"] >= 0
    assert "caveat" in projection and projection["caveat"]


def test_project_spend_uses_observed_stats_when_available():
    stats = [{"model": "qwen3.7-flash"}] * 8 + [{"model": "gpt-5.6-luna"}] * 2
    projection = project_spend(
        {"expected_requests_per_month": 1000}, POOL, stats_records=stats
    )
    rows_by_model = {r["model"]: r for r in projection["rows"]}
    assert rows_by_model["qwen3.7-flash"]["share"] == pytest.approx(0.8)
    assert rows_by_model["gpt-5.6-luna"]["share"] == pytest.approx(0.2)
    assert sum(r["share"] for r in projection["rows"]) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Profile round-trip (persisted via AUTOCONDUCK_HOME)
# --------------------------------------------------------------------------


def test_profile_round_trip_via_autoconduck_home(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOCONDUCK_HOME", str(tmp_path))
    inputs = _fixed_inputs()
    result = compute_tuning(inputs, POOL)
    save_profile(inputs, result)

    profile_path = tmp_path / "tune_profile.json"
    assert profile_path.exists()

    loaded = load_profile()
    assert loaded is not None
    assert loaded["inputs"]["monthly_limit"] == pytest.approx(inputs.monthly_limit)
    assert loaded["tunables"]["ema_alpha"] == pytest.approx(
        result.tunables["ema_alpha"][1]
    )
    assert loaded["per_model_limits"] == result.per_model_limits


def test_load_profile_missing_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOCONDUCK_HOME", str(tmp_path))
    assert load_profile() is None


def test_save_profile_explicit_path(tmp_path):
    inputs = _fixed_inputs()
    result = compute_tuning(inputs, POOL)
    path = tmp_path / "custom_profile.json"
    save_profile(inputs, result, path=path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert "saved_at" in data


# --------------------------------------------------------------------------
# Reset-to-defaults semantics
# --------------------------------------------------------------------------


def test_reset_current_equal_to_defaults_matches_implicit_default():
    inputs = _fixed_inputs()
    implicit = compute_tuning(inputs, POOL)
    explicit = compute_tuning(inputs, POOL, current=_defaults())
    assert implicit.tunables.keys() == explicit.tunables.keys()
    for key in implicit.tunables:
        assert implicit.tunables[key][1] == pytest.approx(explicit.tunables[key][1])


def test_reset_current_matching_new_values_yields_no_changes():
    inputs = _fixed_inputs()
    result = compute_tuning(inputs, POOL)
    already_current = {k: v[1] for k, v in result.tunables.items()}
    # merge with defaults for keys not present (unchanged ones)
    full_current = dict(_defaults())
    full_current.update(already_current)
    reset_result = compute_tuning(inputs, POOL, current=full_current)
    assert reset_result.tunables == {}


def test_defaults_match_default_bands_constant():
    defaults = _defaults()
    assert defaults["phase_bands"] == {k: list(v) for k, v in DEFAULT_BANDS.items()}
    assert defaults["value_to_cost_gamma"] == 1.0
    assert defaults["ambiguous_low"] == 0.60
    assert defaults["ambiguous_high"] == 0.75


# --------------------------------------------------------------------------
# CLI: `tune` subparser is registered with a handler
# --------------------------------------------------------------------------


def test_tune_subparser_registered_with_handler(monkeypatch):
    from autoconduck import main as cli

    called = {}

    def fake_cmd_tune(args):
        called["mode"] = getattr(args, "mode", None)

    monkeypatch.setattr(cli, "cmd_tune", fake_cmd_tune)
    monkeypatch.setattr("sys.argv", ["autoconduck", "tune", "--mode", "simple"])

    try:
        cli.main(["tune", "--mode", "simple"])
    except TypeError:
        cli.main()

    assert called.get("mode") == "simple"


def test_tune_mode_choices_reject_invalid_value():
    from autoconduck import main as cli

    with pytest.raises(SystemExit):
        cli.main(["tune", "--mode", "not-a-real-mode"])
