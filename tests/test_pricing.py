from types import SimpleNamespace
import autoconduck.routing.pricing as pricing


def setup_function():
    pricing._COSTS = {}
    pricing._FALLBACK = {
        "cheap-model": {"price_in": 0.5, "price_out": 0.5},
        "mid-model": {"price_in": 3, "price_out": 3},
        "expensive-model": {"price_in": 15, "price_out": 15},
        "cheap": {"price_in": 1, "price_out": 1},
        "mid": {"price_in": 2, "price_out": 2},
        "subscription": {"subscription": True, "price_in": 0, "price_out": 0},
    }
    pricing._errors.clear()
    pricing._ema.clear()


def test_scaled_cost_is_monotonic():
    assert pricing.scaled_cost("cheap") < pricing.scaled_cost("mid")


def test_select_closest_matches_target():
    cfg = SimpleNamespace(
        degraded_window_s=300,
        degraded_error_rate=0.2,
        value_to_cost_gamma=1.0,
        pseudo_bias_budget=-0.2,
        pseudo_bias_expensive=0.2,
        pseudo_bias_enabled=True,
        ema_min_samples=3,
        closeness_epsilon=0.02,
    )
    assert pricing.select_closest(["mid", "cheap"], 0.7, cfg) == "cheap"


def test_subscription_flag_and_ema_correction():
    assert pricing.is_subscription("subscription")
    pricing.record_usage("cheap", 10, 5)
    pricing.record_usage("cheap", 20, 10)
    assert pricing._ema["cheap"]["samples"] == 2


def test_ema_effective_value_keeps_price_units():
    cfg = SimpleNamespace(
        model_list=[{"id": "cheap", "price_in": 1, "price_out": 1}],
        selection=SimpleNamespace(ema_min_samples=1, quality_min_success_rate=0.5),
    )
    pricing.record_usage("cheap", 750_000, 250_000, cost=1.0)
    assert pricing._entry_effective_value("cheap", cfg) == 1.0


def test_scaled_costs_compute_pool_once(monkeypatch):
    cfg = SimpleNamespace(
        model_list=[
            {"id": "cheap", "price_in": 1, "price_out": 1},
            {"id": "mid", "price_in": 2, "price_out": 2},
        ]
    )
    calls = []
    original = pricing._entry_effective_value
    monkeypatch.setattr(
        pricing,
        "_entry_effective_value",
        lambda model, config: calls.append(model) or original(model, config),
    )
    pricing.select_closest(pricing.pool_ids(cfg), 0.5, cfg)
    assert calls == ["cheap", "mid"]


def test_select_dict_pool_matches_closest_target():
    pool = [
        {"id": "a", "price_in": 0.1, "price_out": 0.1},
        {"id": "b", "price_in": 0.01, "price_out": 0.01},
    ]
    cfg = SimpleNamespace(
        degraded_window_s=300, degraded_error_rate=0.2, model_list=pool
    )
    assert pricing.select_closest(pool, 0.0, cfg) == "b"
    assert pricing.select_closest(pool, 1.0, cfg) == "a"


def test_select_closest_dict_pool_does_not_raise():
    cfg = SimpleNamespace(
        degraded_window_s=300, degraded_error_rate=0.2, model_list=[{"id": "x"}]
    )
    assert pricing.select_closest([{"id": "x"}], 0.15, cfg) == "x"


def test_select_model_by_tier_is_deterministic_for_equal_costs():
    models = [
        {"id": "qwen3.7-flash", "price_in": 0.001, "price_out": 0.002},
        {"id": "muse-spark-1.2", "price_in": 0.001, "price_out": 0.002},
        {"id": "gpt-5.6-luna", "price_in": 0.0002, "price_out": 0.0012},
        {"id": "claude-sonnet-5", "price_in": 0.002, "price_out": 0.01},
    ]
    cfg = SimpleNamespace(model_list=models)
    assert pricing.select_model_by_tier("cheap", cfg) == "claude-sonnet-5"
    assert pricing.select_model_by_tier("mid", cfg) == "claude-sonnet-5"
    assert pricing.select_model_by_tier("expensive", cfg) == "claude-sonnet-5"
