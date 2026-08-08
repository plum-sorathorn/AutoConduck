from types import SimpleNamespace
import autoconduck.pricing as pricing


def setup_function():
    pricing._COSTS = {}
    pricing._FALLBACK = {"cheap": {"price_in": 1, "price_out": 1}, "mid": {"price_in": 2, "price_out": 2}, "subscription": {"subscription": True, "price_in": 0, "price_out": 0}}
    pricing._errors.clear()
    pricing._ema.clear()


def test_scaled_cost_is_monotonic():
    assert pricing.scaled_cost("cheap") < pricing.scaled_cost("mid")


def test_select_cheapest_and_skips_degraded():
    cfg = SimpleNamespace(degraded_window_s=300, degraded_error_rate=.2)
    assert pricing.select(["mid", "cheap"], "autoconduck", cfg) == "cheap"
    for _ in range(3): pricing.record_error("cheap")
    assert pricing.select(["mid", "cheap"], "autoconduck", cfg) == "mid"


def test_subscription_flag_and_ema_correction():
    assert pricing.is_subscription("subscription")
    pricing.record_usage("cheap", 10, 5)
    pricing.record_usage("cheap", 20, 10)
    assert pricing._ema["cheap"] == 0.9 * 15 + 0.1 * 30
