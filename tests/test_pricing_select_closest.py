from types import SimpleNamespace
from autoconduck import pricing

def cfg():
    return SimpleNamespace(model_list=[{"id": "low", "price_in": .01, "price_out": .01}, {"id": "lowmid", "price_in": .1, "price_out": .1}, {"id": "mid", "price_in": 1, "price_out": 1}, {"id": "midhigh", "price_in": 3, "price_out": 3}, {"id": "high", "price_in": 10, "price_out": 10}], selection=SimpleNamespace(value_to_cost_gamma=1, pseudo_bias_enabled=True, pseudo_bias_budget=-.2, pseudo_bias_expensive=.2, ema_min_samples=3))
def test_select_closest_picks_nearest_not_cheapest(): assert pricing.select_closest(pricing.pool_ids(cfg()), .4, cfg()) == "mid"
def test_select_closest_pseudo_bias_shifts_target():
    c=cfg(); assert pricing.scaled_cost(pricing.select_closest(pricing.pool_ids(c), .4, c, pseudo_model="autoconduck-budget"),c) < pricing.scaled_cost(pricing.select_closest(pricing.pool_ids(c), .4,c),c)
    assert pricing.scaled_cost(pricing.select_closest(pricing.pool_ids(c), .4,c,pseudo_model="autoconduck-expensive"),c) > pricing.scaled_cost(pricing.select_closest(pricing.pool_ids(c), .4,c),c)
def test_select_closest_excludes_degraded(): assert pricing.select_closest(["low","mid","high"], .1, cfg(), degraded={"low"}) == "mid"
def test_select_closest_all_degraded_falls_back_to_cheapest(): assert pricing.select_closest(["low","mid"], .5, cfg(), degraded={"low","mid"}) == "low"
def test_select_closest_near_tie_deterministic(): assert pricing.select_closest(["low","mid"], .5, cfg()) == pricing.select_closest(["low","mid"], .5, cfg())
def test_select_closest_never_raises_on_bad_config(): assert pricing.select_closest([], .5, SimpleNamespace()) == ""
