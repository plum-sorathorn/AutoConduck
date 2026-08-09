import json, math, time
from collections import defaultdict, deque
from pathlib import Path
try:
    import litellm
    _COSTS = litellm.model_cost
except Exception: _COSTS = {}
try: _FALLBACK = json.loads((Path(__file__).with_name("pricing_fallback.json")).read_text())
except Exception: _FALLBACK = {}
_errors = defaultdict(deque); _ema = {}; _last_usage = {}
def _pool_id(entry):
    if isinstance(entry, str): return entry
    if isinstance(entry, dict):
        return entry.get("id") or entry.get("model_name") or entry.get("model")
    return None
def _configured_entry(model, config):
    wanted = str(model).removeprefix("openai/")
    for source in (getattr(config, "model_list", []) or [], getattr(config, "custom_models", []) or []):
        for entry in source:
            if not isinstance(entry, dict): continue
            name = entry.get("id") or entry.get("model_name") or entry.get("model")
            if isinstance(entry.get("litellm_params"), dict): name = name or entry["litellm_params"].get("model")
            if str(name or "").removeprefix("openai/") == wanted:
                params = entry.get("litellm_params") if isinstance(entry.get("litellm_params"), dict) else entry
                if "price_in" in params or "price_out" in params:
                    return params
    return None
def _entry(model, config=None):
    raw = _configured_entry(model, config) or _COSTS.get(model) or _COSTS.get(str(model).removeprefix("openai/")) or _FALLBACK.get(model) or _FALLBACK.get(str(model).removeprefix("openai/")) or {}
    return {"price_in": raw.get("input_cost_per_token", raw.get("price_in", 0.0)) * (1_000_000 if "input_cost_per_token" in raw else 1), "price_out": raw.get("output_cost_per_token", raw.get("price_out", 0.0)) * (1_000_000 if "output_cost_per_token" in raw else 1), **raw}
def is_subscription(model): return bool(_FALLBACK.get(model, {}).get("subscription", False))
def _entry_effective_value(model, config):
    entry = _entry(model, config); ema = _ema.get(str(model))
    minimum = getattr(getattr(config, "selection", None), "ema_min_samples", 3)
    if ema and ema["samples"] >= minimum:
        return float(entry.get("price_in", 0)) * ema["prompt"] / 1000 + float(entry.get("price_out", 0)) * ema["completion"] / 1000
    return float(entry.get("price_in", 0)) + float(entry.get("price_out", 0))
def scaled_cost(model, config=None, *, use_ema=True):
    # Compute entries once per model instead of calling _entry() separately for price_in and price_out.
    configured = [str(name) for entry in (getattr(config, "model_list", []) or [])
                  if (name := _pool_id(entry))]
    entries_by_name = {name: _entry(name, config) for name in configured}
    cost_entry = entries_by_name.get(str(model)) or _entry(model, config)
    value = _entry_effective_value(model, config) if use_ema else cost_entry.get("price_in", 0) + cost_entry.get("price_out", 0)
    denominator = max(1.0, max((math.log1p(_entry_effective_value(name, config) if use_ema else e.get("price_in", 0) + e.get("price_out", 0)) for name, e in entries_by_name.items()), default=1.0)) if configured else 1.0
    return math.log1p(value) / denominator
def is_degraded(model, window_s=300, error_rate=.2):
    now=time.time(); q=_errors[str(model)]
    while q and now-q[0][0] > window_s: q.popleft()
    return len(q) > 0 and sum(x[1] for x in q)/len(q) > error_rate
def record_error(model):
    _errors[model].append((time.time(), 1))
def record_usage(model, prompt_tokens, completion_tokens):
    estimate = max(1, int(prompt_tokens + completion_tokens))
    old = _ema.get(model)
    _ema[model] = {"prompt": prompt_tokens if not old else .9 * old["prompt"] + .1 * prompt_tokens, "completion": completion_tokens if not old else .9 * old["completion"] + .1 * completion_tokens, "samples": (old["samples"] + 1 if old else 1)}
    _last_usage[model] = (prompt_tokens, completion_tokens)
def select(pool, pseudo_model, config, usage=None):
    degraded = {m for m in pool_ids_from(pool) if is_degraded(m, getattr(config, "degraded_window_s", 300), getattr(config, "degraded_error_rate", .2))}
    return select_closest(pool, .15, config, pseudo_model=pseudo_model, degraded=degraded)

def select_model_by_tier(tier, config):
    value = {"cheap": .15, "budget": .15, "mid": .5, "balanced": .5, "reasoning": .7, "expensive": .85}.get(tier, .5)
    return select_closest(pool_ids(config), value, config)

def pool_ids(config):
    return [str(n) for e in (getattr(config, "model_list", []) or []) if (n := _pool_id(e)) and (not isinstance(e, dict) or e.get("enabled", True) is not False)]
def pool_ids_from(pool):
    return [str(n) for e in pool if (n := _pool_id(e))]
def cheapest_enabled(config):
    names = pool_ids(config)
    return min(names, key=lambda m: (_entry_effective_value(m, config), m)) if names else ""
def target_scaled_cost(value, pseudo_model, config):
    sel = getattr(config, "selection", config); gamma = float(getattr(sel, "value_to_cost_gamma", 1.0)); bias = 0.0
    if getattr(sel, "pseudo_bias_enabled", True): bias = {"autoconduck-budget": getattr(sel, "pseudo_bias_budget", -.2), "autoconduck-expensive": getattr(sel, "pseudo_bias_expensive", .2)}.get(pseudo_model, 0.0)
    return max(0.0, min(1.0, float(value) ** gamma + bias))
def select_closest(pool, value, config, *, pseudo_model=None, band=None, degraded=None):
    try:
        names = [str(_pool_id(x)) for x in pool if _pool_id(x)]
        eligible = [m for m in names if not (degraded and m in degraded)]
        if band:
            inside = [m for m in eligible if band[0] <= scaled_cost(m, config) <= band[1]]; eligible = inside or eligible
        if not eligible: return cheapest_enabled(config)
        target = target_scaled_cost(value, pseudo_model, config)
        return sorted(eligible, key=lambda m: (abs(scaled_cost(m, config) - target), scaled_cost(m, config), m))[0]
    except Exception:
        return cheapest_enabled(config)
