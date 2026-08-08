import json, math, time
from collections import defaultdict, deque
from pathlib import Path
try:
    import litellm
    _COSTS = litellm.model_cost
except Exception: _COSTS = {}
try: _FALLBACK = json.loads((Path(__file__).with_name("pricing_fallback.json")).read_text())
except Exception: _FALLBACK = {}
_errors = defaultdict(deque); _ema = defaultdict(float); _last_usage = {}
def _entry(model):
    raw = _COSTS.get(model) or _FALLBACK.get(model) or {}
    return {"price_in": raw.get("input_cost_per_token", raw.get("price_in", 0.0)) * (1_000_000 if "input_cost_per_token" in raw else 1), "price_out": raw.get("output_cost_per_token", raw.get("price_out", 0.0)) * (1_000_000 if "output_cost_per_token" in raw else 1), **raw}
def is_subscription(model): return bool(_FALLBACK.get(model, {}).get("subscription", False))
def scaled_cost(model):
    costs = [_entry(m).get("price_in", 0) + _entry(m).get("price_out", 0) for m in set(_FALLBACK) | set(_COSTS)]
    value = _entry(model).get("price_in", 0) + _entry(model).get("price_out", 0)
    return math.log1p(value) / max((math.log1p(c) for c in costs), default=1.0)
def is_degraded(model, window_s=300, error_rate=.2):
    now=time.time(); q=_errors[model]
    while q and now-q[0][0] > window_s: q.popleft()
    return len(q) > 0 and sum(x[1] for x in q)/len(q) > error_rate
def record_error(model):
    _errors[model].append((time.time(), 1))
def record_usage(model, prompt_tokens, completion_tokens):
    estimate = max(1, int(prompt_tokens + completion_tokens)); prior = _ema[model]
    _ema[model] = estimate if not prior else .9 * prior + .1 * estimate
    _last_usage[model] = (prompt_tokens, completion_tokens)
def select(pool, pseudo_model, config, usage=None):
    candidates = [m for m in pool if not is_degraded(m, getattr(config,"degraded_window_s",300), getattr(config,"degraded_error_rate",.2))]
    if not candidates: candidates = list(pool)
    candidates.sort(key=scaled_cost)
    if pseudo_model.endswith("expensive") and candidates: return candidates[-1]
    return candidates[0] if candidates else ""
