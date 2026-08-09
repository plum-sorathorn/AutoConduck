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
def scaled_cost(model, config=None):
    configured = [str(e.get("id") or e.get("model_name") or e.get("model")) for e in (getattr(config, "model_list", []) or []) if isinstance(e, dict)]
    names = set(configured) | set(_FALLBACK) | set(_COSTS)
    costs = [_entry(m, config).get("price_in", 0) + _entry(m, config).get("price_out", 0) for m in names]
    value = _entry(model, config).get("price_in", 0) + _entry(model, config).get("price_out", 0)
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
    candidates.sort(key=lambda model: scaled_cost(model, config))
    if pseudo_model.endswith("expensive") and candidates: return candidates[-1]
    return candidates[0] if candidates else ""

def select_model_by_tier(tier, config):
    entries = [e for e in (getattr(config, "model_list", []) or []) if isinstance(e, dict) and e.get("enabled", True) is not False]
    names = [e.get("id") or e.get("model_name") or e.get("model") for e in entries]
    names = [str(n) for n in names if n]
    if not names: return ""
    ranked = sorted(names, key=lambda model: scaled_cost(model, config))
    index = {"cheap": 0, "mid": len(ranked) // 2, "expensive": -1}.get(tier, len(ranked) // 2)
    return ranked[index]
