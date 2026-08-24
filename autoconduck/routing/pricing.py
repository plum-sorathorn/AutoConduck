import json, math, time
from collections import defaultdict, deque
from pathlib import Path

try:
    import litellm

    _COSTS = litellm.model_cost
except Exception:
    _COSTS = {}
try:
    _FALLBACK = json.loads(
        (Path(__file__).with_name("pricing_fallback.json")).read_text()
    )
except Exception:
    _FALLBACK = {}
_errors = defaultdict(deque)
_ema = {}
_spend = defaultdict(deque)
_last_usage = {}


def _pool_id(entry):
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        return entry.get("id") or entry.get("model_name") or entry.get("model")
    return None


def _configured_entry(model, config):
    from ..config import _configured_model_sources

    wanted = str(model).removeprefix("openai/")
    for entry in _configured_model_sources(config):
        if not isinstance(entry, dict):
            continue
        name = entry.get("id") or entry.get("model_name") or entry.get("model")
        if isinstance(entry.get("litellm_params"), dict):
            name = name or entry["litellm_params"].get("model")
        if str(name or "").removeprefix("openai/") == wanted:
            params = (
                entry.get("litellm_params")
                if isinstance(entry.get("litellm_params"), dict)
                else entry
            )
            if isinstance(params, dict) and any(
                key in params
                for key in (
                    "price_in",
                    "price_out",
                    "quality_score",
                    "max_usd_per_min",
                )
            ):
                return params
    return None


def _entry(model, config=None):
    raw = (
        _configured_entry(model, config)
        or _COSTS.get(model)
        or _COSTS.get(str(model).removeprefix("openai/"))
        or _FALLBACK.get(model)
        or _FALLBACK.get(str(model).removeprefix("openai/"))
        or {}
    )
    return {
        "price_in": raw.get("input_cost_per_token", raw.get("price_in", 0.0))
        * (1_000_000 if "input_cost_per_token" in raw else 1),
        "price_out": raw.get("output_cost_per_token", raw.get("price_out", 0.0))
        * (1_000_000 if "output_cost_per_token" in raw else 1),
        **raw,
    }


def is_subscription(model):
    return bool(_FALLBACK.get(model, {}).get("subscription", False))


def _entry_effective_value(model, config):
    """Return a comparable effective price in USD per million tokens.

    Pricing & Quality Accounting:
      - Static catalog price: price_in + price_out (USD per million tokens).
      - Realized usage EMA: Request cost converted back to USD per million tokens.
      - Variance-aware sample blending: EMA is weighted by sample count and damped
        by observed cost variance so bursty outliers don't destabilize routing.
      - Success rate penalty: Measured as successful requests / attempts; failure
        events (HTTP 5xx, rate limits, fatal errors) penalize effective cost via
        ``observed / max(success_rate, quality_min_success_rate)``.
      - Model quality score: Configured ``quality_score`` acts as an efficiency divisor.
    """
    entry = _entry(model, config)
    ema = _ema.get(str(model))
    selection = getattr(config, "selection", config)
    minimum = getattr(selection, "ema_min_samples", 3)
    base = float(entry.get("price_in", 0)) + float(entry.get("price_out", 0))
    if ema and ema["samples"] >= minimum:
        tokens = max(1.0, float(ema.get("prompt", 0)) + float(ema.get("completion", 0)))
        observed = float(ema.get("cost", 0)) * 1_000_000 / tokens
        if observed > 0:
            rate = float(
                ema.get("success_rate", ema["successes"] / max(1, ema["attempts"]))
            )
            ema_rate = observed / max(
                rate, float(getattr(selection, "quality_min_success_rate", 0.5))
            )
            # Variance-aware sample confidence blending:
            # Low sample counts (<5) or high relative cost variance damp the EMA
            # shift towards the catalog baseline.
            samples = int(ema.get("samples", 1))
            cost_var = float(ema.get("cost_variance", 0.0))
            sample_weight = min(1.0, samples / max(5, minimum + 2))
            var_damp = 1.0 / (1.0 + math.sqrt(max(0.0, cost_var)) / max(0.01, (base / 1_000_000) or 0.01))
            confidence = sample_weight * var_damp
            base = (1.0 - confidence) * base + confidence * ema_rate
    quality = entry.get("quality_score", 1.0)
    try:
        quality = float(quality)
    except (TypeError, ValueError):
        quality = 1.0
    return base / quality if 0 < quality <= 1 else base


def _scaled_costs(models, config=None, *, use_ema=True):
    """Scale a model set once, avoiding repeated pool-wide work while ranking."""
    names = [str(model) for model in models]
    values = {}
    for name in names:
        if use_ema:
            values[name] = _entry_effective_value(name, config)
        else:
            entry = _entry(name, config)
            values[name] = float(entry.get("price_in", 0)) + float(
                entry.get("price_out", 0)
            )
    denominator = max(
        1.0,
        max((math.log1p(max(0.0, value)) for value in values.values()), default=1.0),
    )
    return {
        name: math.log1p(max(0.0, value)) / denominator
        for name, value in values.items()
    }


def scaled_cost(model, config=None, *, use_ema=True):
    configured = [
        str(name)
        for entry in (getattr(config, "model_list", []) or [])
        if (name := _pool_id(entry))
    ]
    catalog = list(_FALLBACK) + list(_COSTS)
    names = configured or list(dict.fromkeys(catalog + [str(model)]))
    costs = _scaled_costs(names, config, use_ema=use_ema)
    if str(model) not in costs:
        names.append(str(model))
        costs = _scaled_costs(names, config, use_ema=use_ema)
    return costs[str(model)]


def is_degraded(model, window_s=300, error_rate=0.2):
    now = time.time()
    q = _errors[str(model)]
    while q and now - q[0][0] > window_s:
        q.popleft()
    return len(q) > 0 and sum(x[1] for x in q) / len(q) > error_rate


def record_error(model):
    _errors[model].append((time.time(), 1))
    record_usage(model, 0, 0, success=False)


def record_usage(model, prompt_tokens, completion_tokens, *, cost=None, success=True):
    """Record observed token usage, cost, and success status for variance-aware EMA."""
    import re
    prompt_tokens, completion_tokens = int(prompt_tokens), int(completion_tokens)
    if cost is None:
        entry = _entry(model)
        cost = (
            prompt_tokens * float(entry.get("price_in", 0))
            + completion_tokens * float(entry.get("price_out", 0))
        ) / 1_000_000
    cost = max(0.0, float(cost))
    old = _ema.get(model)
    alpha = 0.1
    try:
        from ..config import get_config

        alpha = float(
            getattr(getattr(get_config(), "selection", None), "ema_alpha", 0.1)
        )
    except Exception:
        pass
    old_cost = old["cost"] if old else cost
    cost_variance = (
        (1 - alpha) * old.get("cost_variance", 0.0) + alpha * ((cost - old_cost) ** 2)
        if old
        else 0.0
    )
    _ema[model] = {
        "prompt": prompt_tokens
        if not old
        else (1 - alpha) * old["prompt"] + alpha * prompt_tokens,
        "completion": completion_tokens
        if not old
        else (1 - alpha) * old["completion"] + alpha * completion_tokens,
        "cost": cost if not old else (1 - alpha) * old["cost"] + alpha * cost,
        "cost_variance": cost_variance,
        "samples": old["samples"] + 1 if old else 1,
        "attempts": old["attempts"] + 1 if old else 1,
        "successes": old["successes"] + (1 if success else 0)
        if old
        else (1 if success else 0),
        "success_rate": (1 - alpha) * old.get("success_rate", 1.0)
        + alpha * (1.0 if success else 0.0)
        if old
        else (1.0 if success else 0.0),
    }
    _spend[str(model)].append((time.time(), cost))
    _last_usage[model] = (prompt_tokens, completion_tokens)


def select(pool, pseudo_model, config, usage=None):
    degraded = {
        m
        for m in pool_ids_from(pool)
        if is_degraded(
            m,
            getattr(config, "degraded_window_s", 300),
            getattr(config, "degraded_error_rate", 0.2),
        )
    }
    return select_closest(
        pool, 0.15, config, pseudo_model=pseudo_model, degraded=degraded
    )


def select_for_tier(
    tier: str | Any,
    config: Any = None,
    *,
    min_context_window: int = 0,
    requires_tools: bool = False,
    pseudo_model: str = "autoconduck",
) -> str:
    """Select a model matching the requested tier via ModelPool or select_closest."""
    try:
        from autoconduck.routing.model_pool import ModelPool
        pool = ModelPool(config=config)
        return pool.select_for_tier(
            tier,
            min_context_window=min_context_window,
            requires_tools=requires_tools,
            pseudo_model=pseudo_model,
        )
    except Exception:
        tier_val = 0.5
        t_str = str(getattr(tier, "value", tier)).lower()
        if "cheap" in t_str or "fast" in t_str or "budget" in t_str:
            tier_val = 0.15
        elif "frontier" in t_str or "reason" in t_str or "expensive" in t_str:
            tier_val = 0.85
        return select_closest(pool_ids(config), tier_val, config, pseudo_model=pseudo_model)


def select_model_by_tier(tier: str | Any, config: Any = None) -> str:
    """Legacy alias for select_for_tier."""
    return select_for_tier(tier, config)


def pool_ids(config):
    from ..config import _configured_model_sources

    return [
        str(n)
        for e in _configured_model_sources(config)
        if (n := _pool_id(e))
        and (not isinstance(e, dict) or e.get("enabled", True) is not False)
    ]


def pool_ids_from(pool):
    return [str(n) for e in pool if (n := _pool_id(e))]


def cheapest_enabled(config):
    names = pool_ids(config)
    if not names:
        return ""
    eligible = [m for m in names if not is_over_budget(m, config)] or names
    return min(eligible, key=lambda m: (_entry_effective_value(m, config), m))


def is_over_budget(model, config):
    selection = getattr(config, "selection", config)
    if not getattr(selection, "spend_guard_enabled", True):
        return False
    window = max(1, int(getattr(selection, "spend_guard_window_s", 300)))
    now = time.time()
    q = _spend[str(model)]
    while q and now - q[0][0] > window:
        q.popleft()
    entry = _entry(model, config)
    cap = entry.get(
        "max_usd_per_min", getattr(selection, "spend_guard_max_usd_per_min", 0.20)
    )
    try:
        cap = float(cap)
    except (TypeError, ValueError):
        return False
    return cap > 0 and sum(cost for _, cost in q) / window * 60 > cap


def target_scaled_cost(value, pseudo_model, config):
    import re
    sel = getattr(config, "selection", config)
    gamma = float(getattr(sel, "value_to_cost_gamma", 1.0))
    bias = float(getattr(sel, "default_target_bias", 0.0))
    if getattr(sel, "pseudo_bias_enabled", True):
        if pseudo_model == "autoconduck-budget":
            bias = float(getattr(sel, "pseudo_bias_budget", -0.2))
        elif pseudo_model == "autoconduck-expensive":
            bias = float(getattr(sel, "pseudo_bias_expensive", 0.2))
        elif pseudo_model:
            match = re.search(r"(?:bias[=:]|bias-)([-+]?[0-9]*\.?[0-9]+)", str(pseudo_model))
            if match:
                try:
                    bias = float(match.group(1))
                except ValueError:
                    pass
    return max(0.0, min(1.0, float(value) ** gamma + bias))


def is_expensive_model(model: str, config=None) -> bool:
    """Return True if the model's scaled cost exceeds the file-read ceiling (default 0.55)."""
    try:
        from ..config import get_config
        cfg = config or get_config()
        sel = getattr(cfg, "selection", cfg)
        max_cost = float(getattr(sel, "max_file_read_scaled_cost", 0.55))
        return scaled_cost(model, cfg) > max_cost
    except Exception:
        return False


def select_closest(pool, value, config, *, pseudo_model=None, band=None, degraded=None, max_scaled_cost=None):
    try:
        names = [str(_pool_id(x)) for x in pool if _pool_id(x)]
        eligible = [m for m in names if not (degraded and m in degraded)]
        non_over = [m for m in eligible if not is_over_budget(m, config)]
        eligible = non_over or eligible
        if not eligible:
            return cheapest_enabled(config)
        costs = _scaled_costs(names, config)
        if max_scaled_cost is not None:
            under_cap = [m for m in eligible if costs.get(m, 0.0) <= max_scaled_cost]
            if under_cap:
                eligible = under_cap
        if band:
            inside = [m for m in eligible if band[0] <= costs[m] <= band[1]]
            eligible = inside or eligible
        target = target_scaled_cost(value, pseudo_model, config)
        sel = getattr(config, "selection", config)
        latency_sens = float(getattr(sel, "latency_sensitivity", 0.0)) if sel else 0.0
        if latency_sens > 0 and band is None:
            target = max(0.0, target - 0.25 * latency_sens)
        return min(eligible, key=lambda m: (abs(costs[m] - target), costs[m], m))
    except Exception:
        return cheapest_enabled(config)


