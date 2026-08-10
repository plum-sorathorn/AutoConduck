"""Pure budget-to-routing tuning calculations.

This module deliberately has no configuration, filesystem, statistics, or LLM
side effects. Prices are USD per million tokens.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
from typing import Any

EPSILON = .001
TOKENS_PER_MINUTE = 3000
DEFAULT_BANDS = {"planner": [.55, .85], "subagent": [.10, .55], "executor": [.35, .70]}

@dataclass(frozen=True)
class SimpleInputs:
    monthly_limit: float
    unit: str = "usd"
    headroom_pct: float = 25.0
    active_hours_per_month: float = 160.0
    expected_requests_per_month: int | None = None
    input_output_ratio: float = 3.0
    burst_factor: float = 1.8

@dataclass
class TuneResult:
    pressure: float
    target_monthly_spend: float
    rate_per_min: float
    tunables: dict[str, tuple[Any, Any]]
    per_model_limits: dict[str, float]
    warnings: list[str]

def _name(e: Any) -> str:
    return str(e if isinstance(e, str) else e.get("id") or e.get("model_name") or e.get("model") or "")

def _enabled(pool: list[Any]) -> list[dict[str, Any]]:
    return [e if isinstance(e, dict) else {"id": e} for e in pool if _name(e) and (not isinstance(e, dict) or e.get("enabled", True) is not False)]

def blended_price(model: dict[str, Any], io_ratio: float = 3.0) -> float:
    ratio = io_ratio if io_ratio > 0 else 3.0
    return (ratio * float(model.get("price_in", 0) or 0) + float(model.get("price_out", 0) or 0)) / (ratio + 1)

def _shares(records, names):
    counts = {n: 0 for n in names}
    for row in records or []:
        n = str(row.get("model", ""))
        if n in counts: counts[n] += 1
    total = sum(counts.values())
    return {n: (counts[n] / total if total else 1 / len(names)) for n in names}

def token_to_usd(tokens: float, pool: list[Any], io_ratio: float = 3.0) -> float:
    entries = _enabled(pool)
    if not entries: return 0.0
    return float(tokens) / 1_000_000 * sum(blended_price(e, io_ratio) for e in entries) / len(entries)

def _defaults() -> dict[str, Any]:
    return {"value_to_cost_gamma": 1.0, "pseudo_bias_budget": -.20, "pseudo_bias_expensive": .20,
            "phase_bands": {k: list(v) for k, v in DEFAULT_BANDS.items()}, "ema_alpha": .1,
            "quality_min_success_rate": .5, "spend_guard_max_usd_per_min": .20,
            "ambiguous_low": .55, "ambiguous_high": .70}

def compute_tuning(inputs: SimpleInputs, pool: list[Any], *, stats_records=None, current=None) -> TuneResult:
    entries = _enabled(pool); names = [_name(e) for e in entries]
    warnings: list[str] = []
    ratio = inputs.input_output_ratio if inputs.input_output_ratio > 0 else 3.0
    if inputs.unit == "tokens":
        if inputs.input_output_ratio <= 0: warnings.append("Unknown input/output ratio; using labeled 3:1 default.")
        monthly = token_to_usd(inputs.monthly_limit, entries, ratio)
    else: monthly = max(0.0, float(inputs.monthly_limit))
    target = monthly * (1 - float(inputs.headroom_pct) / 100)
    prices = [blended_price(e, ratio) for e in entries] or [EPSILON]
    positive = [p for p in prices if p > 0]
    cmin, cmax = min(prices), max(positive or [EPSILON])
    cmin = cmin if cmin > 0 else EPSILON
    hours = max(float(inputs.active_hours_per_month), 1e-9)
    rate = target / (hours * 60)
    target_position = max(EPSILON, rate * 1_000_000 / TOKENS_PER_MINUTE)
    denominator = math.log1p(cmax) - math.log1p(cmin)
    pressure = 1.0 if denominator <= 0 else max(0.0, min(1.0, (math.log1p(cmax) - math.log1p(target_position)) / denominator))
    cheapest_rate = cmin / 1_000_000 * TOKENS_PER_MINUTE
    if target < cheapest_rate * hours * 60: warnings.append("Target is unreachable: even continuous use of the cheapest model exceeds it; using tightest best-effort guard.")
    if not entries: warnings.append("No enabled models with prices; using epsilon assumptions.")
    old = _defaults(); current = current or old
    # High pressure (tight budget) -> steep gamma (favors cheap models), negative bias, lower phase bands
    new = {"spend_guard_max_usd_per_min": max(.001, rate * inputs.burst_factor),
           "value_to_cost_gamma": 1.0 + pressure * 2.0,
           "pseudo_bias_budget": -.20 - pressure * .20,
           "pseudo_bias_expensive": .20 - pressure * .35,
           "ema_alpha": .10 + pressure * .10,
           "quality_min_success_rate": .5,
           "ambiguous_low": .55 + pressure * .05,
           "ambiguous_high": .70 + pressure * .05}
    bands = {}
    shifts = {"planner": .20, "subagent": .20, "executor": .25}
    for k, band in DEFAULT_BANDS.items():
        lo, hi = band[0] - pressure * shifts[k], band[1] - pressure * shifts[k]
        lo, hi = max(.02, lo), max(.02 + .05, hi)
        if hi - lo < .05: hi = min(1.0, lo + .05); lo = max(.02, hi - .05)
        bands[k] = [round(lo, 3), round(hi, 3)]
    if len(entries) == 1:
        warnings.append("Single-model pool: pool-relative tunables remain at defaults and no per-model override is written.")
        new.update({"value_to_cost_gamma": old["value_to_cost_gamma"], "pseudo_bias_budget": old["pseudo_bias_budget"], "pseudo_bias_expensive": old["pseudo_bias_expensive"], "phase_bands": old["phase_bands"]})
    else: new["phase_bands"] = bands
    limits = {}
    if len(entries) > 1:
        lo_log, hi_log = math.log1p(cmin), math.log1p(cmax)
        span = hi_log - lo_log
        for e, price in zip(entries, prices):
            weight = 1.0 if price <= 0 or span <= 0 else max(0., min(1., 1 - (math.log1p(price) - lo_log) / span))
            limits[_name(e)] = new["spend_guard_max_usd_per_min"] * (.3 + .7 * weight)
    changes = {k: (current.get(k, old.get(k)), v) for k, v in new.items() if current.get(k, old.get(k)) != v}
    return TuneResult(pressure, target, rate, changes, limits, warnings)

def project_spend(tunables, pool, stats_records=None):
    entries = _enabled(pool); names = [_name(e) for e in entries]
    observed = _shares(stats_records, names) if stats_records else {}
    prices = {n: blended_price(e) for n, e in zip(names, entries)}
    if not observed and names:
        weights = {n: 1 / max(prices[n], EPSILON) for n in names}; total = sum(weights.values()); observed = {n: weights[n] / total for n in names}
    requests = tunables.get("expected_requests_per_month") or 0
    rows = [{"model": n, "share": observed.get(n, 0), "requests_per_month": requests * observed.get(n, 0), "cost_per_month": requests * observed.get(n, 0) * prices[n] / 1_000_000} for n in names]
    return {"rows": rows, "total": sum(r["cost_per_month"] for r in rows), "caveat": "Projection is open-loop: demand and future model mix are unobservable."}

def inputs_dict(inputs: SimpleInputs): return asdict(inputs)

def save_profile(inputs: SimpleInputs, result: TuneResult, *, path=None) -> None:
    """Persist the single active tuning profile (UI-facing convenience)."""
    import json
    from datetime import datetime, timezone
    from pathlib import Path
    if path is None:
        from .config import home_dir
        path = home_dir() / "tune_profile.json"
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"version": 1, "saved_at": datetime.now(timezone.utc).isoformat(),
        "inputs": asdict(inputs), "tunables": {k: v[1] for k, v in result.tunables.items()},
        "per_model_limits": result.per_model_limits}, indent=2), encoding="utf-8")

def load_profile(*, path=None) -> dict[str, Any] | None:
    import json
    from pathlib import Path
    if path is None:
        from .config import home_dir
        path = home_dir() / "tune_profile.json"
    try: return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError): return None
