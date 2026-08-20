"""Pure budget-to-routing tuning calculations.

This module deliberately has no configuration, filesystem, statistics, or LLM
side effects. Prices are USD per million tokens.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

EPSILON = 0.001
TOKENS_PER_MINUTE = 3000
DEFAULT_BANDS = {
    "planner": [0.55, 0.85],
    "subagent": [0.10, 0.55],
    "executor": [0.35, 0.70],
}


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
    return str(
        e
        if isinstance(e, str)
        else e.get("id") or e.get("model_name") or e.get("model") or ""
    )


def _enabled(pool: list[Any]) -> list[dict[str, Any]]:
    return [
        e if isinstance(e, dict) else {"id": e}
        for e in pool
        if _name(e) and (not isinstance(e, dict) or e.get("enabled", True) is not False)
    ]


def blended_price(model: dict[str, Any], io_ratio: float = 3.0) -> float:
    ratio = io_ratio if io_ratio > 0 else 3.0
    return (
        ratio * float(model.get("price_in", 0) or 0)
        + float(model.get("price_out", 0) or 0)
    ) / (ratio + 1)


def _shares(records, names):
    counts = dict.fromkeys(names, 0)
    for row in records or []:
        n = str(row.get("model", ""))
        if n in counts:
            counts[n] += 1
    total = sum(counts.values())
    return {n: (counts[n] / total if total else 1 / len(names)) for n in names}


def token_to_usd(tokens: float, pool: list[Any], io_ratio: float = 3.0) -> float:
    entries = _enabled(pool)
    if not entries:
        return 0.0
    return (
        float(tokens)
        / 1_000_000
        * sum(blended_price(e, io_ratio) for e in entries)
        / len(entries)
    )


def _defaults() -> dict[str, Any]:
    return {
        "value_to_cost_gamma": 1.0,
        "pseudo_bias_budget": -0.20,
        "pseudo_bias_expensive": 0.20,
        "phase_bands": {k: list(v) for k, v in DEFAULT_BANDS.items()},
        "ema_alpha": 0.1,
        "quality_min_success_rate": 0.5,
        "spend_guard_max_usd_per_min": 0.20,
        "ambiguous_low": 0.60,
        "ambiguous_high": 0.75,
        "fast_path_max_scaled_cost": 0.50,
    }


def compute_tuning(
    inputs: SimpleInputs, pool: list[Any], *, stats_records=None, current=None
) -> TuneResult:
    entries = _enabled(pool)
    names = [_name(e) for e in entries]
    warnings: list[str] = []
    ratio = inputs.input_output_ratio if inputs.input_output_ratio > 0 else 3.0
    if inputs.unit == "tokens":
        if inputs.input_output_ratio <= 0:
            warnings.append("Unknown input/output ratio; using labeled 3:1 default.")
        monthly = token_to_usd(inputs.monthly_limit, entries, ratio)
    else:
        monthly = max(0.0, float(inputs.monthly_limit))
    target = monthly * (1 - float(inputs.headroom_pct) / 100)
    prices = [blended_price(e, ratio) for e in entries] or [EPSILON]
    positive = [p for p in prices if p > 0]
    cmin, cmax = min(prices), max(positive or [EPSILON])
    cmin = cmin if cmin > 0 else EPSILON
    hours = max(float(inputs.active_hours_per_month), 1e-9)
    rate = target / (hours * 60)
    target_position = max(EPSILON, rate * 1_000_000 / TOKENS_PER_MINUTE)
    denominator = math.log1p(cmax) - math.log1p(cmin)
    pressure = (
        1.0
        if denominator <= 0
        else max(
            0.0,
            min(1.0, (math.log1p(cmax) - math.log1p(target_position)) / denominator),
        )
    )
    cheapest_rate = cmin / 1_000_000 * TOKENS_PER_MINUTE
    if target < cheapest_rate * hours * 60:
        warnings.append(
            "Target is unreachable: even continuous use of the cheapest model exceeds it; using tightest best-effort guard."
        )
    if not entries:
        warnings.append("No enabled models with prices; using epsilon assumptions.")
    old = _defaults()
    current = current or old
    fast_max_scaled = round(max(0.15, min(0.75, 0.65 - pressure * 0.40)), 3)
    # High pressure (tight budget) -> steep gamma (favors cheap models), negative bias, lower phase bands, lower fast path cap
    new = {
        "spend_guard_max_usd_per_min": max(0.001, rate * inputs.burst_factor),
        "value_to_cost_gamma": 1.0 + pressure * 2.0,
        "pseudo_bias_budget": -0.20 - pressure * 0.20,
        "pseudo_bias_expensive": 0.20 - pressure * 0.35,
        "ema_alpha": 0.10 + pressure * 0.10,
        "quality_min_success_rate": 0.5,
        "ambiguous_low": 0.60 + pressure * 0.05,
        "ambiguous_high": 0.75 + pressure * 0.05,
        "fast_path_max_scaled_cost": fast_max_scaled,
    }
    bands = {}
    shifts = {"planner": 0.20, "subagent": 0.20, "executor": 0.25}
    for k, band in DEFAULT_BANDS.items():
        lo, hi = band[0] - pressure * shifts[k], band[1] - pressure * shifts[k]
        lo, hi = max(0.02, lo), max(0.02 + 0.05, hi)
        if hi - lo < 0.05:
            hi = min(1.0, lo + 0.05)
            lo = max(0.02, hi - 0.05)
        bands[k] = [round(lo, 3), round(hi, 3)]
    if len(entries) == 1:
        warnings.append(
            "Single-model pool: pool-relative tunables remain at defaults and no per-model override is written."
        )
        new.update(
            {
                "value_to_cost_gamma": old["value_to_cost_gamma"],
                "pseudo_bias_budget": old["pseudo_bias_budget"],
                "pseudo_bias_expensive": old["pseudo_bias_expensive"],
                "phase_bands": old["phase_bands"],
                "fast_path_max_scaled_cost": old["fast_path_max_scaled_cost"],
            }
        )
    else:
        new["phase_bands"] = bands
    limits = {}
    if len(entries) > 1:
        lo_log, hi_log = math.log1p(cmin), math.log1p(cmax)
        span = hi_log - lo_log
        for e, price in zip(entries, prices):
            weight = (
                1.0
                if price <= 0 or span <= 0
                else max(0.0, min(1.0, 1 - (math.log1p(price) - lo_log) / span))
            )
            limits[_name(e)] = new["spend_guard_max_usd_per_min"] * (0.3 + 0.7 * weight)
    changes = {
        k: (current.get(k, old.get(k)), v)
        for k, v in new.items()
        if current.get(k, old.get(k)) != v
    }
    return TuneResult(pressure, target, rate, changes, limits, warnings)


def project_spend(tunables, pool, stats_records=None):
    entries = _enabled(pool)
    names = [_name(e) for e in entries]
    observed = _shares(stats_records, names) if stats_records else {}
    prices = {n: blended_price(e) for n, e in zip(names, entries)}
    if not observed and names:
        weights = {n: 1 / max(prices[n], EPSILON) for n in names}
        total = sum(weights.values())
        observed = {n: weights[n] / total for n in names}
    requests = tunables.get("expected_requests_per_month") or 0
    rows = [
        {
            "model": n,
            "share": observed.get(n, 0),
            "requests_per_month": requests * observed.get(n, 0),
            "cost_per_month": requests * observed.get(n, 0) * prices[n] / 1_000_000,
        }
        for n in names
    ]
    return {
        "rows": rows,
        "total": sum(r["cost_per_month"] for r in rows),
        "caveat": "Projection is open-loop: demand and future model mix are unobservable.",
    }


def inputs_dict(inputs: SimpleInputs):
    return asdict(inputs)


def infer_monthly_budget(
    stats_records, *, days: int = 30, active_hours_per_month: float = 160.0
) -> float:
    """Project realized spend from recent stats forward to a monthly budget.

    Pure: no filesystem access. Reads rows produced by ``stats.record`` (each
    carries a ``ts`` ISO timestamp and a ``cost`` in USD). Uses the last
    ``days`` of data, scales linearly to 30 days, and rounds up to a dollar so
    the auto-tuner has a stable, universally-applicable default instead of a
    guessed magic number.
    """
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=max(1, int(days)))
    total = 0.0
    span_days = 0.0
    oldest = newest = None
    for row in stats_records or []:
        if not isinstance(row, dict):
            continue
        ts = row.get("ts")
        if not ts:
            continue
        try:
            when = datetime.fromisoformat(str(ts))
        except ValueError:
            continue
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        if when < cutoff:
            continue
        cost = float(row.get("cost", 0.0) or 0.0)
        total += cost
        oldest = when if oldest is None or when < oldest else oldest
        newest = when if newest is None or when > newest else newest
    if newest is None or oldest is None:
        return 0.0
    span_days = max(1.0, (newest - oldest).total_seconds() / 86400.0)
    # Linear projection to a 30-day month. If the observed window is shorter
    # than the requested horizon we still extrapolate from the realized rate;
    # the caller sees the inferred number and can override.
    monthly = total / span_days * 30.0
    return max(0.0, math.ceil(monthly))


def auto_tune_inputs(stats_records, *, days: int = 30, **fixed) -> SimpleInputs | None:
    """Build SimpleInputs straight from observed traffic, no manual budget.

    Returns None when there is not enough realized spend to project from, so
    callers can fall back to prompting the user instead of guessing.
    """
    monthly = infer_monthly_budget(stats_records, days=days)
    if monthly <= 0:
        return None
    return SimpleInputs(monthly_limit=monthly, **fixed)


@dataclass
class SearchCandidate:
    """One evaluated control configuration from the empirical search."""

    controls: dict[str, Any]
    cost_per_success: float
    total_cost: float
    successes: int
    failures: int
    eligible_requests: int
    escalation_rate: float
    feasible: bool  # True if the quality floor was satisfied


def _baseline_success_rate(stats_records) -> float:
    """Observed success share under the controls actually in effect."""
    total = 0
    ok = 0
    for row in stats_records or []:
        if not isinstance(row, dict):
            continue
        total += 1
        if bool(row.get("success", True)):
            ok += 1
    return (ok / total) if total else 0.0


def _baseline_escalation_rate(stats_records) -> float:
    """Observed slow-path share under the controls actually in effect."""
    total = 0
    slow = 0
    for row in stats_records or []:
        if not isinstance(row, dict):
            continue
        total += 1
        if str(row.get("path", "")).lower() == "slow":
            slow += 1
    return (slow / total) if total else 0.0


def search_controls(
    stats_records,
    pool,
    *,
    config,
    quality_floor_ratio: float = 1.25,
    max_candidates: int = 64,
    seed: int | None = None,
) -> list[SearchCandidate]:
    """Find control parameters that minimize cost subject to a quality floor.

    Objective: minimize cost-per-successful-task, where success is read from
    each realized stats row (``success`` field, falling back to True).  A
    candidate is *feasible* only if its simulated slow-path share stays within
    ``quality_floor_ratio`` of the baseline escalation rate observed in the
    real traffic — this is the quality floor.  Infeasible candidates are kept
    in the result but marked ``feasible=False`` so the caller can see what was
    rejected and why.

    The search is counterfactual replay: for each request we already know the
    realized cost and success for the model that *was* chosen.  We only score
    candidates whose simulated selection matches a model we have outcome data
    for; requests whose simulated model has no observed history are counted as
    ``eligible_requests`` exclusions rather than silently scored.

    Pool/coordinator: ``config`` is the live ``AppConfig`` used for
    ``pricing.select_closest``; ``pool`` is the enabled model pool.  Both are
    passed through unchanged.  No I/O, no LLM calls — this is pure offline
    calibration over recorded traffic.
    """
    from itertools import product

    from autoconduck.routing import pricing

    rows = [r for r in (stats_records or []) if isinstance(r, dict) and r.get("model")]
    if not rows:
        return []
    baseline_success = _baseline_success_rate(rows)
    # Precompute the outcome lookup: model -> list of (cost, success).  Cost
    # is normalized to USD per request so the objective is comparable across
    # models with different token economics.
    outcomes: dict[str, list[tuple[float, bool]]] = {}
    complexities: list[float | None] = []
    for r in rows:
        model = str(r.get("model"))
        cost = float(r.get("cost", 0.0) or 0.0)
        success = bool(r.get("success", True))
        outcomes.setdefault(model, []).append((cost, success))
        complexities.append(
            float(r["complexity"]) if r.get("complexity") is not None else None
        )

    names = [_name(e) for e in _enabled(pool)]
    if not names:
        return []

    # Coarse grid over the high-leverage controls that compute_tuning itself
    # bends via pressure.  Each axis is a small set of physically meaningful
    # values; the product stays small (well under max_candidates) so the
    # replay is cheap and deterministic.
    gammas = (1.0, 1.5, 2.0, 2.5)
    budget_biases = (-0.40, -0.30, -0.20)
    expensive_biases = (0.05, 0.20)
    fast_caps = (0.35, 0.45, 0.50)
    grid = list(product(gammas, budget_biases, expensive_biases, fast_caps))
    if max_candidates and len(grid) > max_candidates:
        stride = max(1, len(grid) // max_candidates)
        grid = grid[::stride]

    baseline_controls = _defaults()
    results: list[SearchCandidate] = []
    for gamma, bb, eb, cap in grid:
        controls = dict(baseline_controls)
        controls.update(
            {
                "value_to_cost_gamma": gamma,
                "pseudo_bias_budget": bb,
                "pseudo_bias_expensive": eb,
                "fast_path_max_scaled_cost": cap,
            }
        )
        # Apply candidate controls onto a throwaway selection view without
        # mutating the real config.  We build a lightweight proxy so the
        # pricing helpers see the candidate values.
        sel = _SelectionProxy(getattr(config, "selection", config), controls)
        proxy_cfg = _ConfigProxy(config, sel)

        total_cost = 0.0
        successes = 0
        failures = 0
        eligible = 0
        slow = 0
        for r, cpx in zip(rows, complexities):
            pseudo = str(r.get("pseudo_model", "autoconduck"))
            value = cpx if cpx is not None else 0.5
            try:
                chosen = pricing.select_closest(
                    names,
                    value,
                    proxy_cfg,
                    pseudo_model=pseudo,
                    max_scaled_cost=controls["fast_path_max_scaled_cost"],
                )
            except Exception:
                continue
            obs = outcomes.get(chosen)
            if not obs:
                continue
            eligible += 1
            # Use the median realized cost/success for the chosen model as the
            # replay estimate for this request — robust to per-request noise.
            cost, success = _median_outcome(obs)
            total_cost += cost
            if success:
                successes += 1
            else:
                failures += 1
            if str(r.get("path", "")).lower() == "slow":
                slow += 1
        esc_rate = (slow / len(rows)) if rows else 0.0
        cand_success_rate = (successes / eligible) if eligible else 0.0
        cost_per_success = (total_cost / successes) if successes else float("inf")
        # Quality floor: the candidate's simulated success rate must stay within
        # ``quality_floor_ratio`` of the baseline observed success rate.  This is
        # the real outcome-based quality gate — cost minimization is only
        # acceptable while it does not materially reduce task success.
        floor = baseline_success / max(1e-6, quality_floor_ratio)
        feasible = successes > 0 and (
            baseline_success <= 0 or cand_success_rate >= floor
        )
        results.append(
            SearchCandidate(
                controls=controls,
                cost_per_success=cost_per_success,
                total_cost=total_cost,
                successes=successes,
                failures=failures,
                eligible_requests=eligible,
                escalation_rate=esc_rate,
                feasible=feasible,
            )
        )
    # Rank: feasible candidates first by cost-per-success, then infeasible last.
    results.sort(key=lambda c: (0 if c.feasible else 1, c.cost_per_success))
    return results


def _median_outcome(obs: list[tuple[float, bool]]) -> tuple[float, bool]:
    """Median cost and majority success for a model's observed outcomes."""
    costs = sorted(c for c, _ in obs)
    median_cost = costs[len(costs) // 2] if costs else 0.0
    success_rate = sum(1 for _, s in obs if s) / max(1, len(obs))
    return median_cost, success_rate >= 0.5


class _SelectionProxy:
    """Read-through proxy overlaying candidate tuning values on selection."""

    def __init__(self, base, overrides: dict[str, Any]):
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "_overrides", overrides)

    def __getattr__(self, name):
        ov = object.__getattribute__(self, "_overrides")
        if name in ov:
            return ov[name]
        return getattr(object.__getattribute__(self, "_base"), name)


class _ConfigProxy:
    """Wraps an AppConfig so pricing sees the candidate SelectionConfig."""

    def __init__(self, base, selection):
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "_selection", selection)

    def __getattr__(self, name):
        if name == "selection":
            return object.__getattribute__(self, "_selection")
        return getattr(object.__getattribute__(self, "_base"), name)


def save_profile(inputs: SimpleInputs, result: TuneResult, *, path=None) -> None:
    """Persist the single active tuning profile (UI-facing convenience)."""
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    if path is None:
        from .config import home_dir

        path = home_dir() / "tune_profile.json"
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "version": 1,
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "inputs": asdict(inputs),
                "tunables": {k: v[1] for k, v in result.tunables.items()},
                "per_model_limits": result.per_model_limits,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def recalibrate_weights_from_records(
    stats_records: list[dict[str, Any]],
    current_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Refit complexity weights from historical escalation and de-escalation decisions.

    Analyzes past routing events to boost weights for signals prevalent in escalated tasks
    and dampen weights that did not correlate with high task value.
    """
    defaults = {
        "length": 0.08,
        "structural": 0.12,
        "scope_breadth": 0.12,
        "code_density": 0.05,
        "abstraction_level": 0.12,
        "uncertainty_hedge": 0.08,
        "cross_domain": 0.12,
        "task_novelty": 0.08,
        "imperative_strength": 0.15,
        "multi_step": 0.08,
    }
    weights = dict(current_weights or defaults)
    if not stats_records:
        return weights

    escalated_count = 0
    total_valid = 0
    for record in stats_records:
        if not isinstance(record, dict):
            continue
        total_valid += 1
        is_esc = bool(
            record.get("escalated")
            or str(record.get("path", "")).lower() == "slow"
            or float(record.get("complexity", 0.0) or 0.0) >= 0.75
        )
        if is_esc:
            escalated_count += 1

    if total_valid < 5:
        return weights

    # If the user's workload escalates frequently (>40%), emphasize abstraction, scope, and cross-domain
    esc_rate = escalated_count / max(1, total_valid)
    if esc_rate > 0.40:
        weights["scope_breadth"] = weights.get("scope_breadth", 0.12) * 1.25
        weights["cross_domain"] = weights.get("cross_domain", 0.12) * 1.20
        weights["abstraction_level"] = weights.get("abstraction_level", 0.12) * 1.15
        weights["imperative_strength"] = weights.get("imperative_strength", 0.15) * 1.10
    elif esc_rate < 0.15:
        # Mostly light editing
        weights["length"] = weights.get("length", 0.08) * 1.20
        weights["code_density"] = weights.get("code_density", 0.05) * 1.20

    # Normalize weights so they sum to 1.00
    total_w = sum(weights.values())
    return {k: round(v / total_w, 4) for k, v in weights.items()}


def load_profile(*, path=None) -> dict[str, Any] | None:
    import json
    from pathlib import Path

    if path is None:
        from .config import home_dir

        path = home_dir() / "tune_profile.json"
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
