"""Offline calibration and counterfactual replay parameter search."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

from autoconduck.tuning.engine import _defaults, _enabled, _name


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


def _baseline_success_rate(stats_records: list[dict[str, Any]] | None) -> float:
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


def _baseline_escalation_rate(stats_records: list[dict[str, Any]] | None) -> float:
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


def _median_outcome(obs: list[tuple[float, bool]]) -> tuple[float, bool]:
    """Median cost and majority success for a model's observed outcomes."""
    costs = sorted(c for c, _ in obs)
    median_cost = costs[len(costs) // 2] if costs else 0.0
    success_rate = sum(1 for _, s in obs if s) / max(1, len(obs))
    return median_cost, success_rate >= 0.5


class _SelectionProxy:
    """Read-through proxy overlaying candidate tuning values on selection."""

    def __init__(self, base: Any, overrides: dict[str, Any]) -> None:
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "_overrides", overrides)

    def __getattr__(self, name: str) -> Any:
        ov = object.__getattribute__(self, "_overrides")
        if name in ov:
            return ov[name]
        return getattr(object.__getattribute__(self, "_base"), name)


class _ConfigProxy:
    """Wraps an AppConfig so pricing sees the candidate SelectionConfig."""

    def __init__(self, base: Any, selection: Any) -> None:
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "_selection", selection)

    def __getattr__(self, name: str) -> Any:
        if name == "selection":
            return object.__getattribute__(self, "_selection")
        return getattr(object.__getattribute__(self, "_base"), name)


def search_controls(
    stats_records: list[dict[str, Any]] | None,
    pool: list[Any],
    *,
    config: Any,
    quality_floor_ratio: float = 1.25,
    max_candidates: int = 64,
    seed: int | None = None,
) -> list[SearchCandidate]:
    """Find control parameters that minimize cost subject to a quality floor."""
    from autoconduck.routing import pricing

    rows = [
        r
        for r in (stats_records or [])
        if isinstance(r, dict) and r.get("model")
    ]
    if not rows:
        return []
    baseline_success = _baseline_success_rate(rows)
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
        cost_per_success = (
            (total_cost / successes) if successes else float("inf")
        )
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
    results.sort(key=lambda c: (0 if c.feasible else 1, c.cost_per_success))
    return results
