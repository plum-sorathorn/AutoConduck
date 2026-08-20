"""Empirical tuning: outcome logging, auto-budget, and cost/quality search."""

from datetime import datetime, timedelta, timezone

import pytest

from autoconduck.tuning import (
    SearchCandidate,
    auto_tune_inputs,
    infer_monthly_budget,
    search_controls,
)

POOL = [
    {"id": "cheap", "price_in": 0.10, "price_out": 0.40},
    {"id": "mid", "price_in": 1.00, "price_out": 4.00},
    {"id": "expensive", "price_in": 5.00, "price_out": 15.00},
]


def _row(model, cost, success=True, complexity=0.5, path="FAST", days_ago=1):
    ts = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    return {
        "ts": ts,
        "model": model,
        "cost": cost,
        "success": success,
        "complexity": complexity,
        "path": path,
        "pseudo_model": "autoconduck",
    }


# ---------------------------------------------------------------------------
# Phase 0 — outcome signals are persisted on stats rows (shape contract).
# ---------------------------------------------------------------------------


def test_record_persists_success_and_complexity(tmp_path, monkeypatch):
    from autoconduck import stats
    from autoconduck.config import home_dir

    monkeypatch.setattr(home_dir, "__call__", lambda: tmp_path, raising=False)
    # Direct call to record() must accept and persist the new fields.
    stats.record(
        "FAST",
        "autoconduck",
        "cheap",
        100,
        50,
        cost=0.01,
        success=True,
        complexity=0.3,
        task_value=0.3,
    )
    rows = stats.load_records()
    assert rows
    row = rows[-1]
    assert row["success"] is True
    assert row["complexity"] == pytest.approx(0.3)
    assert row["task_value"] == pytest.approx(0.3)


def test_record_omits_complexity_when_unavailable(tmp_path, monkeypatch):
    from autoconduck import stats
    from autoconduck.config import home_dir

    monkeypatch.setattr(home_dir, "__call__", lambda: tmp_path, raising=False)
    stats.record("FAST", "autoconduck", "cheap", 100, 50, cost=0.01)
    row = stats.load_records()[-1]
    assert "complexity" not in row
    assert "success" in row  # success is always written now


# ---------------------------------------------------------------------------
# Phase 1 — auto-budget infers a monthly limit from realized spend.
# ---------------------------------------------------------------------------


def test_infer_monthly_budget_projects_realized_spend():
    # 30 rows at $0.10 each spread over 3 days -> ~$1 over 3 days -> $10/month.
    records = [_row("cheap", 0.10, days_ago=d) for d in range(3)]
    monthly = infer_monthly_budget(records, days=30)
    assert monthly >= 1  # rounded up to a dollar


def test_infer_monthly_budget_zero_without_history():
    assert infer_monthly_budget([], days=30) == 0.0
    # Rows outside the window don't count.
    old = [_row("cheap", 1.0, days_ago=90)]
    assert infer_monthly_budget(old, days=30) == 0.0


def test_auto_tune_inputs_returns_none_without_history():
    assert auto_tune_inputs([], days=30) is None


def test_auto_tune_inputs_builds_simpleinputs_from_history():
    records = [_row("cheap", 0.50, days_ago=d) for d in range(5)]
    inputs = auto_tune_inputs(records, days=30)
    assert inputs is not None
    assert inputs.monthly_limit > 0


# ---------------------------------------------------------------------------
# Phase 2 — empirical cost/quality search.
# ---------------------------------------------------------------------------


def _cfg():
    from autoconduck.config import Config

    cfg = Config()
    cfg.model_list = [{"id": m["id"]} for m in POOL]
    return cfg


def test_search_controls_returns_ranked_candidates():
    records = []
    # A mix: cheap model mostly succeeds at low cost; expensive used on slow path.
    for d in range(10):
        records.append(
            _row("cheap", 0.02, success=True, complexity=0.3, path="FAST", days_ago=d)
        )
    for d in range(10):
        records.append(
            _row(
                "expensive", 0.30, success=True, complexity=0.9, path="SLOW", days_ago=d
            )
        )
    cfg = _cfg()
    results = search_controls(records, POOL, config=cfg)
    assert results
    assert all(isinstance(c, SearchCandidate) for c in results)
    # Feasible candidates sort ahead of infeasible ones.
    feasible_flags = [c.feasible for c in results]
    assert feasible_flags == sorted(feasible_flags, reverse=True)
    # The best candidate has a finite cost-per-success when it has any successes.
    best = results[0]
    if best.successes:
        assert best.cost_per_success < float("inf")


def test_search_controls_empty_when_no_records():
    assert search_controls([], POOL, config=_cfg()) == []


def test_search_controls_feasibility_respects_quality_floor():
    # All-fast traffic: baseline escalation rate is 0.  A floor at 1.25x of 0
    # is still 0, so the feasibility gate must not reject everything on the
    # basis of dropping escalation that never existed.
    records = [
        _row("cheap", 0.02, success=True, complexity=0.2, path="FAST", days_ago=d)
        for d in range(8)
    ]
    results = search_controls(records, POOL, config=_cfg())
    assert results
    assert any(c.feasible for c in results)
