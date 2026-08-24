"""Profile serialization and weight recalibration helpers for tuning."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autoconduck.tuning.engine import SimpleInputs, TuneResult


def save_profile(
    inputs: SimpleInputs, result: TuneResult, *, path: str | Path | None = None
) -> None:
    """Persist the single active tuning profile to disk."""
    if path is None:
        from autoconduck.config import home_dir

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


def load_profile(*, path: str | Path | None = None) -> dict[str, Any] | None:
    """Load the persisted tuning profile if present."""
    if path is None:
        from autoconduck.config import home_dir

        path = home_dir() / "tune_profile.json"
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def recalibrate_weights_from_records(
    stats_records: list[dict[str, Any]],
    current_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Refit complexity weights from historical escalation and de-escalation decisions."""
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

    esc_rate = escalated_count / max(1, total_valid)
    if esc_rate > 0.40:
        weights["scope_breadth"] = weights.get("scope_breadth", 0.12) * 1.25
        weights["cross_domain"] = weights.get("cross_domain", 0.12) * 1.20
        weights["abstraction_level"] = weights.get("abstraction_level", 0.12) * 1.15
        weights["imperative_strength"] = weights.get("imperative_strength", 0.15) * 1.10
    elif esc_rate < 0.15:
        weights["length"] = weights.get("length", 0.08) * 1.20
        weights["code_density"] = weights.get("code_density", 0.05) * 1.20

    total_w = sum(weights.values())
    return {k: round(v / total_w, 4) for k, v in weights.items()}
