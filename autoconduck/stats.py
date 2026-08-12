"""Persistent, best-effort usage accounting."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import home_dir
from .routing import pricing

_latest_selection: dict[str, Any] = {}

def record_selection(task_value: float, target_scaled_cost: float, model: str, config) -> None:
    if getattr(getattr(config, "selection", None), "expose_value_in_stats", True):
        _latest_selection.update(last_task_value=task_value, last_target_scaled_cost=target_scaled_cost, last_selected_model=model)


def stats_path() -> Path:
    return home_dir() / "run" / "stats.jsonl"


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    try:
        entry = pricing._entry(model)
        return (prompt_tokens * float(entry.get("price_in", 0)) + completion_tokens * float(entry.get("price_out", 0))) / 1_000_000
    except Exception:
        return 0.0


def record(path: str, pseudo_model: str, model: str, prompt_tokens: int, completion_tokens: int, *, cost: float | None = None, success: bool = True) -> None:
    try:
        prompt_tokens, completion_tokens = int(prompt_tokens), int(completion_tokens)
        pricing.record_usage(model, prompt_tokens, completion_tokens, cost=cost, success=success)
        row = {"ts": datetime.now(timezone.utc).isoformat(), "path": path, "pseudo_model": pseudo_model,
               "model": model, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
               "cost": cost if cost is not None else estimate_cost(model, prompt_tokens, completion_tokens)}
        target = stats_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    except Exception:
        pass


def load_records(limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with stats_path().open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                    if isinstance(row, dict): rows.append(row)
                except (ValueError, TypeError):
                    continue
    except OSError:
        return []
    return rows[-limit:] if limit is not None else rows


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0}
    models: dict[str, dict[str, Any]] = {}
    paths: dict[str, int] = {}
    pseudos: dict[str, int] = {}
    for row in records:
        p, c = int(row.get("prompt_tokens", 0) or 0), int(row.get("completion_tokens", 0) or 0)
        cost = float(row.get("cost", 0) or 0)
        totals["calls"] += 1; totals["prompt_tokens"] += p; totals["completion_tokens"] += c; totals["total_tokens"] += p + c; totals["cost"] += cost
        model = str(row.get("model", "unknown")); item = models.setdefault(model, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0})
        item["calls"] += 1; item["prompt_tokens"] += p; item["completion_tokens"] += c; item["total_tokens"] += p + c; item["cost"] += cost
        paths[str(row.get("path", "unknown"))] = paths.get(str(row.get("path", "unknown")), 0) + 1
        pseudos[str(row.get("pseudo_model", "unknown"))] = pseudos.get(str(row.get("pseudo_model", "unknown")), 0) + 1
    result = {"totals": totals, "models": dict(sorted(models.items(), key=lambda x: (-x[1]["cost"], -x[1]["total_tokens"], x[0]))), "paths": paths, "pseudos": pseudos}
    result.update(_latest_selection)
    return result


def render_table(agg: dict[str, Any]) -> str:
    total = agg["totals"]
    lines = ["MODEL | CALLS | PROMPT TOK | COMPL TOK | TOTAL TOK | EST COST ($)", "-" * 76]
    for model, row in agg["models"].items():
        lines.append(f"{model} | {row['calls']} | {row['prompt_tokens']} | {row['completion_tokens']} | {row['total_tokens']} | {row['cost']:.4f}")
    lines.append(f"TOTAL | {total['calls']} | {total['prompt_tokens']} | {total['completion_tokens']} | {total['total_tokens']} | {total['cost']:.4f}")
    if total["cost"] == 0:
        lines.append("EST COST: n/a (no price data for these models)")
    lines.append("PATHS: " + ", ".join(f"{k}={v}" for k, v in sorted(agg["paths"].items())))
    lines.append("PSEUDOS: " + ", ".join(f"{k}={v}" for k, v in sorted(agg["pseudos"].items())))
    return "\n".join(lines)


def render_json(agg: dict[str, Any]) -> str:
    return json.dumps(agg, indent=2)


def _usage(obj: Any) -> tuple[int, int]:
    usage = getattr(obj, "usage", None)
    if isinstance(obj, dict): usage = obj.get("usage", usage)
    if isinstance(usage, dict): return int(usage.get("prompt_tokens", 0) or 0), int(usage.get("completion_tokens", 0) or 0)
    return int(getattr(usage, "prompt_tokens", 0) or 0), int(getattr(usage, "completion_tokens", 0) or 0)


def install_recorder(llm: Any) -> None:
    original = getattr(llm, "acompletion", None)
    if original is None or getattr(original, "_autoconduck_recorder", False): return
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        path = kwargs.pop("_path", "unknown")
        pseudo = kwargs.pop("_pseudo", "unknown")
        model = str(kwargs.get("model", "unknown"))
        try:
            if not kwargs.get("stream"):
                result = await original(*args, **kwargs)
                p, c = _usage(result)
                hidden = getattr(result, "_hidden_params", {}) or {}
                record(path, pseudo, model, p, c, cost=hidden.get("response_cost"))
                return result
            options = dict(kwargs.get("stream_options") or {}); options.setdefault("include_usage", True); kwargs["stream_options"] = options
            response = await original(*args, **kwargs)
            prompt = completion = 0
            final_usage: tuple[int, int] | None = None
            async def relay():
                nonlocal prompt, completion, final_usage
                try:
                    async for chunk in response:
                        p, c = _usage(chunk); prompt = max(prompt, p); completion += c
                        if _usage(chunk) != (0, 0):
                            final_usage = _usage(chunk)
                        yield chunk
                    record(path, pseudo, model, *(final_usage or (prompt, completion)))
                except Exception:
                    pricing.record_error(model)
                    raise
            return relay()
        except Exception:
            pricing.record_error(model)
            raise
    wrapped._autoconduck_recorder = True
    llm.acompletion = wrapped
