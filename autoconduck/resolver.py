"""Shared business logic for model resolution and LiteLLM dispatch.

These functions are pure (no FastAPI/Pydantic deps) so they can be imported
cheaply and monkeypatched by tests that want to control routing behaviour
without needing a running server.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any


# Lightweight decision record holder
class _RoutingDecision:
    """Container for one routing decision."""
    decisions: list[dict[str, Any]] = []
    count: int = 0
    last_time: float = 0.0


_decisions = _RoutingDecision()


def record_decision(path: str, model: str | None) -> None:
    """Record a routing decision for /stats."""
    _decisions.count += 1
    _decisions.last_time = time.time()
    _decisions.decisions.append({"path": path, "model": model or "", "time": time.time()})


async def resolve_model(
    body_model: str,
    messages: list[dict],
    on_progress=None,
) -> tuple[str | None, dict]:
    """Resolve a pseudo/custom model to a concrete target + extra kwargs.

    Returns ``(target, extra_kwargs)`` suitable for passing to litellm.acompletion.
    """
    cfg = _get_config()
    target = body_model

    if body_model in _PSEUDO_SET:
        path, model = await _do_router_dispatch(messages, body_model, cfg)
        if on_progress is not None:
            try:
                on_progress({"kind": "route", "path": path})
            except Exception:
                pass
        record_decision(path, model)
        if path == "SLOW" and model is None:
            answer = await _do_slow_route(messages, body_model, on_progress=on_progress)
            if answer is not None:
                return None, {"__answer__": answer}
            # Fall through to FAST model selection below

        if model is None:
            model = _pick_fast_model(body_model, cfg)

        target = model

    return target, _litellm_extra(target, cfg)


async def call_model(model: str, body: dict) -> dict:
    """Non-streaming call through litellm. Used by tests & non-stream routes."""
    llm = _get_litellm()
    if llm is None:
        raise RuntimeError("litellm unavailable")

    kwargs: dict[str, Any] = dict(body)
    kwargs.pop("stream", None)
    kwargs["model"] = model
    cfg = _get_config()
    kwargs.update(_litellm_extra(model, cfg))

    result = await llm.acompletion(**kwargs)
    return result.model_dump() if hasattr(result, "model_dump") else result


# ---- Internal helpers ----


def _get_config():
    from .config import get_config
    return get_config()


def _get_litellm():
    try:
        import litellm
        return litellm
    except ImportError:
        return None


_PSEUDO_SET = frozenset({"autoconduck", "autoconduck-budget", "autoconduck-expensive"})


async def _do_router_dispatch(messages, body_model, cfg):
    from .routing.dispatcher import route
    try:
        decision = await asyncio.to_thread(route, messages, [], pseudo_model=body_model, config=cfg)
        return getattr(decision, "path", "FAST").upper(), getattr(decision, "model", None)
    except Exception:
        return "FAST", None


async def _do_slow_route(messages, body_model, on_progress=None, plan=None):
    try:
        if plan is None:
            from autoconduck.routing.slm_planner import SLMPlanner

            planner = SLMPlanner()
            plan = await planner.plan(messages)

        if on_progress is not None:
            try:
                tier_name = getattr(
                    plan.suggested_tier, "value", str(plan.suggested_tier)
                )
                subtasks_cnt = len(getattr(plan, "subtasks", []))
                on_progress(
                    f"⚡ SLM Plan generated ({tier_name}): {subtasks_cnt} subtasks"
                )
            except Exception:
                pass

        from autoconduck.orchestrator.dynamic_factory import (
            DynamicState,
            build_dynamic_graph,
        )

        runner = build_dynamic_graph(plan)
        initial_state = DynamicState(
            session_id="session_resolver",
            thread_id="thread_resolver",
            plan=plan,
            messages=messages,
        )
        res = await runner.ainvoke(initial_state)
        final_res = (
            getattr(res, "final_result", None)
            if not isinstance(res, dict)
            else res.get("final_result")
        )
        if final_res:
            return final_res
        synth = (
            getattr(res, "synthesizer_output", None)
            if not isinstance(res, dict)
            else res.get("synthesizer_output")
        )
        return {"content": synth or "Task completed."}
    except Exception:
        return None


def _pick_fast_model(body_model, cfg):
    try:
        from .routing.pricing import select_closest, pool_ids
        from .config import resolve_orchestrator_model
        selection = getattr(cfg, "selection", cfg)
        max_fast_cost = getattr(selection, "fast_path_max_scaled_cost", 0.50)
        try:
            max_fast_cost = float(max_fast_cost)
        except (TypeError, ValueError):
            max_fast_cost = 0.50
        return (select_closest(pool_ids(cfg), .15, cfg, pseudo_model=body_model, max_scaled_cost=max_fast_cost)
                or resolve_orchestrator_model(cfg))
    except Exception:
        from .config import resolve_orchestrator_model
        return resolve_orchestrator_model(cfg)


def _litellm_extra(target: str, cfg) -> dict:
    from autoconduck.server.messages_api import litellm_params_for
    return litellm_params_for(target, cfg)
