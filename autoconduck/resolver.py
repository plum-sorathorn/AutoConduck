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
) -> tuple[str | None, dict]:
    """Resolve a pseudo/custom model to a concrete target + extra kwargs.

    Returns ``(target, extra_kwargs)`` suitable for passing to litellm.acompletion.
    """
    cfg = _get_config()
    target = body_model

    if body_model in _PSEUDO_SET:
        path, model = await _do_router_dispatch(messages, body_model, cfg)
        record_decision(path, model)
        if path == "SLOW" and model is None:
            answer = await _do_slow_route(messages, body_model)
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
    from .dispatcher import route
    try:
        decision = await asyncio.to_thread(route, messages, [], pseudo_model=body_model, config=cfg)
        return getattr(decision, "path", "FAST").upper(), getattr(decision, "model", None)
    except Exception:
        return "FAST", None


async def _do_slow_route(messages, body_model):
    from .orchestrator import run
    try:
        return await run(messages, [], pseudo_model=body_model)
    except Exception:
        return None


def _pick_fast_model(body_model, cfg):
    try:
        from .pricing import select_closest, pool_ids
        return select_closest(pool_ids(cfg), .15, cfg, pseudo_model=body_model)
    except Exception:
        from .config import resolve_orchestrator_model
        return resolve_orchestrator_model(cfg)


def _litellm_extra(target: str, cfg) -> dict:
    from .messages_api import litellm_params_for
    return litellm_params_for(target, cfg)
