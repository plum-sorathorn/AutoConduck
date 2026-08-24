"""Routing dispatcher integration, active tool turn detection, and upstream model resolution."""

from __future__ import annotations

import logging
import time
from typing import Any

import autoconduck.config as config_module


def is_active_tool_session(messages: list[Any]) -> bool:
    """Return True if the conversation is an active agentic tool loop.

    In an active tool loop, the client agent (Pi, Claude Code, OpenCode, etc.)
    is managing its own tool execution loop. AutoConduck relays requests
    directly to the selected model rather than hijacking the turn with the
    multi-agent LangGraph orchestrator.
    """
    try:
        from autoconduck.server.turn_guard import TurnGuard

        res = TurnGuard().classify_turn(messages)
        if res.is_stagnant:
            return False
        return res.is_tool_loop
    except Exception:
        pass
    if not isinstance(messages, list) or not messages:
        return False
    for m in messages:
        if not isinstance(m, dict):
            continue
        if (
            m.get("role") in ("tool", "function", "toolResult")
            or "tool_calls" in m
            or "function_call" in m
        ):
            return True
        content = m.get("content")
        if isinstance(content, list) and any(
            isinstance(b, dict) and b.get("type") in ("tool_use", "tool_result")
            for b in content
        ):
            return True
    return False


async def call_litellm(
    model: str,
    body: Any,
    path: str | None = None,
    pseudo: str | None = None,
    messages: list[Any] | None = None,
    *,
    normalize_messages_for_llm: Any = None,
    sanitize_tools: Any = None,
    litellm_params_for: Any = None,
) -> dict[str, Any]:
    """Execute a single direct LiteLLM acompletion request."""
    from autoconduck.server.server_streaming import _litellm

    llm = _litellm()
    if llm is None:
        raise RuntimeError("litellm unavailable")
    kwargs = body.model_dump(exclude_none=True)
    if messages is not None and normalize_messages_for_llm is not None:
        kwargs["messages"] = normalize_messages_for_llm(messages)
    if kwargs.get("tools") and sanitize_tools is not None:
        kwargs["tools"] = sanitize_tools(kwargs["tools"])
    kwargs.update(model=model, drop_params=True)
    kwargs.pop("stream", None)
    if litellm_params_for is not None:
        kwargs.update(litellm_params_for(model, config_module.get_config()))
    kwargs["_path"] = path if path is not None else "unknown"
    kwargs["_pseudo"] = pseudo if pseudo is not None else "unknown"
    result = await llm.acompletion(**kwargs)
    return result.model_dump() if hasattr(result, "model_dump") else result


async def route_target(
    body_model: str,
    messages: list[Any],
    request: Any = None,
    on_progress: Any = None,
    client_type: str | None = None,
    decisions: list[dict[str, Any]] | None = None,
    *,
    PSEUDO_MODELS: set[str],
    litellm_params_for: Any,
    normalize_messages_for_llm: Any,
) -> tuple[str | None, dict[str, Any]]:
    """Determine routing path, model selection, and orchestration execution."""
    started = time.perf_counter()
    cfg = config_module.get_config()
    messages = normalize_messages_for_llm(messages)
    target, path = body_model, "direct"
    request_depth = 0
    if request is not None and hasattr(request, "headers"):
        try:
            request_depth = int(request.headers.get("x-autoconduck-depth", "0"))
        except (ValueError, TypeError):
            request_depth = 0
        if client_type is None:
            client_type = request.headers.get("x-agent-id", None)
        if client_type is None:
            ua = request.headers.get("user-agent", "").lower()
            if "pi" in ua or "pi-coding-agent" in ua:
                client_type = "pi"
            elif "opencode" in ua:
                client_type = "opencode"
            elif "claude" in ua:
                client_type = "claude"
            elif "aider" in ua:
                client_type = "aider"
            elif "cursor" in ua:
                client_type = "cursor"
    is_nested = request_depth >= 1
    decision = None
    if body_model in PSEUDO_MODELS:
        try:
            from autoconduck.routing.dispatcher import route

            history = decisions[-5:] if decisions else []
            decision = route(messages, history, pseudo_model=body_model, config=cfg)
            path = getattr(decision, "path", "FAST").upper()
            route_name = getattr(decision, "route", "fast_direct")
            tier = getattr(decision, "tier", "balanced")
            plan = getattr(decision, "plan", None)
            model = getattr(decision, "model", None)
        except Exception:
            decision, path, route_name, tier, plan, model = (
                None,
                "FAST",
                "fast_direct",
                "balanced",
                None,
                None,
            )
        task_complexity = float(getattr(decision, "complexity", 0.5))
        if path == "SLOW" and is_nested:
            path = "FAST"
            route_name = "fast_direct"
            model = model or None
            logging.getLogger("autoconduck").info(
                "Nested orchestrator call (depth=%d) downgraded to FAST",
                request_depth,
            )
        if on_progress is not None:
            try:
                on_progress(
                    {"kind": "route", "path": path, "route": route_name, "tier": tier}
                )
            except Exception:
                pass
        if decisions is not None:
            decisions.append(
                {
                    "path": path,
                    "route": route_name,
                    "tier": tier,
                    "model": model or body_model,
                    "time": time.time(),
                }
            )
        logging.getLogger("autoconduck").info(
            "route=%s tier=%s model=%s ms=%.1f",
            route_name,
            tier,
            model or body_model,
            (time.perf_counter() - started) * 1000,
        )

        in_tool_loop = is_active_tool_session(messages)
        if in_tool_loop and path == "SLOW":
            logging.getLogger("autoconduck").debug(
                "Active tool loop detected — delegating completion directly to selected model %s",
                model or body_model,
            )
        if (
            path == "SLOW"
            and not in_tool_loop
            and not (request is not None and await request.is_disconnected())
        ):
            try:
                from autoconduck.orchestrator import run

                result = await run(
                    messages,
                    [],
                    pseudo_model=body_model,
                    task_value=task_complexity,
                    request=request,
                    on_progress=on_progress,
                    client_type=client_type,
                    is_nested=is_nested,
                    plan=plan,
                )
                if result is not None:
                    tool_calls = getattr(result, "tool_calls", None) or (
                        result.get("tool_calls") if isinstance(result, dict) else None
                    )
                    content = (
                        str(result)
                        if not isinstance(result, dict)
                        else result.get("content", str(result))
                    )
                    ans: dict[str, Any] = {"content": content}
                    if tool_calls:
                        ans["tool_calls"] = tool_calls
                    return None, {
                        "__answer__": ans,
                        "_path": path,
                        "_pseudo": body_model,
                        "_route": route_name,
                        "_tier": tier,
                        "_plan": plan,
                        "_complexity": task_complexity,
                    }
            except Exception as exc:
                logging.getLogger("autoconduck").warning(
                    "Orchestrator execution failed: %s", exc
                )
        if not model:
            try:
                from autoconduck.config import resolve_orchestrator_model
                from autoconduck.routing.pricing import pool_ids, select_closest

                selected = select_closest(
                    pool_ids(cfg), 0.15, cfg, pseudo_model=body_model
                )
                model = selected or resolve_orchestrator_model(cfg)
                if not selected:
                    logging.getLogger("autoconduck").warning(
                        "Model pool is empty - no models configured; falling back to %s",
                        model,
                    )
            except Exception:
                from autoconduck.config import resolve_orchestrator_model

                model = resolve_orchestrator_model(cfg)
        if not model:
            logging.getLogger("autoconduck").warning(
                "No model available for request"
            )
        target = model
    extra = litellm_params_for(target, cfg)
    extra.update(
        _path=path if body_model in PSEUDO_MODELS else "direct",
        _pseudo=body_model,
    )
    if body_model in PSEUDO_MODELS:
        extra.update(
            _complexity=float(
                getattr(decision, "complexity", 0.5) if decision else 0.5
            ),
            _route=getattr(decision, "route", "fast_direct") if decision else "direct",
            _tier=getattr(decision, "tier", None) if decision else None,
            _plan=getattr(decision, "plan", None) if decision else None,
        )
    return target, extra
