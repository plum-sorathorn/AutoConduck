from dataclasses import dataclass
from typing import Literal, Any
import os
from urllib.parse import urlsplit
from . import pricing
from autoconduck.routing.model_pool import CapabilitySLA
from autoconduck.routing.slm_planner import SLMPlanner, ExecutionPlan, SubTaskSpec
from autoconduck.server.turn_guard import TurnGuard, TurnAction, TurnClassificationResult


def _user_messages(messages: list) -> list:
    return [
        message
        for message in messages
        if not isinstance(message, dict) or message.get("role", "user") == "user"
    ]


@dataclass(frozen=True)
class RoutingDecision:
    path: Literal["fast", "slow"]
    confidence_band: Literal["fast", "slow", "ambiguous"]
    confidence: float
    complexity: float
    reason: str
    model: str | None = None
    route: str = "fast_direct"
    plan: Any = None
    tier: str | None = None


def route(
    messages: list,
    history: Any = None,
    pseudo_model: str = "autoconduck",
    tiebreaker: Any = None,
    config: Any = None,
) -> RoutingDecision:
    if config is None:
        from ..config import get_config
        config = get_config()

    from ..config import resolve_orchestrator_model

    # Step 1: Turn Guard 0ms Classification (<2ms overhead)
    guard = TurnGuard()
    guard_res = guard.classify_turn(messages)

    plan = None
    if guard_res.target_action == TurnAction.DIRECT_ACTIVE_TIER:
        path = "fast"
        route_name = "fast_direct"
        confidence_band = "fast"
        confidence = 1.0
        complexity = 0.2
        last_tool = (guard_res.last_tool_name or "").lower()
        is_routine_tool = any(
            t in last_tool
            for t in ["read", "glob", "list", "grep", "bash", "status", "diff", "command", "file", "view", "tool"]
        )
        reason = f"tool_loop_bypass: {guard_res.last_tool_name or 'tool'}"
        sla = CapabilitySLA(min_context=16000, requires_tools=True, max_cost=1.0 if is_routine_tool else 1.5)
        model = pricing.select_for_sla(sla, config=config, pseudo_model=pseudo_model) or resolve_orchestrator_model(config)
        tier = "capability_sla"

    elif guard_res.target_action == TurnAction.ESCALATE_SLM:
        path = "slow"
        route_name = "dynamic_dag"
        confidence_band = "slow"
        confidence = 0.95
        complexity = 0.85
        tier = "capability_sla"
        reason = f"stagnation_escalation: {guard_res.stagnation_reason}"
        planner = SLMPlanner()
        plan = planner.create_escalation_plan(
            messages, reason=f"stagnation_escalation: {guard_res.stagnation_reason}"
        )
        model = pricing.select_for_sla(plan.suggested_sla, config=config, pseudo_model=pseudo_model) or resolve_orchestrator_model(config)

    else:
        # Step 2: Embedded SLM Task Architect (<100ms circuit breaker)
        planner = SLMPlanner()
        plan = planner.plan_sync(messages, config)

        if plan.route == "dynamic_dag":
            path = "slow"
            route_name = "dynamic_dag"
            confidence_band = "slow"
            confidence = plan.confidence
            complexity = 0.85 if plan.task_type in ("refactor", "full_workflow") else 0.75
            tier = "capability_sla"
            reason = plan.rationale or f"dynamic_dag_{plan.task_type}"
            model = pricing.select_for_sla(
                plan.synthesizer_sla, config=config, pseudo_model=pseudo_model
            ) or resolve_orchestrator_model(config)
        else:
            path = "fast"
            route_name = "fast_direct"
            confidence_band = "fast"
            confidence = plan.confidence
            complexity = 0.2
            tier = "capability_sla"
            reason = plan.rationale or f"fast_direct_{plan.task_type}"
            model = pricing.select_for_sla(
                plan.suggested_sla, config=config, pseudo_model=pseudo_model
            ) or resolve_orchestrator_model(config)

    if model:
        try:
            from ..stats import record_selection
            record_selection(
                complexity,
                0.5,
                model,
                config,
            )
        except Exception:
            pass

    return RoutingDecision(
        path=path,
        confidence_band=confidence_band,
        confidence=confidence,
        complexity=complexity,
        reason=reason,
        model=model,
        route=route_name,
        plan=plan,
        tier=tier,
    )


def pick_fast_model(body_model: str = "autoconduck", cfg: Any = None) -> str:
    """Pick an economical fast model within the capability SLA."""
    if cfg is None:
        from ..config import get_config
        cfg = get_config()
    try:
        from ..config import resolve_orchestrator_model
        return (
            pricing.select_for_sla(
                CapabilitySLA(min_context=16000, requires_tools=True, max_cost=1.0),
                config=cfg,
                pseudo_model=body_model,
            )
            or resolve_orchestrator_model(cfg)
        )
    except Exception:
        from ..config import resolve_orchestrator_model
        return resolve_orchestrator_model(cfg)
