from dataclasses import dataclass, replace
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
    candidates_considered: int = 0
    candidates_excluded_by: dict[str, int] | None = None
    binding_constraint: str | None = None
    min_capability_score_applied: float = 0.0
    spend_cap_engaged: bool = False
    fallback_reason: str | None = None
    capability_fit_applied: float | None = None
    binding_capability_dim: str | None = None


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
    selection_info = None
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
        selection_info = pricing.select_for_sla_detailed(sla, config=config, pseudo_model=pseudo_model)
        model = selection_info.model or resolve_orchestrator_model(config)
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
        selection_info = _select_planned(plan.suggested_sla, plan, config, pseudo_model, "escalate")
        model = selection_info.model or resolve_orchestrator_model(config)

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
            selection_info = _select_planned(plan.suggested_sla, plan, config, pseudo_model, "refactor" if plan.task_type in ("refactor", "full_workflow", "multi_edit") else None)
            model = selection_info.model or resolve_orchestrator_model(config)
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
        candidates_considered=selection_info.candidates_considered if selection_info else 0,
        candidates_excluded_by=selection_info.candidates_excluded_by if selection_info else None,
        binding_constraint=selection_info.binding_constraint if selection_info else None,
        min_capability_score_applied=selection_info.min_capability_score_applied if selection_info else 0.0,
        spend_cap_engaged=selection_info.spend_cap_engaged if selection_info else False,
        fallback_reason=selection_info.fallback_reason if selection_info else None,
        capability_fit_applied=selection_info.capability_fit_applied if selection_info else None,
        binding_capability_dim=selection_info.binding_capability_dim if selection_info else None,
    )


def _select_planned(sla: CapabilitySLA, plan: Any, config: Any, pseudo_model: str, tier: str | None):
    try:
        confidence = max(0.0, min(1.0, float(plan.confidence)))
        selection = getattr(config, "selection", None)
        base = sla.min_capability_score
        if base > 0:
            floor = min(base + float(getattr(selection, "confidence_floor_k", 0.15)) * (1 - confidence), float(getattr(selection, "confidence_floor_max", 0.6)))
        else:
            floor = base
        ceiling = None
        if tier:
            ceilings = getattr(selection, "path_price_cap_usd_per_mtok", {})
            ceiling = ceilings.get(tier)
        modified = replace(
            sla,
            min_capability_score=floor,
            max_price_usd_per_mtok=ceiling,
            task_type=getattr(plan, "task_type", None),
        )
        return pricing.select_for_sla_detailed(modified, config=config, pseudo_model=pseudo_model)
    except Exception:
        return pricing.select_for_sla_detailed(sla, config=config, pseudo_model=pseudo_model)


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
