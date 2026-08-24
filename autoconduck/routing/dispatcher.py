from dataclasses import dataclass
from typing import Literal, Any
import os
from urllib.parse import urlsplit
from . import pricing
from autoconduck.routing.slm_planner import SLMPlanner, ExecutionPlan, ModelTier
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
        tier = ModelTier.BALANCED.value
        reason = f"tool_loop_bypass: {guard_res.last_tool_name or 'tool'}"
        model = pricing.select_for_tier(
            ModelTier.BALANCED, config=config, pseudo_model=pseudo_model
        ) or pricing.select_closest(
            pricing.pool_ids(config), 0.25, config, pseudo_model=pseudo_model
        ) or resolve_orchestrator_model(config)

    elif guard_res.target_action == TurnAction.ESCALATE_SLM:
        path = "slow"
        route_name = "dynamic_dag"
        confidence_band = "slow"
        confidence = 0.95
        complexity = 0.85
        tier = ModelTier.FRONTIER_REASONING.value
        reason = f"stagnation_escalation: {guard_res.stagnation_reason}"
        planner = SLMPlanner()
        plan = planner._create_fallback_plan(
            messages, reason=f"stagnation_escalation: {guard_res.stagnation_reason}"
        )
        plan.route = "dynamic_dag"
        plan.suggested_tier = ModelTier.FRONTIER_REASONING
        model = pricing.select_for_tier(
            ModelTier.FRONTIER_REASONING, config=config, pseudo_model=pseudo_model
        ) or pricing.select_closest(
            pricing.pool_ids(config), 0.85, config, pseudo_model=pseudo_model
        ) or resolve_orchestrator_model(config)

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
            tier = (
                plan.synthesizer_tier.value
                if hasattr(plan.synthesizer_tier, "value")
                else str(plan.synthesizer_tier)
            )
            reason = plan.rationale or f"dynamic_dag_{plan.task_type}"
            model = pricing.select_for_tier(
                plan.synthesizer_tier, config=config, pseudo_model=pseudo_model
            ) or pricing.select_closest(
                pricing.pool_ids(config), 0.85, config, pseudo_model=pseudo_model
            ) or resolve_orchestrator_model(config)
        else:
            path = "fast"
            route_name = "fast_direct"
            confidence_band = "fast"
            confidence = plan.confidence
            complexity = (
                0.1
                if plan.suggested_tier == ModelTier.CHEAP_FAST
                else (0.2 if plan.suggested_tier == ModelTier.BALANCED else 0.5)
            )
            tier = (
                plan.suggested_tier.value
                if hasattr(plan.suggested_tier, "value")
                else str(plan.suggested_tier)
            )
            reason = plan.rationale or f"fast_direct_{plan.task_type}"
            selection = getattr(config, "selection", config)
            max_fast_cost = float(getattr(selection, "fast_path_max_scaled_cost", 0.50))
            model = pricing.select_for_tier(
                plan.suggested_tier, config=config, pseudo_model=pseudo_model
            ) or pricing.select_closest(
                pricing.pool_ids(config),
                min(max_fast_cost, complexity),
                config,
                pseudo_model=pseudo_model,
                max_scaled_cost=max_fast_cost,
            ) or resolve_orchestrator_model(config)

    if model:
        try:
            from ..stats import record_selection
            record_selection(
                complexity,
                pricing.target_scaled_cost(complexity, pseudo_model, config),
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

