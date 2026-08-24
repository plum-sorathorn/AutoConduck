from dataclasses import dataclass
from typing import Literal, Any
import os
from urllib.parse import urlsplit
from . import pricing
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

    # Step 1: Turn Guard 0ms Classification
    guard = TurnGuard()
    guard_res = guard.classify_turn(messages)

    if guard_res.target_action == TurnAction.DIRECT_ACTIVE_TIER:
        path = "fast"
        confidence_band = "fast"
        confidence = 1.0
        complexity = 0.2
        reason = f"tool_loop_bypass: {guard_res.last_tool_name or 'tool'}"
    elif guard_res.target_action == TurnAction.ESCALATE_SLM:
        path = "slow"
        confidence_band = "slow"
        confidence = 0.95
        complexity = 0.85
        reason = f"stagnation_escalation: {guard_res.stagnation_reason}"
    else:
        # User turn: estimate complexity
        user_msgs = _user_messages(messages)
        last = user_msgs[-1] if user_msgs else ""
        content = (
            last.get("content", "")
            if isinstance(last, dict)
            else getattr(last, "content", str(last))
        )
        content_len = len(str(content).split())
        complexity = min(1.0, max(0.1, content_len / 150.0))

        slow_thresh = float(getattr(getattr(config, "selection", config), "slow_threshold", 0.75))
        if complexity >= slow_thresh:
            path = "slow"
            confidence_band = "slow"
            confidence = 0.9
            reason = "high_complexity_workflow"
        else:
            path = "fast"
            confidence_band = "fast"
            confidence = 0.9
            reason = "fast_direct_dispatch"

    from ..config import resolve_orchestrator_model
    selection = getattr(config, "selection", config)
    max_fast_cost = float(getattr(selection, "fast_path_max_scaled_cost", 0.50))

    target_comp = 0.85 if path == "slow" else min(max_fast_cost, complexity)
    model = pricing.select_closest(
        pricing.pool_ids(config),
        target_comp,
        config,
        pseudo_model=pseudo_model,
        max_scaled_cost=max_fast_cost if path == "fast" else None,
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
    )

