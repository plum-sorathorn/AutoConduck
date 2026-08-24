"""Async dynamic DAG orchestration runner."""

from __future__ import annotations

import logging
from typing import Any

from autoconduck.orchestrator.dynamic_factory import DynamicState, build_dynamic_graph
from autoconduck.routing.slm_planner import ExecutionPlan, SLMPlanner

logger = logging.getLogger(__name__)


async def run_dynamic_orchestration(
    messages: list[dict[str, Any]],
    pseudo_model: str = "autoconduck",
    on_progress: Any = None,
    plan: ExecutionPlan | None = None,
    **kwargs: Any,
) -> dict[str, Any] | None:
    """Execute dynamic LangGraph orchestration pipeline for a user turn."""
    try:
        if plan is None:
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

        runner = build_dynamic_graph(plan)
        initial_state = DynamicState(
            session_id=kwargs.get("session_id", "session_orchestrator"),
            thread_id=kwargs.get("thread_id", "thread_orchestrator"),
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
    except Exception as exc:
        logger.warning("Dynamic orchestration execution error: %s", exc)
        return None
