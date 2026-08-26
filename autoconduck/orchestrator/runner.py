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
                subtasks_cnt = len(getattr(plan, "subtasks", []))
                confidence = getattr(plan, "confidence", None)
                confidence_text = f"{float(confidence) * 100:.0f}%" if confidence is not None and float(confidence) <= 1 else f"{confidence}%" if confidence is not None else "n/a"
                lines = [
                    "SLM PLAN",
                    f"route: {getattr(plan, 'route', 'dynamic_dag')}",
                    f"task_type: {getattr(plan, 'task_type', 'unknown')}",
                    f"confidence: {confidence_text}",
                    f"subtasks: {subtasks_cnt}",
                ]
                for task in plan.subtasks:
                    goal = " ".join(str(task.goal).split())
                    if len(goal) > 70:
                        goal = goal[:67].rstrip() + "..."
                    lines.append(f"- {task.role} [{task.id}]: {goal}")
                on_progress(
                    {"node": "slm_plan", "state": "completed", "step_detail": "\n".join(lines)}
                )
            except Exception:
                pass

        # Rebuild the graph every turn: the plan is re-derived per turn, so no caching/checkpointer.
        runner = build_dynamic_graph(plan, on_progress=on_progress)
        if on_progress is not None and plan.subtasks:
            try:
                on_progress({"node": "subagent_pool", "state": "running", "step_detail": f"Deploying {len(plan.subtasks)} subagents in parallel"})
            except Exception:
                pass
        initial_state = DynamicState(
            session_id=kwargs.get("session_id", "session_orchestrator"),
            thread_id=kwargs.get("thread_id", "thread_orchestrator"),
            plan=plan,
            messages=messages,
            client_type=kwargs.get("client_type"),
            user_agent=kwargs.get("user_agent", ""),
            is_nested=kwargs.get("is_nested", False),
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
