"""Dynamic SLM Orchestration and LangGraph Factory."""

from .dynamic_factory import DynamicState, build_dynamic_graph
from .session_guard import SessionGuard, SessionGuardResult
from .roles import RoleConfig, ROLES
from .runner import run_dynamic_orchestration


async def run(messages, history=None, pseudo_model="autoconduck", **kwargs):
    """Run dynamic DAG orchestration workflow."""
    on_progress = kwargs.pop("on_progress", None)
    plan = kwargs.pop("plan", None)
    return await run_dynamic_orchestration(
        messages=messages,
        pseudo_model=pseudo_model,
        on_progress=on_progress,
        plan=plan,
        **kwargs,
    )


__all__ = [
    "run",
    "run_dynamic_orchestration",
    "DynamicState",
    "build_dynamic_graph",
    "SessionGuard",
    "SessionGuardResult",
    "RoleConfig",
    "ROLES",
]
