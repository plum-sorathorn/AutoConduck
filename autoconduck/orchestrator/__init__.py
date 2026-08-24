"""Dynamic SLM Orchestration and LangGraph Factory."""

from .dynamic_factory import DynamicState, build_dynamic_graph
from .session_guard import SessionGuard, SessionGuardResult
from .roles import RoleConfig, ROLES

async def run(messages, history=None, pseudo_model="autoconduck", **kwargs):
    """Run dynamic DAG orchestration workflow."""
    from autoconduck.resolver import _do_slow_route
    import inspect

    try:
        sig = inspect.signature(_do_slow_route)
        if "plan" in sig.parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        ):
            return await _do_slow_route(
                messages,
                pseudo_model,
                on_progress=kwargs.get("on_progress"),
                plan=kwargs.get("plan"),
            )
        return await _do_slow_route(
            messages,
            pseudo_model,
            on_progress=kwargs.get("on_progress"),
        )
    except TypeError:
        return await _do_slow_route(
            messages,
            pseudo_model,
            on_progress=kwargs.get("on_progress"),
        )


__all__ = [
    "run",
    "DynamicState",
    "build_dynamic_graph",
    "SessionGuard",
    "SessionGuardResult",
    "RoleConfig",
    "ROLES",
]
