"""Fault-tolerant LangGraph orchestration for the slow path."""

from typing import Any

from pydantic import BaseModel, Field

try:  # LangGraph is an optional pillar for minimal installs.
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Send

    _LANGGRAPH_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on environment
    END = START = StateGraph = Send = None
    _LANGGRAPH_AVAILABLE = False

from .compactor import compact
from .planner import TaskPlan, build_task_plan
from .subagents import run_subagent


class State(BaseModel):
    messages: list = Field(default_factory=list)
    history: Any = None
    pseudo_model: str = "autoconduck"
    plan: TaskPlan | None = None
    subagent_outputs: dict[str, str] = Field(default_factory=dict)
    compacted: str = ""
    result: str | None = None
    fallback: bool = False
    attempt: int = 0
    task_value: float = 0.5


def _response_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        return str(response["choices"][0]["message"]["content"])
    return str(response.choices[0].message.content)


async def _call(client: Any, model: str, messages: list[dict[str, str]]) -> str:
    import asyncio
    from typing import Any as TypingAny
    from autoconduck.config import (
        orchestrator_litellm_params,
        get_config,
        qualify_model,
    )

    params: TypingAny = orchestrator_litellm_params(get_config())
    params["model"] = qualify_model(model)
    params["_path"] = "orchestrator-executor"
    params["_pseudo"] = "autoconduck"
    if client is not None and hasattr(client, "completion"):
        return _response_text(
            await asyncio.to_thread(client.completion, messages=messages, **params)
        )
    if client is not None and hasattr(client, "chat"):
        return _response_text(
            await asyncio.to_thread(
                client.chat.completions.create, messages=messages, **params
            )
        )
    import litellm

    return _response_text(await litellm.acompletion(messages=messages, **params))


def _executor_model(
    pseudo_model: str, cfg=None, task_value=0.5, compactor_summary="", subtask_count=0
) -> str:
    try:
        from autoconduck import pricing
        from autoconduck.config import get_config
        from autoconduck.evaluator import complexity_of

        cfg = cfg or get_config()
        lo, hi = cfg.selection.phase_bands["executor"]
        raw = (
            0.5 * task_value
            + 0.3 * complexity_of(compactor_summary, cfg)
            + 0.2 * min(1, subtask_count / 6)
        )
        return pricing.select_closest(
            pricing.pool_ids(cfg),
            lo + (hi - lo) * max(0, min(1, raw)),
            cfg,
            pseudo_model=pseudo_model,
            band=(lo, hi),
        )
    except Exception:
        pass
    from autoconduck.config import select_model_by_tier

    return select_model_by_tier("expensive", cfg)


async def run(
    messages: list,
    history,
    pseudo_model: str = "autoconduck",
    client=None,
    task_value: float = 0.5,
    request=None,
) -> str | None:
    if not _LANGGRAPH_AVAILABLE:
        return None
    try:
        if request is not None and hasattr(request, "is_disconnected") and await request.is_disconnected():
            return None
        from autoconduck.config import get_config

        cfg = get_config()

        async def planner_node(state: State) -> dict:
            if request is not None and hasattr(request, "is_disconnected") and await request.is_disconnected():
                return {"fallback": True}
            plan = build_task_plan(
                state.messages, client=client, cfg=cfg, task_value=state.task_value
            )
            attempt = state.attempt + 1
            return {
                "plan": plan,
                "attempt": attempt,
                "fallback": plan is None and attempt >= 2,
            }

        async def subagent_pool_node(state: State) -> dict:
            if state.plan is None or (request is not None and hasattr(request, "is_disconnected") and await request.is_disconnected()):
                return {"fallback": True}
            completed: dict[str, str] = {}
            pending = list(state.plan.subtasks)
            # This is the Send fan-out boundary; dependency waves preserve context.
            while pending:
                if request is not None and hasattr(request, "is_disconnected") and await request.is_disconnected():
                    return {"fallback": True}
                ready = [
                    task
                    for task in pending
                    if all(dep in completed for dep in task.depends_on)
                ]
                if not ready:
                    return {"fallback": True}
                # Construct Send envelopes so the pool remains compatible with the
                # native map/reduce API while keeping injectable clients deterministic.
                import asyncio
                import inspect

                calls = [
                    run_subagent(
                        task,
                        "\n".join(completed[dep] for dep in task.depends_on),
                        client=client,
                        cfg=cfg,
                        plan_breadth=len(state.plan.subtasks),
                        budget_hint=state.plan.budget_hint
                        if state.plan.budget_hint is not None
                        else state.task_value,
                    )
                    for task in ready
                ]
                results = await asyncio.gather(
                    *(
                        call
                        if inspect.isawaitable(call)
                        else asyncio.sleep(0, result=call)
                        for call in calls
                    )
                )
                for task, result in sorted(
                    zip(ready, results), key=lambda pair: pair[0].id
                ):
                    Send("subagent_pool", {"task_id": task.id})
                    completed[task.id] = result
                    pending.remove(task)
            return {"subagent_outputs": completed}

        def compactor_node(state: State) -> dict:
            ordered = [
                state.subagent_outputs[task.id]
                for task in sorted(
                    state.plan.subtasks if state.plan else [],
                    key=lambda task: (-len(task.depends_on), task.id),
                )
                if task.id in state.subagent_outputs
            ]
            return {"compacted": compact(ordered)}

        async def executor_node(state: State) -> dict:
            if request is not None and hasattr(request, "is_disconnected") and await request.is_disconnected():
                return {"fallback": True}
            user_text = "\n".join(
                str(m.get("content", ""))
                if isinstance(m, dict)
                else getattr(m, "content", str(m))
                for m in state.messages
                if not isinstance(m, dict) or m.get("role", "user") == "user"
            )
            prompt = f"Original request:\n{user_text}\n\nAnalyst summary:\n{state.compacted}"
            task_value = getattr(state, "task_value", 0.5)
            return {
                "result": await _call(
                    client,
                    _executor_model(
                        state.pseudo_model,
                        cfg,
                        task_value,
                        state.compacted,
                        len(state.subagent_outputs),
                    ),
                    [{"role": "user", "content": prompt}],
                )
            }

        def after_plan(state: State):
            if state.plan is not None:
                return "pool"
            return "end" if state.attempt >= 2 else "retry"

        def after_pool(state: State):
            return "end" if state.fallback else "compact"

        graph = StateGraph(State)
        graph.add_node("planner", planner_node)
        graph.add_node("subagent_pool", subagent_pool_node)
        graph.add_node("compactor", compactor_node)
        graph.add_node("executor", executor_node)
        graph.add_edge(START, "planner")
        graph.add_conditional_edges(
            "planner",
            after_plan,
            {"pool": "subagent_pool", "retry": "planner", "end": END},
        )
        graph.add_conditional_edges(
            "subagent_pool", after_pool, {"compact": "compactor", "end": END}
        )
        graph.add_edge("compactor", "executor")
        graph.add_edge("executor", END)
        final = await graph.compile().ainvoke(
            State(
                messages=messages,
                history=history,
                pseudo_model=pseudo_model,
                task_value=task_value,
            )
        )
        state = State.model_validate(final)
        return None if state.fallback else state.result
    except Exception:
        return None
