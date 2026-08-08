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


def _response_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        return str(response["choices"][0]["message"]["content"])
    return str(response.choices[0].message.content)


def _call(client: Any, model: str, messages: list[dict[str, str]]) -> str:
    from autoconduck.config import orchestrator_litellm_params, get_config, qualify_model
    params = orchestrator_litellm_params(get_config())
    params["model"] = qualify_model(model)
    if client is not None and hasattr(client, "completion"):
        return _response_text(client.completion(messages=messages, **params))
    if client is not None and hasattr(client, "chat"):
        return _response_text(client.chat.completions.create(messages=messages, **params))
    import litellm
    return _response_text(litellm.completion(messages=messages, **params))


def _executor_model(pseudo_model: str, cfg=None) -> str:
    try:
        from autoconduck import pricing
        selected = pricing.select(pseudo_model)
        if isinstance(selected, str):
            return selected
        if selected:
            return str(getattr(selected, "model", selected))
    except Exception:
        pass
    from autoconduck.config import resolve_orchestrator_model
    return resolve_orchestrator_model(cfg)


def run(messages: list, history, pseudo_model: str = "autoconduck", client=None) -> str | None:
    if not _LANGGRAPH_AVAILABLE:
        return None
    try:
        from autoconduck.config import get_config
        cfg = get_config()
        def planner_node(state: State) -> dict:
            plan = build_task_plan(state.messages, client=client, cfg=cfg)
            attempt = state.attempt + 1
            return {"plan": plan, "attempt": attempt, "fallback": plan is None and attempt >= 2}

        def subagent_pool_node(state: State) -> dict:
            if state.plan is None:
                return {"fallback": True}
            completed: dict[str, str] = {}
            pending = list(state.plan.subtasks)
            # This is the Send fan-out boundary; dependency waves preserve context.
            while pending:
                ready = [task for task in pending if all(dep in completed for dep in task.depends_on)]
                if not ready:
                    return {"fallback": True}
                # Construct Send envelopes so the pool remains compatible with the
                # native map/reduce API while keeping injectable clients deterministic.
                for task in ready:
                    Send("subagent_pool", {"task_id": task.id})
                    upstream = "\n".join(completed[dep] for dep in task.depends_on)
                    completed[task.id] = run_subagent(task, upstream, client=client, cfg=cfg)
                    pending.remove(task)
            return {"subagent_outputs": completed}

        def compactor_node(state: State) -> dict:
            ordered = [state.subagent_outputs[task.id] for task in sorted(
                state.plan.subtasks if state.plan else [],
                key=lambda task: (-len(task.depends_on), task.id),
            ) if task.id in state.subagent_outputs]
            return {"compacted": compact(ordered)}

        def executor_node(state: State) -> dict:
            prompt = f"Original request:\n{state.messages}\n\nAnalyst summary:\n{state.compacted}"
            return {"result": _call(client, _executor_model(state.pseudo_model, cfg), [{"role": "user", "content": prompt}])}

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
        graph.add_conditional_edges("planner", after_plan, {"pool": "subagent_pool", "retry": "planner", "end": END})
        graph.add_conditional_edges("subagent_pool", after_pool, {"compact": "compactor", "end": END})
        graph.add_edge("compactor", "executor")
        graph.add_edge("executor", END)
        final = graph.compile().invoke(State(messages=messages, history=history, pseudo_model=pseudo_model))
        state = State.model_validate(final)
        return None if state.fallback else state.result
    except Exception:
        return None
