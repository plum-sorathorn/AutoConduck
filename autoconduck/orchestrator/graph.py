"""Fault-tolerant LangGraph orchestration for the slow path."""

from typing import Any
import logging
import asyncio
from pathlib import Path

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
from .helpers import _response_text, _executor_model
from .executor_loop import run_executor_tool_loop, strip_tool_call_tags


from .recon import ReconTarget, build_recon_plan
from autoconduck.progress import ProgressEvent
from .roles import assign_subagent_role, role_card


PHASE_BANDS = {
    "recon": [0.10, 0.45],
    "planner": [0.55, 0.85],
    "subagent": [0.10, 0.55],
    "executor": [0.35, 0.70],
}


def _is_subagent_error(value: str) -> bool:
    return isinstance(value, str) and value.startswith("__SUBAGENT_ERROR__")


class State(BaseModel):
    messages: list = Field(default_factory=list)
    history: Any = None
    pseudo_model: str = "autoconduck"
    recon: Any = None
    plan: TaskPlan | None = None
    subagent_outputs: dict[str, str] = Field(default_factory=dict)
    subagent_error_count: int = 0
    compacted: str = ""
    result: str | None = None
    fallback: bool = False
    attempt: int = 0
    task_value: float = 0.5


async def _call(client: Any, model: str, messages: list[dict[str, str]]) -> str:
    import asyncio
    from typing import Any as TypingAny
    from autoconduck.config import get_config
    from autoconduck.messages_api import normalize_messages_for_llm, litellm_params_for

    messages = normalize_messages_for_llm(messages)
    cfg = get_config()
    params: TypingAny = litellm_params_for(model, cfg)
    params["_path"] = "orchestrator-executor"
    params["_pseudo"] = "autoconduck"
    params["drop_params"] = True
    logger = logging.getLogger("autoconduck.orchestrator")
    prompt_log = logger.info if getattr(getattr(cfg, "selection", None), "dump_prompts", True) else logger.debug
    prompt_log(
        "EXECUTOR PROMPT (%s):\n%s",
        model,
        "\n".join(m.get("content", "") if isinstance(m, dict) else str(m) for m in messages),
    )
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


async def run(
    messages: list,
    history,
    pseudo_model: str = "autoconduck",
    client=None,
    task_value: float = 0.5,
    request=None,
    on_progress: Any = None,
) -> str | None:
    if not _LANGGRAPH_AVAILABLE:
        return None
    try:
        if request is not None and hasattr(request, "is_disconnected") and await request.is_disconnected():
            return None
        import time
        import logging
        from autoconduck.config import get_config
        from autoconduck.messages_api import normalize_messages_for_llm
        from autoconduck import stats
        log = logging.getLogger("autoconduck.orchestrator")

        def _emit(**kw: Any) -> None:
            is_subagent_row = kw.get("kind") == "subagent"
            if not is_subagent_row:
                try:
                    stats.update_active_routing(**kw)
                except Exception:
                    pass
            if on_progress is not None:
                try:
                    name = kw.get("node", "")
                    active = kw.get("active", True)
                    state = "running" if active else "done"
                    detail = kw.get("step_detail", kw.get("detail", ""))
                    if kw.get("kind") == "subagent":
                        # Per-subagent row: name carries the role, index/total
                        # identify this subagent within the pool.
                        kind = "subagent"
                        name = kw.get("role", name)
                        index = kw.get("index", 0)
                        total = kw.get("total", 0)
                    elif name == "idle" and not active:
                        # Terminal footer row for the whole workflow.
                        kind = "footer"

                        index = 0
                        total = 0
                    elif name == "subagent_pool" or name == "analysts":
                        kind = "pool"
                        index = 0
                        total = kw.get("subtasks_total", 0)
                    else:
                        kind = "node"
                        index = 0
                        total = kw.get("subtasks_total", 0)
                    on_progress(
                        ProgressEvent(
                            kind=kind,
                            name=name,
                            state=state,
                            detail=detail,
                            index=index,
                            total=total,
                        )
                    )
                except Exception:
                    pass

        cfg = get_config()
        messages = normalize_messages_for_llm(messages)

        direct_threshold = float(
            getattr(getattr(cfg, "selection", None), "min_orchestrator_complexity", 0.62)
        )
        if task_value < direct_threshold:
            import logging
            logging.getLogger("autoconduck.orchestrator").debug(
                "Direct-executor short-circuit (task_value=%.2f < %.2f)", task_value, direct_threshold
            )
            user_text = "\n".join(
                str(m.get("content", "")) if isinstance(m, dict) else getattr(m, "content", str(m))
                for m in messages
                if not isinstance(m, dict) or m.get("role", "user") == "user"
            )
            try:
                exec_model = _executor_model(pseudo_model, cfg, task_value, "", 0)
                return await _call(client, exec_model, [{"role": "user", "content": user_text}])
            except Exception:
                return None

        async def analyst_pool_node(state: State) -> dict:
            if request is not None and hasattr(request, "is_disconnected") and await request.is_disconnected():
                return {"fallback": True}
            log.info("Starting analyst_pool node")
            _emit(active=True, path="SLOW", pseudo_model=state.pseudo_model,
                  task_value=state.task_value, node="analysts",
                  step_detail="Investigating codebase architecture...", start_time=time.time())

            from .planner import _extract_file_paths, OutputContract, SubTask
            from .skeletons import resolve_1hop_dependencies
            from .roles import resolve_analysts_for_task

            # 1. Grounding & 1-hop AST discovery
            explicit_files = _extract_file_paths(state.messages)
            candidate_files = resolve_1hop_dependencies(explicit_files, max_total_files=6)

            # Fallback to recon locator if 0 explicit files found
            if not candidate_files:
                target = build_recon_plan(state.messages, client=client, cfg=cfg, task_value=state.task_value)
                candidate_files = target.files[:4] if target and target.files else []

            # 2. Dynamic role assignment based on task_value and scope
            user_text = "\n".join(
                str(m.get("content", "")) if isinstance(m, dict) else str(m)
                for m in state.messages
                if not isinstance(m, dict) or m.get("role", "user") == "user"
            )
            analyst_specs = resolve_analysts_for_task(user_text, candidate_files, task_value=state.task_value)
            log.info("Deploying %d specialized analysts for task", len(analyst_specs))
            _emit(node="analysts", subtasks_total=len(analyst_specs),
                  step_detail=f"Running {len(analyst_specs)} specialized analyst(s)...")

            started = {f"analyst-{i}": time.time() for i in range(1, len(analyst_specs) + 1)}
            for index, (role, goal, scope) in enumerate(analyst_specs, 1):
                _emit(kind="subagent", active=True, node="analysts",
                      index=index, total=len(analyst_specs), role=role,
                      step_detail=f"{role} · {goal[:60]}")

            # 3. Parallel analyst fan-out
            async def run_one_analyst(index: int, role: str, goal: str, scope: list[str]) -> tuple[str, str]:
                task = SubTask(
                    id=f"analyst-{index}-{role}",
                    goal=goal,
                    scope=scope,
                    output_contract=OutputContract(
                        description="Key symbols, behavior, root cause, dependencies, and risks with file:line references; concise findings."
                    ),
                    constraints=["Do not modify files", "Cite exact symbols and lines", "Keep findings concise"],
                    role=role,
                )
                try:
                    res = await run_subagent(
                        task, "", client=client, cfg=cfg,
                        plan_breadth=len(analyst_specs), budget_hint=state.task_value,
                    )
                    return role, str(res)
                except Exception as exc:
                    return role, f"[WARNING: {role.upper()}_ANALYST_ERROR - {exc}]"

            results = await asyncio.gather(*(
                run_one_analyst(i, role, goal, scope)
                for i, (role, goal, scope) in enumerate(analyst_specs, 1)
            ))

            findings = []
            outputs = {}
            for index, (role, res) in enumerate(results, 1):
                elapsed = max(0.0, time.time() - started.get(f"analyst-{index}", time.time()))
                _emit(kind="subagent", active=False, node="analysts",
                      index=index, total=len(analyst_specs), role=role, elapsed_s=elapsed,
                      step_detail=f"{role} analysis complete")
                outputs[f"{role}_{index}"] = res
                if not _is_subagent_error(res):
                    findings.append(f"### [{role.upper()} FINDINGS]\n{str(res)[:1500]}")

            return {
                "subagent_outputs": outputs,
                "compacted": compact(findings) if findings else "",
            }

        async def planner_node(state: State) -> dict:
            if request is not None and hasattr(request, "is_disconnected") and await request.is_disconnected():
                return {"fallback": True}
            log.info("Starting master planner node")
            _emit(node="planner", step_detail="Master Planner synthesizing verified DAG task plan...")
            plan = build_task_plan(
                state.messages, client=client, cfg=cfg, task_value=state.task_value, ground_truth=state.compacted
            )
            if plan is not None:
                max_sub = int(getattr(getattr(cfg, "selection", None), "max_subtasks", 4))
                if len(plan.subtasks) > max_sub:
                    plan = plan.model_copy(update={"subtasks": plan.subtasks[:max_sub]})
                log.info("Planner built plan with %d subtasks", len(plan.subtasks))
            attempt = state.attempt + 1
            return {
                "plan": plan,
                "attempt": attempt,
                "fallback": plan is None and attempt >= 2,
            }

        from .handoff import format_execution_handoff

        def compactor_node(state: State) -> dict:
            _emit(node="compactor", step_detail="Formatting execution handoff for Pi...")
            handoff_result = format_execution_handoff(
                plan=state.plan,
                subagent_outputs=state.subagent_outputs,
                compacted=state.compacted,
            )
            return {"result": handoff_result}

        graph = StateGraph(State)
        graph.add_node("analysts", analyst_pool_node)
        graph.add_node("planner", planner_node)
        graph.add_node("compactor", compactor_node)

        graph.add_edge(START, "analysts")
        graph.add_edge("analysts", "planner")
        graph.add_edge("planner", "compactor")
        graph.add_edge("compactor", END)
        try:
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
        finally:
            try:
                _emit(active=False, node="idle", step_detail="Completed")
            except Exception:
                pass
    except Exception:
        return None
