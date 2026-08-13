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
from .helpers import _response_text, _executor_model


from .recon import ReconTarget, build_recon_plan


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
    params: TypingAny = litellm_params_for(model, get_config())
    params["_path"] = "orchestrator-executor"
    params["_pseudo"] = "autoconduck"
    params["drop_params"] = True
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


class FileClaimRegistry:
    """Prevent two executor subagents from drafting changes for the same file.

    Each file path can only be claimed by one group. Claims are held for the
    lifetime of the registry (one executor invocation). The registry is used
    before concurrent drafting begins so conflicts are detected eagerly and
    merged into the first-claiming group.
    """

    def __init__(self) -> None:
        self._owners: dict[str, str] = {}

    def claim(self, group_id: str, scope: list[str]) -> bool:
        """Attempt to claim scope for group_id.

        Returns True if every file is unclaimed or already owned by group_id.
        Returns False (and claims nothing) if any file is owned by a different group.
        """
        conflicts = [f for f in scope if self._owners.get(f, group_id) != group_id]
        if conflicts:
            return False
        for f in scope:
            self._owners[f] = group_id
        return True

    def merge_into(self, owner_id: str, scope: list[str]) -> None:
        """Silently transfer unclaimed files in scope to owner_id."""
        for f in scope:
            self._owners.setdefault(f, owner_id)


async def _run_executor_subagents(
    *,
    plan: TaskPlan,
    subagent_outputs: dict,
    compacted: str,
    user_text: str,
    client: Any,
    cfg: Any,
    pseudo_model: str,
    task_value: float,
) -> str:
    """Fan the executor work into semantically-scoped write-draft subagents."""
    import asyncio
    import logging

    log = logging.getLogger("autoconduck.orchestrator")

    registry = FileClaimRegistry()
    groups: dict[str, dict] = {}

    for task in plan.subtasks:
        gid = task.id
        scope = task.scope or []
        if registry.claim(gid, scope):
            groups[gid] = {"scope": list(scope), "subtask_ids": [task.id]}
        else:
            owner_id = None
            for f in scope:
                owner_id = registry._owners.get(f)
                if owner_id and owner_id in groups:
                    break
            if owner_id and owner_id in groups:
                for f in scope:
                    if f not in groups[owner_id]["scope"]:
                        groups[owner_id]["scope"].append(f)
                groups[owner_id]["subtask_ids"].append(task.id)
                registry.merge_into(owner_id, scope)

    if not groups:
        return await _call(
            client,
            _executor_model(pseudo_model, cfg, task_value, compacted, 0),
            [{"role": "user", "content": f"Original request:\n{user_text}\n\nAnalyst summary:\n{compacted}"}],
        )

    async def _draft_for_group(group_id: str, group: dict) -> tuple:
        scope_files = ", ".join(group["scope"]) if group["scope"] else "general"
        outputs = []
        for sid in group["subtask_ids"]:
            raw_out = str(subagent_outputs.get(sid, ""))
            if len(raw_out) > 1200:
                raw_out = raw_out[:1200] + "\n...[truncated]"
            outputs.append(f"[{sid}]\n{raw_out}")
        relevant_outputs = "\n\n".join(outputs)
        from .subagents import SubTask, OutputContract, run_subagent as _rs
        draft_task = SubTask(
            id=group_id,
            goal=f"Draft changes for: {scope_files}",
            scope=group["scope"],
            output_contract=OutputContract(
                description="Proposed changes per file with precise line-level details"
            ),
            constraints=["Do not modify files outside the listed scope"],
            role="write",
        )
        log.debug("EXECUTOR SUBAGENT DRAFT [%s] scope=%s", group_id, scope_files)
        result = await _rs(
            draft_task,
            relevant_outputs,
            client=client,
            cfg=cfg,
            plan_breadth=len(groups),
            budget_hint=task_value,
        )
        return group_id, result

    draft_tasks = [_draft_for_group(gid, g) for gid, g in groups.items()]
    draft_results_list = await asyncio.gather(*draft_tasks, return_exceptions=True)

    drafts: dict[str, str] = {}
    for item in draft_results_list:
        if isinstance(item, Exception):
            log.warning("Executor draft subagent failed: %s", item)
            continue
        gid, text = item
        if not _is_subagent_error(text):
            drafts[gid] = text

    if not drafts:
        return await _call(
            client,
            _executor_model(pseudo_model, cfg, task_value, compacted, 0),
            [{"role": "user", "content": f"Original request:\n{user_text}\n\nAnalyst summary:\n{compacted}"}],
        )

    drafts_text = "\n\n".join(
        f"=== GROUP {gid} (scope: {', '.join(groups[gid]['scope'])}) ===\n{text}"
        for gid, text in sorted(drafts.items())
    )
    synthesis_prompt = (
        "You are the final executor. Below are proposed change drafts from per-file analysts. "
        "Synthesize them into a single coherent response to the original request. "
        "If drafts overlap for the same file, merge them intelligently.\n\n"
        f"ORIGINAL REQUEST:\n{user_text}\n\n"
        f"ANALYST SUMMARY:\n{compacted}\n\n"
        f"PROPOSED CHANGE DRAFTS:\n{drafts_text}"
    )
    executor_model = _executor_model(pseudo_model, cfg, task_value, compacted, len(drafts))
    return await _call(client, executor_model, [{"role": "user", "content": synthesis_prompt}])


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
        import time
        from autoconduck.config import get_config
        from autoconduck.messages_api import normalize_messages_for_llm

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

        async def recon_node(state: State) -> dict:
            if request is not None and hasattr(request, "is_disconnected") and await request.is_disconnected():
                return {"fallback": True}
            try:
                from autoconduck import stats
                stats.update_active_routing(
                    active=True, path="SLOW", pseudo_model=state.pseudo_model,
                    task_value=state.task_value, node="recon",
                    step_detail="Recon discovering target files...",
                    start_time=time.time()
                )
            except Exception:
                pass
            target = build_recon_plan(state.messages, client=client, cfg=cfg, task_value=state.task_value)
            return {"recon": target}

        async def recon_subagent_pool_node(state: State) -> dict:
            if state.recon is None or not getattr(state.recon, "files", []):
                return {"compacted": ""}
            files = state.recon.files[:5]
            try:
                from autoconduck import stats
                stats.update_active_routing(
                    node="subagents",
                    subtasks_total=len(files),
                    step_detail=f"Running {len(files)} recon analyst subagent(s)..."
                )
            except Exception:
                pass
            from .planner import _read_files
            raw_files = _read_files(files)
            evidence = [f"[{path}]\n{content[:1500]}" for path, content in raw_files.items()]
            return {"compacted": compact(evidence)}

        async def planner_node(state: State) -> dict:
            if request is not None and hasattr(request, "is_disconnected") and await request.is_disconnected():
                return {"fallback": True}
            try:
                from autoconduck import stats
                stats.update_active_routing(
                    node="planner",
                    step_detail="Planner generating JSON DAG task plan...",
                )
            except Exception:
                pass
            plan = build_task_plan(
                state.messages, client=client, cfg=cfg, task_value=state.task_value, ground_truth=state.compacted
            )
            if plan is not None:
                max_sub = int(getattr(getattr(cfg, "selection", None), "max_subtasks", 3))
                if len(plan.subtasks) > max_sub:
                    plan = plan.model_copy(update={"subtasks": plan.subtasks[:max_sub]})
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
            try:
                from autoconduck import stats
                stats.update_active_routing(
                    node="subagents",
                    subtasks_total=len(pending),
                    step_detail=f"Running {len(pending)} analyst subagent(s) in parallel..."
                )
            except Exception:
                pass
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
                try:
                    from autoconduck import stats
                    stats.update_active_routing(subtasks_completed=len(completed))
                except Exception:
                    pass
            error_count = sum(_is_subagent_error(value) for value in completed.values())
            return {"subagent_outputs": completed, "subagent_error_count": error_count}

        def compactor_node(state: State) -> dict:
            try:
                from autoconduck import stats
                stats.update_active_routing(
                    node="compactor", step_detail="Compacting analyst findings..."
                )
            except Exception:
                pass
            ordered = [
                state.subagent_outputs[task.id]
                for task in sorted(
                    state.plan.subtasks if state.plan else [],
                    key=lambda task: (-len(task.depends_on), task.id),
                )
                if task.id in state.subagent_outputs
                and not _is_subagent_error(state.subagent_outputs[task.id])
            ]
            merged = state.compacted + "\n" + compact(ordered) if state.compacted else compact(ordered)
            return {"compacted": merged}

        async def executor_node(state: State) -> dict:
            if request is not None and hasattr(request, "is_disconnected") and await request.is_disconnected():
                return {"fallback": True}
            try:
                from autoconduck import stats
                stats.update_active_routing(
                    node="executor", step_detail="Executor synthesizing final response..."
                )
            except Exception:
                pass
            user_text = "\n".join(
                str(m.get("content", ""))
                if isinstance(m, dict)
                else getattr(m, "content", str(m))
                for m in state.messages
                if not isinstance(m, dict) or m.get("role", "user") == "user"
            )
            task_value = getattr(state, "task_value", 0.5)
            valid_outputs = {
                task_id: output
                for task_id, output in state.subagent_outputs.items()
                if not _is_subagent_error(output)
            }

            plan = state.plan
            use_subagents = (
                plan is not None
                and getattr(getattr(cfg, "selection", None), "enable_executor_subagents", False)
                and len(plan.subtasks) >= 2
                and any(len(t.scope) > 0 for t in plan.subtasks)
            )
            if use_subagents:
                try:
                    result = await _run_executor_subagents(
                        plan=plan,
                        subagent_outputs=valid_outputs,
                        compacted=state.compacted,
                        user_text=user_text,
                        client=client,
                        cfg=cfg,
                        pseudo_model=state.pseudo_model,
                        task_value=task_value,
                    )
                    return {"result": result}
                except Exception:
                    pass

            analyst_summary = state.compacted
            prompt = f"Original request:\n{user_text}\n\nAnalyst summary:\n{analyst_summary}"
            return {
                "result": await _call(
                    client,
                    _executor_model(
                        state.pseudo_model,
                        cfg,
                        task_value,
                        analyst_summary,
                        len(valid_outputs),
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
        graph.add_node("recon", recon_node)
        graph.add_node("recon_subagent_pool", recon_subagent_pool_node)
        graph.add_node("planner", planner_node)
        graph.add_node("subagent_pool", subagent_pool_node)
        graph.add_node("compactor", compactor_node)
        graph.add_node("executor", executor_node)

        graph.add_edge(START, "recon")
        graph.add_edge("recon", "recon_subagent_pool")
        graph.add_edge("recon_subagent_pool", "planner")
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
                from autoconduck import stats
                stats.update_active_routing(active=False, node="idle", step_detail="Completed")
            except Exception:
                pass
    except Exception:
        return None
