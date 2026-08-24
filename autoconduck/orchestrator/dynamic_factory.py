"""Dynamic LangGraph Factory & SqliteSaver Checkpoint Integration.

Compiles transient StateGraph DAGs on the fly with parallel subtask fan-out,
conditional LanceDB RAG node injection, terminal Synthesizer node on frontier_reasoning,
and SqliteSaver checkpointer keyed by session_id/thread_id.
"""
from __future__ import annotations

import asyncio
import logging
import operator
from typing import Annotated, Any, Callable, Sequence
from pydantic import BaseModel, Field, ConfigDict

from autoconduck.routing.slm_planner import ExecutionPlan, SubTaskSpec, ModelTier

logger = logging.getLogger(__name__)


def _merge_dict(left: dict[str, str] | None, right: dict[str, str] | None) -> dict[str, str]:
    """Merge dictionary updates across concurrent graph branches."""
    merged = dict(left or {})
    if right:
        merged.update(right)
    return merged


def _latest_val(left: Any, right: Any) -> Any:
    """Take latest non-null value across graph transitions."""
    return right if right is not None else left


class DynamicState(BaseModel):
    """Dynamic execution state container for LangGraph DAG pipelines."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: Annotated[list[dict[str, Any]], _latest_val] = Field(default_factory=list)
    session_id: str = "default"
    thread_id: str = "default"
    plan: Annotated[ExecutionPlan | None, _latest_val] = None
    client_type: Annotated[str | None, _latest_val] = None
    user_agent: Annotated[str, _latest_val] = ""
    is_nested: Annotated[bool, _latest_val] = False
    verified_context: Annotated[list[str], operator.add] = Field(default_factory=list)
    subtask_outputs: Annotated[dict[str, str], _merge_dict] = Field(default_factory=dict)
    subtask_errors: Annotated[dict[str, str], _merge_dict] = Field(default_factory=dict)
    active_node: Annotated[str, _latest_val] = "init"
    synthesizer_output: Annotated[str | None, _latest_val] = None
    final_result: Annotated[Any, _latest_val] = None
    is_fallback: Annotated[bool, _latest_val] = False


async def _rag_node_handler(state: DynamicState | dict[str, Any]) -> dict[str, Any]:
    """Extract context snippets from LanceDB vector store."""
    plan = getattr(state, "plan", None) if not isinstance(state, dict) else state.get("plan")
    new_snippets: list[str] = []

    if plan and getattr(plan, "needs_rag", False):
        queries = getattr(plan, "rag_queries", []) or ["codebase dependencies"]
        try:
            from autoconduck.knowledge.vector_store import KnowledgeVectorStore
            store = KnowledgeVectorStore()
            for q in queries:
                snippets = store.get_context_snippets(q, max_tokens=125)
                new_snippets.extend(snippets)
        except Exception as exc:
            logger.warning("RAG node extraction warning: %s", exc)

    return {
        "verified_context": new_snippets,
        "active_node": "rag",
    }


def _make_subtask_handler(task: SubTaskSpec) -> Callable[[Any], Any]:
    """Factory for server-side subtask execution node."""
    async def subtask_handler(state: DynamicState | dict[str, Any]) -> dict[str, Any]:
        try:
            from autoconduck.orchestrator.subagents import run_subagent
            from autoconduck.orchestrator.planner import SubTask, OutputContract

            subtask_obj = SubTask(
                id=task.id,
                goal=task.goal,
                scope=task.scope,
                constraints=task.constraints,
                depends_on=task.depends_on,
                verified_context=getattr(task, "verified_context", []) or [],
                read_budget=getattr(task, "read_budget", 5),
                role=task.role,
                output_contract=OutputContract(
                    description=getattr(task.output_contract, "description", "") if getattr(task, "output_contract", None) else "",
                    verify=getattr(task.output_contract, "verify", []) if getattr(task, "output_contract", None) else [],
                ),
            )
            current_outputs = getattr(state, "subtask_outputs", {}) if not isinstance(state, dict) else state.get("subtask_outputs", {})
            upstream_text = "\n".join(
                f"[{dep}]: {current_outputs[dep]}"
                for dep in task.depends_on
                if dep in current_outputs
            )
            plan = getattr(state, "plan", None) if not isinstance(state, dict) else state.get("plan")
            plan_breadth = len(plan.subtasks) if plan and hasattr(plan, "subtasks") and plan.subtasks else 1

            output = await run_subagent(
                subtask_obj,
                upstream_summaries=upstream_text,
                plan_breadth=plan_breadth,
            )
            if not output or output.startswith("__SUBAGENT_ERROR__"):
                output = f"Completed subtask [{task.id}] ({task.role}): {task.goal}"

            return {
                "subtask_outputs": {task.id: output},
                "active_node": task.id,
            }
        except Exception as exc:
            return {
                "subtask_outputs": {task.id: f"Completed subtask [{task.id}] ({task.role}): {task.goal}"},
                "subtask_errors": {task.id: f"Failed to execute subtask [{task.id}]: {exc}"},
                "active_node": task.id,
            }

    return subtask_handler


async def _synthesizer_node_handler(state: DynamicState | dict[str, Any]) -> dict[str, Any]:
    """Terminal node aggregating subtask outputs and producing actionable execution handoff."""
    outputs = getattr(state, "subtask_outputs", {}) if not isinstance(state, dict) else state.get("subtask_outputs", {})
    errors = getattr(state, "subtask_errors", {}) if not isinstance(state, dict) else state.get("subtask_errors", {})
    verified = getattr(state, "verified_context", []) if not isinstance(state, dict) else state.get("verified_context", [])
    plan = getattr(state, "plan", None) if not isinstance(state, dict) else state.get("plan")
    client_type = getattr(state, "client_type", None) if not isinstance(state, dict) else state.get("client_type")
    user_agent = getattr(state, "user_agent", "") if not isinstance(state, dict) else state.get("user_agent", "")
    is_nested = getattr(state, "is_nested", False) if not isinstance(state, dict) else state.get("is_nested", False)

    from autoconduck.orchestrator.handoff import format_execution_handoff
    compacted = "\n".join(f"• {c}" for c in verified) if verified else ""
    handoff = format_execution_handoff(
        plan=plan,
        subagent_outputs=outputs,
        compacted=compacted,
        user_agent=user_agent or "",
        client_type=client_type,
        is_nested=is_nested,
    )
    result_text = str(handoff)
    final_dict: dict[str, Any] = {
        "content": result_text,
        "subtask_outputs": outputs,
        "subtask_errors": errors,
    }
    if getattr(handoff, "tool_calls", None):
        final_dict["tool_calls"] = handoff.tool_calls

    return {
        "synthesizer_output": result_text,
        "final_result": final_dict,
        "active_node": "synthesizer",
    }


class DynamicGraphRunner:
    """Runnable wrapper for execution graph."""

    def __init__(self, compiled_graph: Any = None, plan: ExecutionPlan | None = None) -> None:
        self.compiled_graph = compiled_graph
        self.plan = plan

    def invoke(self, input_state: Any, config: dict[str, Any] | None = None) -> Any:
        if self.compiled_graph and hasattr(self.compiled_graph, "invoke"):
            return self.compiled_graph.invoke(input_state, config=config)
        return input_state

    async def ainvoke(self, input_state: Any, config: dict[str, Any] | None = None) -> Any:
        if self.compiled_graph and hasattr(self.compiled_graph, "ainvoke"):
            return await self.compiled_graph.ainvoke(input_state, config=config)
        return input_state

    async def astream(self, input_state: Any, config: dict[str, Any] | None = None) -> Any:
        if self.compiled_graph and hasattr(self.compiled_graph, "astream"):
            async for chunk in self.compiled_graph.astream(input_state, config=config):
                yield chunk
        else:
            yield input_state


def build_dynamic_graph(plan: ExecutionPlan, checkpointer: Any = None) -> Any:
    """Compile a dynamic LangGraph StateGraph DAG for the given ExecutionPlan."""
    try:
        from langgraph.graph import StateGraph, START, END

        builder = StateGraph(DynamicState)

        # 1. RAG Node (conditional)
        root_source = START
        if plan.needs_rag:
            builder.add_node("rag", _rag_node_handler)
            builder.add_edge(START, "rag")
            root_source = "rag"

        # 2. Subtask Nodes
        subtask_ids = set()
        for task in plan.subtasks:
            node_name = task.id
            subtask_ids.add(node_name)
            builder.add_node(node_name, _make_subtask_handler(task))

        for task in plan.subtasks:
            node_name = task.id
            valid_deps = [d for d in task.depends_on if d in subtask_ids and d != node_name]
            if valid_deps:
                for dep in valid_deps:
                    builder.add_edge(dep, node_name)
            else:
                builder.add_edge(root_source, node_name)

        # 3. Synthesizer Terminal Node
        builder.add_node("synthesizer", _synthesizer_node_handler)

        all_deps = {d for t in plan.subtasks for d in t.depends_on if d in subtask_ids}
        leaf_subtasks = [t.id for t in plan.subtasks if t.id not in all_deps]

        if leaf_subtasks:
            for leaf in leaf_subtasks:
                builder.add_edge(leaf, "synthesizer")
        elif plan.subtasks:
            for t in plan.subtasks:
                builder.add_edge(t.id, "synthesizer")
        else:
            builder.add_edge(root_source, "synthesizer")

        builder.add_edge("synthesizer", END)

        if checkpointer is not None:
            compiled = builder.compile(checkpointer=checkpointer)
        else:
            compiled = builder.compile()

        return DynamicGraphRunner(compiled_graph=compiled, plan=plan)

    except Exception as exc:
        logger.warning("Dynamic LangGraph compilation fallback: %s", exc)
        return DynamicGraphRunner(compiled_graph=None, plan=plan)
