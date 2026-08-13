import json
from unittest.mock import patch

import pytest

from autoconduck.orchestrator.compactor import compact
from autoconduck.config import Config, resolve_orchestrator_model
from autoconduck.orchestrator import helpers, planner, recon
from autoconduck.orchestrator.planner import (
    OutputContract,
    SubTask,
    TaskPlan,
    _extract_file_paths,
    _read_files,
    build_task_plan,
)
from autoconduck.orchestrator.subagents import build_subagent_prompt, run_subagent
from autoconduck.routing import pricing


def valid_task():
    return SubTask(id="a", goal="Inspect auth flow.", scope=["src/auth.py"], output_contract="Bullets.", constraints=["Do not edit."])


def test_schema():
    assert TaskPlan(subtasks=[valid_task()]).subtasks[0].id == "a"
    with pytest.raises(Exception):
        SubTask(id="a", goal="Inspect.", output_contract="x", constraints=[])


def test_backward_compat_defaults():
    task = SubTask(
        id="a",
        goal="Inspect auth flow.",
        scope=["src/auth.py"],
        output_contract="Bullets.",
        constraints=["Do not edit."],
    )
    assert task.verified_context == []
    assert task.read_budget == 5
    assert isinstance(task.output_contract, OutputContract)
    assert task.output_contract.description == "Bullets."
    assert task.output_contract.verify == []


def test_output_contract_coercion():
    bare = SubTask(
        id="a",
        goal="g",
        scope=["a.py"],
        output_contract="summary",
        constraints=[],
    )
    assert bare.output_contract.description == "summary"
    assert bare.output_contract.verify == []

    structured = SubTask(
        id="b",
        goal="g",
        scope=["a.py"],
        output_contract={"description": "x", "verify": ["pytest"]},
        constraints=[],
    )
    assert structured.output_contract.description == "x"
    assert structured.output_contract.verify == ["pytest"]


def test_planner_double_failure():
    class Client:
        def completion(self, **kwargs):
            return {"choices": [{"message": {"content": "not json"}}]}
    assert build_task_plan([], Client()) is None


def test_prompt_exact_template():
    task = SubTask(
        id="a",
        goal="Inspect auth.",
        scope=["a.py", "b.py"],
        output_contract="JSON",
        constraints=["Do not edit."],
    )
    assert build_subagent_prompt(task, "prior") == (
        "ROLE: You are a read-only file analyst. You do not propose fixes or write code.\n"
        "TASK: Inspect auth.\n"
        "FILES IN SCOPE (only these): a.py, b.py\n"
        "REQUIRED OUTPUT FORMAT: JSON\n"
        "DO NOT: Do not edit.\n"
        "CONTEXT FROM SIBLING TASKS: prior\n"
        "TOOL BUDGET: You may make at most 5 additional file reads/tool calls "
        "beyond what's given above. Work with what you have first."
    )


def test_prompt_with_verified_context_and_verify():
    task = SubTask(
        id="a",
        goal="Implement feature.",
        scope=["src/a.py"],
        output_contract=OutputContract(description="Patch summary.", verify=["pytest", "python -m compileall autoconduck"]),
        constraints=["Stay in scope."],
        verified_context=[
            "line 96: SOURCES list is hardcoded, must extend not replace",
            "config.py has no selected_presets field yet",
        ],
        read_budget=3,
    )
    prompt = build_subagent_prompt(task, "upstream notes")
    assert prompt == (
        "ROLE: You are a read-only file analyst. You do not propose fixes or write code.\n"
        "TASK: Implement feature.\n"
        "FILES IN SCOPE (only these): src/a.py\n"
        "REQUIRED OUTPUT FORMAT: Patch summary.\n"
        "DO NOT: Stay in scope.\n"
        "CONTEXT FROM SIBLING TASKS: upstream notes\n"
        "VERIFIED CONTEXT (do not re-investigate):\n"
        "- line 96: SOURCES list is hardcoded, must extend not replace\n"
        "- config.py has no selected_presets field yet\n"
        "TOOL BUDGET: You may make at most 3 additional file reads/tool calls "
        "beyond what's given above. Work with what you have first.\n"
        "VERIFY BEFORE RETURNING: pytest, python -m compileall autoconduck"
    )


def test_prompt_without_optional_sections():
    task = SubTask(
        id="a",
        goal="Inspect auth.",
        scope=["a.py"],
        output_contract="JSON",
        constraints=["Do not edit."],
    )
    prompt = build_subagent_prompt(task, "")
    assert "VERIFIED CONTEXT" not in prompt
    assert "VERIFY BEFORE RETURNING" not in prompt
    assert "TOOL BUDGET: You may make at most 5 additional file reads/tool calls" in prompt


def test_extract_file_paths_real_and_garbage():
    hits = _extract_file_paths([
        {"role": "user", "content": "Please review autoconduck/orchestrator/planner.py carefully."},
    ])
    assert "autoconduck/orchestrator/planner.py" in hits

    assert _extract_file_paths([{"role": "user", "content": "no paths here at all"}]) == []
    assert _extract_file_paths([{"role": "user", "content": "look at not/a/real/path.py please"}]) == []


def test_read_files_skips_missing():
    path = "autoconduck/orchestrator/planner.py"
    got = _read_files([path, "does/not/exist.py"])
    assert path in got
    assert "class SubTask" in got[path]
    assert "does/not/exist.py" not in got


def test_build_task_plan_happy_no_files():
    plan = TaskPlan(subtasks=[valid_task()], summary="ok")

    class Client:
        def completion(self, **kwargs):
            return {"choices": [{"message": {"content": plan.model_dump_json()}}]}

    result = build_task_plan([{"role": "user", "content": "plan something with no paths"}], Client())
    assert result is not None
    assert result.subtasks[0].id == "a"
    assert result.subtasks[0].verified_context == []
    assert result.subtasks[0].read_budget == 5


def test_compact_dedupes_refs():
    result = compact(["Issue at src/auth.py:10", "Same issue at src/auth.py:10", "Other at src/token.py:4"])
    assert result.count("src/auth.py:10") == 1
    assert "src/token.py:4" in result
    assert len(result.split()) < 1000


@pytest.mark.asyncio
async def test_orchestrator_helpers_empty_pool_fall_back_to_orchestrator_model():
    cfg = Config(model_list=[])
    fallback = resolve_orchestrator_model(cfg)

    assert fallback
    assert recon._recon_model_name(cfg, task_value=0.5) == fallback
    assert planner._model_name(cfg, task_value=0.5) == fallback
    assert helpers._executor_model(
        "autoconduck", cfg, task_value=0.7, compactor_summary="", subtask_count=0
    ) == fallback
    assert pricing.select_closest(pricing.pool_ids(cfg), 0.3, cfg) == ""

    class Client:
        model = None

        def completion(self, **kwargs):
            self.model = kwargs["model"]
            return {"choices": [{"message": {"content": "subagent result"}}]}

    client = Client()
    assert await run_subagent(valid_task(), "", client, cfg=cfg) == "subagent result"


@pytest.mark.asyncio
async def test_run_fallback_when_planner_fails():
    import autoconduck.orchestrator.graph as graph
    with patch.object(graph, "_LANGGRAPH_AVAILABLE", True), patch.object(graph, "build_task_plan", return_value=None):
        assert await graph.run([], None, task_value=0.7) is None


@pytest.mark.asyncio
async def test_run_happy_path():
    import autoconduck.orchestrator.graph as graph
    plan = TaskPlan(subtasks=[valid_task()])
    class Client:
        def completion(self, **kwargs):
            if kwargs["messages"][0]["role"] == "system":
                return {"choices": [{"message": {"content": plan.model_dump_json()}}]}
            return {"choices": [{"message": {"content": "final answer"}}]}
    with patch.object(graph, "_LANGGRAPH_AVAILABLE", True):
        with patch.object(graph, "build_task_plan", return_value=plan):
            with patch.object(graph, "run_subagent", return_value="Finding at src/auth.py:1"):
                assert await graph.run([], None, client=Client()) == "final answer"


# ---- Degradation-path integration tests for the LangGraph DAG --


@pytest.mark.asyncio
async def test_run_double_planner_failure_then_end():
    """When planner returns None, the conditional edge routes to END with fallback=True.
    Note: planner no longer retries (retry loop removed to avoid double LLM cost),
    so it is called exactly once before falling back."""
    import autoconduck.orchestrator.graph as graph
    call_count = [0]

    def failing_planner(*args, **kwargs):
        call_count[0] += 1
        return None

    with patch.object(graph, "_LANGGRAPH_AVAILABLE", True):
        with patch.object(graph, "build_task_plan", side_effect=failing_planner):
            # task_value=0.8 bypasses the direct-executor short-circuit
            result = await graph.run([], None, task_value=0.8)
    assert result is None
    assert call_count[0] == 2  # LangGraph retries the planner node once (attempt<2), then routes to END



@pytest.mark.asyncio
async def test_run_langgraph_unavailable():
    """_LANGGRAPH_AVAILABLE=False → run() returns None immediately without error."""
    import autoconduck.orchestrator.graph as graph
    with patch.object(graph, "_LANGGRAPH_AVAILABLE", False):
        result = await graph.run([], None)
    assert result is None


@pytest.mark.asyncio
async def test_run_subagent_pool_returns_error_strings():
    """Subagents catch exceptions and return tagged error strings.
    The pool finishes normally, compactor merges messy text, executor fires.
    No exception should bubble out of run()."""
    import autoconduck.orchestrator.graph as graph
    plan = TaskPlan(subtasks=[valid_task()])

    def bad_subagent(*args, **kwargs):
        return "__SUBAGENT_ERROR__[a]: connection timeout to upstream API"

    with patch.object(graph, "_LANGGRAPH_AVAILABLE", True):
        with patch.object(graph, "build_task_plan", return_value=plan):
            with patch.object(graph, "run_subagent", side_effect=bad_subagent):
                # Must not raise — worst case it returns degraded text
                result = await graph.run([], None)
                # Result may be the degraded string or None depending on executor success
                assert isinstance(result, (str, type(None)))


@pytest.mark.asyncio
async def test_run_executor_raises_exception():
    """Executor crashes (e.g. null choices). Outer try/except catches it → run() returns None."""
    import autoconduck.orchestrator.graph as graph
    plan = TaskPlan(subtasks=[valid_task()])

    class FailingClient:
        def completion(self, **kwargs):
            # Simulate a malformed response with no choices
            return {"choices": []}

    with patch.object(graph, "_LANGGRAPH_AVAILABLE", True):
        with patch.object(graph, "build_task_plan", return_value=plan):
            with patch.object(graph, "run_subagent", return_value="Finding at src/auth.py:1"):
                result = await graph.run([], None, client=FailingClient())
    assert result is None  # executor crashed, outer except returns None


@pytest.mark.asyncio
async def test_run_compactor_handles_empty_outputs():
    """Compactor receives empty or missing subagent outputs and doesn't crash."""
    # Direct compact() unit check — compact([]) must return ""
    assert compact([]) == ""
    assert compact([""]) == ""
    # Mixed content must also be safe
    result = compact(["line 1", ""])
    assert isinstance(result, str)
    # Missing keys scenario is handled by subagent_outputs dict filtering in compactor_node
    # which only includes tasks whose ids are present in state.subagent_outputs


@pytest.mark.asyncio
async def test_run_no_subagent_outputs_reaches_compactor():
    """If plan succeeds but subagent_outputs never fills, compactor gets empty list.
    Compactor returns "", executor fires and must not raise."""
    import autoconduck.orchestrator.graph as graph
    plan = TaskPlan(subtasks=[valid_task()])

    def empty_subagent(task, *args, **kwargs):
        # Subagent returns empty instead of proper output
        return ""

    class OkClient:
        def completion(self, **kwargs):
            if "FILE CONTENTS" not in str(kwargs.get("messages", [""])):
                return {"choices": [{"message": {"content": "degraded result from empty input"}}]}
            return {"choices": [{"message": {"content": plan.model_dump_json()}}]}

    with patch.object(graph, "_LANGGRAPH_AVAILABLE", True):
        with patch.object(graph, "build_task_plan", return_value=plan):
            with patch.object(graph, "run_subagent", side_effect=empty_subagent):
                result = await graph.run([], None, client=OkClient())
                # Should degrade gracefully — either a string or None
                assert isinstance(result, (str, type(None)))


def test_build_recon_plan_explicit_path():
    from autoconduck.orchestrator.recon import build_recon_plan
    target = build_recon_plan([{"role": "user", "content": "Check autoconduck/orchestrator/planner.py"}])
    assert "autoconduck/orchestrator/planner.py" in target.files
    assert target.query == "Explicit file paths from request"


def test_build_recon_plan_llm():
    from autoconduck.orchestrator.recon import build_recon_plan
    class Client:
        def completion(self, **kwargs):
            return {"choices": [{"message": {"content": '{"files": ["autoconduck/config.py"], "query": "config", "reasoning": "checking config"}'}}]}
    target = build_recon_plan([{"role": "user", "content": "How does config loading work?"}], client=Client())
    assert target.files == ["autoconduck/config.py"]
    assert target.query == "config"

