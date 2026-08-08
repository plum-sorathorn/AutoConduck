import json
from unittest.mock import patch

import pytest

from autoconduck.orchestrator.compactor import compact
from autoconduck.orchestrator.planner import (
    OutputContract,
    SubTask,
    TaskPlan,
    _extract_file_paths,
    _read_files,
    build_task_plan,
)
from autoconduck.orchestrator.subagents import build_subagent_prompt


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


def test_run_fallback_when_planner_fails():
    import autoconduck.orchestrator.graph as graph
    with patch.object(graph, "_LANGGRAPH_AVAILABLE", True), patch.object(graph, "build_task_plan", return_value=None):
        assert graph.run([], None) is None


def test_run_happy_path():
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
                assert graph.run([], None, client=Client()) == "final answer"
