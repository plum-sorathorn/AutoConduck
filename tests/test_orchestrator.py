import json
from unittest.mock import patch

import pytest

from autoconduck.orchestrator.compactor import compact
from autoconduck.orchestrator.planner import SubTask, TaskPlan, build_task_plan
from autoconduck.orchestrator.subagents import build_subagent_prompt


def valid_task():
    return SubTask(id="a", goal="Inspect auth flow.", scope=["src/auth.py"], output_contract="Bullets.", constraints=["Do not edit."])


def test_schema():
    assert TaskPlan(subtasks=[valid_task()]).subtasks[0].id == "a"
    with pytest.raises(Exception):
        SubTask(id="a", goal="Inspect.", output_contract="x", constraints=[])


def test_planner_double_failure():
    class Client:
        def completion(self, **kwargs):
            return {"choices": [{"message": {"content": "not json"}}]}
    assert build_task_plan([], Client()) is None


def test_prompt_exact_template():
    task = SubTask(id="a", goal="Inspect auth.", scope=["a.py", "b.py"], output_contract="JSON", constraints=["Do not edit."])
    assert build_subagent_prompt(task, "prior") == (
        "ROLE: You are a read-only file analyst. You do not propose fixes or write code.\n"
        "TASK: Inspect auth.\nFILES IN SCOPE (only these): a.py, b.py\n"
        "REQUIRED OUTPUT FORMAT: JSON\nDO NOT: Do not edit.\n"
        "CONTEXT FROM SIBLING TASKS: prior"
    )


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
