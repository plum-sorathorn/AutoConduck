"""LangGraph orchestrator, planner, subagents, and executor blueprint unit tests."""
import json
import pytest
from unittest.mock import MagicMock, patch

from autoconduck.config import Config, SelectionConfig
from autoconduck.orchestrator.compactor import compact
from autoconduck.orchestrator.planner import (
    OutputContract,
    SubTask,
    TaskPlan,
    build_task_plan,
)
from autoconduck.orchestrator.recon import ReconTarget, build_recon_plan
from autoconduck.orchestrator.roles import RoleConfig, role_card
from autoconduck.orchestrator.subagents import (
    build_subagent_prompt,
    run_subagent,
    subagent_target,
)


def test_subtask_and_task_plan_schema():
    subtask = SubTask(
        id="t1",
        goal="Inspect routing logic",
        scope=["autoconduck/routing/dispatcher.py"],
        output_contract=OutputContract(description="Summary of routing rules"),
        constraints=["Do not modify files"],
        depends_on=[],
        role="read",
    )
    plan = TaskPlan(
        subtasks=[subtask],
        summary="Test task plan",
        budget_hint=0.5,
    )
    assert plan.subtasks[0].id == "t1"
    assert plan.budget_hint == 0.5


def test_build_subagent_prompt_structure():
    task = SubTask(
        id="t1",
        goal="Inspect authentication",
        scope=["autoconduck/auth.py"],
        output_contract=OutputContract(description="Security notes"),
        constraints=["Do not edit"],
        verified_context=["Auth uses YAML file"],
        read_budget=3,
        role="read",
    )
    prompt = build_subagent_prompt(task, "Sibling context")
    assert "ROLE: You are a read-only file analyst" in prompt
    assert "FILES IN SCOPE (only these): autoconduck/auth.py" in prompt
    assert "VERIFIED CONTEXT" in prompt
    assert "TOOL BUDGET: You may make at most 3" in prompt


def test_subagent_target_read_vs_write():
    cfg = Config(
        model_list=[
            {"id": "cheap", "price_in": 0.1, "price_out": 0.1, "enabled": True},
            {"id": "pricey", "price_in": 2.0, "price_out": 5.0, "enabled": True},
        ],
        selection=SelectionConfig(phase_bands={"subagent": [0.10, 0.55]}),
    )
    read_target = subagent_target("analyze", "read", 1, 0.5, cfg)
    write_target = subagent_target("analyze", "write", 1, 0.5, cfg)
    assert read_target < write_target


def test_compact_dedupes_references():
    findings = [
        "Issue detected at autoconduck/auth.py:25",
        "Issue detected at autoconduck/auth.py:25",
        "Different issue at autoconduck/config.py:10",
    ]
    merged = compact(findings)
    assert merged.count("autoconduck/auth.py:25") == 1
    assert "autoconduck/config.py:10" in merged


def test_recon_plan_target_extraction():
    prompt = "Please check autoconduck/digest.py and autoconduck/config.py"
    plan = build_recon_plan([{"role": "user", "content": prompt}], cfg=Config())
    # Deterministic file path extraction from prompt
    assert isinstance(plan, ReconTarget)
    assert isinstance(plan.files, list)


def test_role_card_generation():
    reviewer = role_card("reviewer")
    assert "read" in reviewer.lower() or "review" in reviewer.lower()
    executor = role_card("executor")
    assert len(executor) > 10


def test_extract_text_tool_calls_opensource_format():
    from autoconduck.orchestrator.executor_loop import extract_text_tool_calls, strip_tool_call_tags

    sample = (
        "I understand. Let me check the files first.<tool_calls:opensource>\n"
        "<tool_call:opensource>read<tool_sep:opensource>\n"
        "<arg_key:opensource>path</arg_key:opensource>\n"
        "<arg_value:opensource>autoconduck/subagents.py</arg_value:opensource>\n"
        "</tool_call:opensource>\n"
        "<tool_call:opensource>read<tool_sep:opensource>\n"
        "<arg_key:opensource>path</arg_key:opensource>\n"
        "<arg_value:opensource>autoconduck/tools.py</arg_value:opensource>\n"
        "</tool_call:opensource>"
    )
    calls = extract_text_tool_calls(sample)
    assert len(calls) == 2
    assert calls[0]["function"]["name"] == "read"
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "autoconduck/subagents.py"}
    assert calls[1]["function"]["name"] == "read"
    assert json.loads(calls[1]["function"]["arguments"]) == {"path": "autoconduck/tools.py"}

    cleaned = strip_tool_call_tags(sample)
    assert "I understand. Let me check the files first." in cleaned
    assert "<tool_calls:opensource>" not in cleaned


def test_extract_text_tool_calls_generic_json():
    from autoconduck.orchestrator.executor_loop import extract_text_tool_calls, strip_tool_call_tags

    sample = '<tool_call>{"name": "grep", "arguments": {"pattern": "def score"}}</tool_call>'
    calls = extract_text_tool_calls(sample)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "grep"
    assert json.loads(calls[0]["function"]["arguments"]) == {"pattern": "def score"}

    cleaned = strip_tool_call_tags(sample)
    assert cleaned == ""


def test_read_only_tools_are_routed_to_fast_model():
    from autoconduck.orchestrator.tools import is_read_only_tool, tool_model

    cfg = Config(
        model_list=[
            {"id": "cheap", "price_in": 0.1, "price_out": 0.1, "enabled": True},
            {"id": "pricey", "price_in": 2.0, "price_out": 5.0, "enabled": True},
        ]
    )
    assert is_read_only_tool("read")
    assert is_read_only_tool("grep")
    assert tool_model("read", "pricey", cfg) == "cheap"
    assert tool_model("edit", "pricey", cfg) == "pricey"


@pytest.mark.asyncio
async def test_executor_stagnation_injects_mitigation_note(tmp_path):
    from autoconduck.orchestrator.executor_loop import (
        LoopState,
        calculate_stagnation,
        run_executor_tool_loop,
    )

    class FakeClient:
        def __init__(self):
            self.messages = []

        def completion(self, *, messages, **kwargs):
            self.messages.append((messages, kwargs))
            return {"choices": [{"message": {"content": "", "tool_calls": [{
                "id": str(len(self.messages)), "function": {
                    "name": "read", "arguments": '{"path":"missing.py"}'
                }
            }]}}]}

    client = FakeClient()
    cfg = Config(model_list=[
        {"id": "cheap", "price_in": 0.1, "price_out": 0.1, "enabled": True},
        {"id": "pricey", "price_in": 2.0, "price_out": 5.0, "enabled": True},
    ])
    result = await run_executor_tool_loop(client, "pricey", "system", "user",
        allowed_scope=["missing.py"], workspace_root=tmp_path, cfg=cfg,
        max_rounds=4, tool_retry_cap=10)
    assert isinstance(result, str)
    assert any("<loop-stagnation:true>" in str(message.get("content", ""))
               for messages, _ in client.messages for message in messages)
    assert any(kwargs.get("model") == "openai/cheap"
               for _, kwargs in client.messages[3:])

    state = LoopState(
        call_signatures=["same", "same", "same"],
        error_streak=3,
        distinct_files_touched={"missing.py"},
        total_calls=3,
    )
    assert calculate_stagnation(state) > 0.70


def test_skeletons_python_ast():
    from autoconduck.orchestrator.skeletons import extract_python_skeleton

    code = '''"""Module docstring for test."""
import sys
from os import path

class Router:
    """Class docstring."""
    def __init__(self, mode: str = "fast") -> None:
        self.mode = mode

    def route(self, query: str) -> bool:
        """Route method doc."""
        return True

def standalone_fn(val: int) -> str:
    """Helper doc."""
    return str(val)
'''
    skel = extract_python_skeleton(code)
    assert '"""Module docstring for test."""' in skel
    assert "from os import path" in skel
    assert "class Router:" in skel
    assert "def route(self, query: str) -> bool" in skel
    assert "def standalone_fn(val: int) -> str" in skel
    assert "self.mode = mode" not in skel  # Implementation body omitted


def test_skeletons_gitignore_and_excludes(tmp_path):
    from autoconduck.orchestrator.skeletons import is_ignored_path, load_gitignore_patterns

    gi_file = tmp_path / ".gitignore"
    gi_file.write_text("secrets/\n*.tmp\ncustom_build/\n", encoding="utf-8")
    patterns = load_gitignore_patterns(tmp_path)

    assert is_ignored_path("graphify-out/GRAPH_REPORT.md", tmp_path, patterns) is True
    assert is_ignored_path(".autoconduck/run/server.log", tmp_path, patterns) is True
    assert is_ignored_path("package-lock.json", tmp_path, patterns) is True
    assert is_ignored_path("secrets/keys.json", tmp_path, patterns) is True
    assert is_ignored_path("custom_build/output.js", tmp_path, patterns) is True
    assert is_ignored_path("autoconduck/config.py", tmp_path, patterns) is False


def test_skeletons_dependency_map():
    from autoconduck.orchestrator.skeletons import extract_dependency_map

    files = {
        "autoconduck/orchestrator/graph.py": "from .planner import TaskPlan\nimport sys",
        "autoconduck/orchestrator/planner.py": "class TaskPlan: pass\n",
    }
    dep_map = extract_dependency_map(files)
    assert "CROSS-FILE DEPENDENCY MAP:" in dep_map
    assert "autoconduck/orchestrator/graph.py -> autoconduck/orchestrator/planner.py" in dep_map


def test_planner_extract_file_paths_ignores_gitignore(tmp_path):
    from autoconduck.orchestrator.planner import _extract_file_paths

    (tmp_path / "valid.py").write_text("print(1)", encoding="utf-8")
    (tmp_path / "ignored.tmp").write_text("temp", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("*.tmp\n", encoding="utf-8")

    msgs = [
        {"role": "user", "content": "Please check valid.py and ignored.tmp"}
    ]
    paths = _extract_file_paths(msgs, root=tmp_path)
    assert "valid.py" in paths
    assert "ignored.tmp" not in paths


def test_format_execution_handoff_with_subagents():
    from autoconduck.orchestrator.handoff import format_execution_handoff
    from autoconduck.orchestrator.planner import TaskPlan, SubTask, OutputContract

    plan = TaskPlan(
        subtasks=[
            SubTask(
                id="t1",
                goal="Inspect token handling",
                scope=["autoconduck/auth.py"],
                output_contract=OutputContract(description="Auth notes", verify=["pytest"]),
                constraints=["Do not modify keys"],
                depends_on=[],
            ),
            SubTask(
                id="t2",
                goal="Update API endpoint",
                scope=["autoconduck/messages_api.py"],
                output_contract=OutputContract(description="API notes", verify=[]),
                constraints=[],
                depends_on=["t1"],
            ),
        ],
        summary="Refactor auth tokens",
    )
    outputs = {"t1": "auth.py:25 has get_provider_key."}
    with patch("autoconduck.orchestrator.handoff.check_subagent_support", return_value=(True, True)):
        res = format_execution_handoff(plan, outputs, "")
        assert "## Implementation Plan & Verified Context" in res
        assert "Refactor auth tokens" in res
        assert "#### 1. `t1`: Inspect token handling" in res
        assert "#### 2. `t2`: Update API endpoint" in res
        assert "auth.py:25 has get_provider_key." in res
        assert "pi-subagents" in res
        assert res.tool_calls is not None
        assert len(res.tool_calls) == 1
        assert res.tool_calls[0]["function"]["name"] == "subagent"
        args = json.loads(res.tool_calls[0]["function"]["arguments"])
        assert "workflowScript" in args
        assert "runs.all" in args["workflowScript"] or "runs.run" in args["workflowScript"]
        assert "worker" in args["workflowScript"]
        assert "pytest" in args["workflowScript"]


def test_format_execution_handoff_linear_fallback_warning():
    from autoconduck.orchestrator.handoff import format_execution_handoff
    from autoconduck.orchestrator.planner import TaskPlan, SubTask

    plan = TaskPlan(
        subtasks=[
            SubTask(id="t1", goal="Fix bug", scope=["src/app.py"], constraints=[]),
        ],
        summary="Fix bug in app",
    )
    with patch("autoconduck.orchestrator.handoff.check_subagent_support", return_value=(True, False)):
        res = format_execution_handoff(plan, {}, "")
        assert "Linear Execution Mode" in res
        assert "pi-subagents" in res
        assert "Executing subtasks linearly" in res


def test_resolve_1hop_dependencies(tmp_path):
    from autoconduck.orchestrator.skeletons import resolve_1hop_dependencies

    # Setup dummy project files
    (tmp_path / "main.py").write_text("import helper\nfrom config import settings\n", encoding="utf-8")
    (tmp_path / "helper.py").write_text("def assist(): pass\n", encoding="utf-8")
    (tmp_path / "config.py").write_text("settings = {}\n", encoding="utf-8")

    resolved = resolve_1hop_dependencies(["main.py"], root=tmp_path, max_total_files=5)
    assert "main.py" in resolved
    assert "helper.py" in resolved or "config.py" in resolved


def test_resolve_analysts_for_task():
    from autoconduck.orchestrator.roles import resolve_analysts_for_task

    # Low complexity with single file -> 1 analyst
    analysts_low = resolve_analysts_for_task("fix typo in auth.py", ["auth.py"], task_value=0.3)
    assert len(analysts_low) == 1
    assert analysts_low[0][0] == "reviewer"

    # High complexity with bug/test -> 3 analysts
    analysts_high = resolve_analysts_for_task("fix concurrency bug in session handling and run pytest", ["session.py", "cache.py"], task_value=0.8)
    assert len(analysts_high) == 3
    roles = [r[0] for r in analysts_high]
    assert "reviewer" in roles
    assert "scout" in roles
    assert "oracle" in roles


def test_check_subagent_support_agent_filtering():
    from autoconduck.orchestrator.handoff import check_subagent_support

    # Non-pi client types must always return False, False
    assert check_subagent_support(client_type="claude") == (False, False)
    assert check_subagent_support(client_type="opencode") == (False, False)

    # Non-pi user agents must return False, False
    assert check_subagent_support(user_agent="OpenCode/1.0") == (False, False)
    assert check_subagent_support(user_agent="Claude-Code/0.2.9") == (False, False)
    assert check_subagent_support(user_agent="anthropic-sdk-typescript/0.27.0") == (False, False)
