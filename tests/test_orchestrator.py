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
