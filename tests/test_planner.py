from autoconduck.orchestrator.planner import _extract_json


def test_extract_json_fenced_and_nested():
    assert _extract_json('```json\n{"subtasks": [{"output_contract": {"description": "x"}}]}\n```') == '{"subtasks": [{"output_contract": {"description": "x"}}]}'


def test_extract_json_unfenced_and_truncated():
    assert _extract_json('prefix {"a": {"b": 1}} suffix') == '{"a": {"b": 1}}'
    assert _extract_json('{"subtasks":[{"id":"t1"') == '{"subtasks":[{"id":"t1"'
    assert _extract_json('no object') is None


def test_task_plan_loose_type_coercion():
    from autoconduck.orchestrator.planner import TaskPlan

    # Loose output_contract as list, read_budget as empty string, scope as single string
    data = {
        "subtasks": [
            {
                "id": "t1",
                "goal": "check dashboard",
                "scope": "autoconduck/tui/dashboard.py",
                "output_contract": ["description: analyze keybindings"],
                "constraints": "do not edit",
                "read_budget": "",
            },
            {
                "id": "t2",
                "goal": "check menu",
                "scope": ["autoconduck/tui/menu.py"],
                "output_contract": "summarize",
                "constraints": [],
                "read_budget": "10",
            },
        ]
    }
    plan = TaskPlan.model_validate(data)
    assert len(plan.subtasks) == 2
    assert plan.subtasks[0].scope == ["autoconduck/tui/dashboard.py"]
    assert plan.subtasks[0].output_contract.description == "description: analyze keybindings"
    assert plan.subtasks[0].constraints == ["do not edit"]
    assert plan.subtasks[0].read_budget == 5
    assert plan.subtasks[1].read_budget == 10
    assert plan.subtasks[1].output_contract.description == "summarize"


def test_task_plan_dict_subtasks_coercion():
    from autoconduck.orchestrator.planner import TaskPlan

    data = {
        "subtasks": {
            "0": {"id": "t1", "goal": "step 1", "scope": [], "constraints": []},
            "1": {"id": "t2", "goal": "step 2", "scope": [], "constraints": []},
        }
    }
    plan = TaskPlan.model_validate(data)
    assert len(plan.subtasks) == 2
    assert plan.subtasks[0].id == "t1"
    assert plan.subtasks[1].id == "t2"

