from autoconduck.orchestrator.planner import _extract_json


def test_extract_json_fenced_and_nested():
    assert _extract_json('```json\n{"subtasks": [{"output_contract": {"description": "x"}}]}\n```') == '{"subtasks": [{"output_contract": {"description": "x"}}]}'


def test_extract_json_unfenced_and_truncated():
    assert _extract_json('prefix {"a": {"b": 1}} suffix') == '{"a": {"b": 1}}'
    assert _extract_json('{"subtasks":[{"id":"t1"') == '{"subtasks":[{"id":"t1"'
    assert _extract_json('no object') is None
