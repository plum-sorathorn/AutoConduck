import sys
from unittest.mock import patch

from autoconduck.jsonutil import parse_json_text


def test_jsonutil_repairs_common_model_outputs():
    for text in ('{"a": 1}', '```json\n{"a": 1}\n```', '{"a": 1', '{"a": "unterminated', '{"a":[1,2,3', '{"a":{"b":1', "{'a': 1,}", 'noise {"a": 1}'):
        parsed, error, preview = parse_json_text(text)
        assert parsed is not None
        assert preview


def test_jsonutil_rejects_garbage():
    parsed, error, _ = parse_json_text("not json")
    assert parsed is None


def test_jsonutil_builtin_fallback_without_json_repair():
    with patch.dict(sys.modules, {"json_repair": None}):
        assert parse_json_text('{"a": "unterm')[0] == {"a": "unterm"}
        assert parse_json_text('{"a":[1,2,3')[0] == {"a": [1, 2, 3]}
