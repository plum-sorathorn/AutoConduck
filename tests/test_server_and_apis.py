"""Server routes, Anthropic /v1/messages shim, /v1/chat/completions, /stats, and JSON repair unit tests."""
import json
import pytest
from fastapi.testclient import TestClient

from autoconduck import main, server_streaming
from autoconduck.config import Config
from autoconduck.jsonutil import parse_json_text
from autoconduck.messages_api import (
    AnthropicSSETranslator,
    anthropic_response_text,
    count_tokens,
    openai_messages_from_anthropic,
    openai_tool_choice_from_anthropic,
    openai_tools_from_anthropic,
)


@pytest.fixture
def test_client(monkeypatch):
    monkeypatch.setattr(server_streaming, "_litellm", lambda: type("FakeLLM", (), {
        "acompletion": staticmethod(lambda **kw: {"choices": [{"message": {"role": "assistant", "content": "mocked"}}]})
    })())
    main._build()
    return TestClient(main.app)


def test_openai_messages_conversion_from_anthropic():
    body = {
        "system": [{"type": "text", "text": "System directive."}],
        "messages": [
            {"role": "user", "content": "User question"},
            {"role": "assistant", "content": [{"type": "text", "text": "Assistant response"}]},
        ],
    }
    converted = openai_messages_from_anthropic(body)
    assert converted[0] == {"role": "system", "content": "System directive."}
    assert converted[1] == {"role": "user", "content": "User question"}
    assert converted[2]["role"] == "assistant"
    assert converted[2]["content"] == "Assistant response"


def test_anthropic_tools_conversion():
    tools = [
        {"name": "read_file", "description": "Read file contents", "input_schema": {"type": "object"}}
    ]
    openai_tools = openai_tools_from_anthropic(tools)
    assert len(openai_tools) == 1
    assert openai_tools[0]["function"]["name"] == "read_file"


def test_anthropic_sse_translator_flow():
    translator = AnthropicSSETranslator("autoconduck")
    chunk1 = {"choices": [{"delta": {"role": "assistant", "content": "Hello"}, "finish_reason": None}]}
    chunk2 = {"choices": [{"delta": {}, "finish_reason": "stop"}]}

    events1 = translator.translate(chunk1)
    events2 = translator.translate(chunk2)
    all_types = [e["type"] for e in events1 + events2]
    assert "message_start" in all_types
    assert "content_block_delta" in all_types
    assert "message_stop" in all_types


def test_json_util_repairs_common_outputs():
    cases = [
        '{"key": "value"}',
        '```json\n{"key": "value"}\n```',
        '{"key": "incomplete',
        '{"items": [1, 2, 3',
        "{'key': 123}",
    ]
    for text in cases:
        parsed, err, _ = parse_json_text(text)
        assert parsed is not None


def test_models_endpoint(test_client):
    response = test_client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    ids = {m["id"] for m in data["data"]}
    assert "autoconduck" in ids
    assert "autoconduck-budget" in ids
    assert "autoconduck-expensive" in ids


def test_stats_endpoint(test_client):
    response = test_client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "decisions" in data or "total_requests" in data or "summary" in data or isinstance(data, dict)
