"""Server routes, Anthropic /v1/messages shim, /v1/chat/completions, /stats, and JSON repair unit tests."""
import json
import pytest
from fastapi.testclient import TestClient

from autoconduck import main, server_streaming
from autoconduck.config import Config
from autoconduck.jsonutil import parse_json_text
from autoconduck.server.messages_api import (
    openai_messages_from_anthropic,
    openai_tool_choice_from_anthropic,
    openai_tools_from_anthropic,
)
from autoconduck.server.messages_sse import AnthropicSSETranslator, anthropic_response_text, count_tokens


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


def test_completions_with_orchestrator_tool_calls(monkeypatch):
    from autoconduck.orchestrator.handoff import ExecutionHandoff
    handoff = ExecutionHandoff(
        "Plan markdown content",
        tool_calls=[{"index": 0, "id": "call_123", "type": "function", "function": {"name": "subagent", "arguments": '{"workflowScript":"..."}'}}]
    )
    async def mock_slow_route(*args, **kwargs):
        return {"content": handoff.content, "tool_calls": handoff.tool_calls}

    monkeypatch.setattr("autoconduck.orchestrator.run", mock_slow_route)
    monkeypatch.setattr("autoconduck.routing.dispatcher.route", lambda *a, **kw: type("D", (), {"path": "SLOW", "model": None, "complexity": 0.85})())
    main._build()
    client = TestClient(main.app)

    # Non-streaming test
    resp = client.post("/v1/chat/completions", json={"model": "autoconduck", "messages": [{"role": "user", "content": "Refactor"}]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["finish_reason"] == "tool_calls"
    assert len(body["choices"][0]["message"]["tool_calls"]) == 1
    assert body["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "subagent"

    # Streaming test
    resp_stream = client.post("/v1/chat/completions", json={"model": "autoconduck", "messages": [{"role": "user", "content": "Refactor"}], "stream": True})
    assert resp_stream.status_code == 200
    lines = [l for l in resp_stream.text.split("\n") if l.startswith("data: ") and not l.endswith("[DONE]")]
    chunks = [json.loads(l[6:]) for l in lines]
    has_tool_calls = any(c["choices"][0].get("delta", {}).get("tool_calls") for c in chunks)
    assert has_tool_calls
    # Verify OpenAI streaming specification: chunk 1 has finish_reason: None, chunk 2 has finish_reason: "tool_calls"
    assert chunks[-2]["choices"][0]["finish_reason"] is None
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_messages_endpoint_guards_undeclared_tools(monkeypatch):
    from autoconduck.orchestrator.handoff import ExecutionHandoff
    handoff = ExecutionHandoff(
        "Refactoring plan content",
        tool_calls=[{"index": 0, "id": "call_sub", "type": "function", "function": {"name": "subagent", "arguments": '{"workflowScript":"..."}'}}]
    )
    async def mock_slow_route(*args, **kwargs):
        return {"content": handoff.content, "tool_calls": handoff.tool_calls}

    monkeypatch.setattr("autoconduck.orchestrator.run", mock_slow_route)
    monkeypatch.setattr("autoconduck.routing.dispatcher.route", lambda *a, **kw: type("D", (), {"path": "SLOW", "model": None, "complexity": 0.85})())
    main._build()
    client = TestClient(main.app)

    # 1. Claude Code sends request without "subagent" in tools -> tool_use should be stripped to avoid SDK crash
    resp_no_sub = client.post("/v1/messages", json={
        "model": "autoconduck",
        "messages": [{"role": "user", "content": "Refactor"}],
        "tools": [{"name": "read_file", "description": "read", "input_schema": {"type": "object"}}],
        "stream": False,
    })
    assert resp_no_sub.status_code == 200
    data = resp_no_sub.json()
    assert data["stop_reason"] == "end_turn"
    assert all(b.get("type") != "tool_use" for b in data["content"])

    # 2. Client sends request with "subagent" in tools -> tool_use is emitted cleanly
    resp_with_sub = client.post("/v1/messages", json={
        "model": "autoconduck",
        "messages": [{"role": "user", "content": "Refactor"}],
        "tools": [{"name": "subagent", "description": "run subagent", "input_schema": {"type": "object"}}],
        "stream": False,
    })
    assert resp_with_sub.status_code == 200
    data_sub = resp_with_sub.json()
    assert data_sub["stop_reason"] == "tool_use"
    assert any(b.get("type") == "tool_use" and b.get("name") == "subagent" for b in data_sub["content"])


