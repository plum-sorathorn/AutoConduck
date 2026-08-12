from __future__ import annotations

import os
import types

import pytest

from autoconduck import main
from autoconduck import server_streaming
from autoconduck import messages_api as m


def test_openai_messages_conversion_with_system_and_blocks():
    body = {
        "system": [{"type": "text", "text": "You are helpful."}],
        "messages": [
            {"role": "user", "content": "Hello there"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Sure, "},
                    {"type": "text", "text": "here you go."},
                ],
            },
        ],
    }
    out = m.openai_messages_from_anthropic(body)
    assert out[0] == {"role": "system", "content": "You are helpful."}
    assert out[1] == {"role": "user", "content": "Hello there"}
    assert out[2]["role"] == "assistant"
    assert out[2]["content"] == "Sure, here you go."


def test_tool_use_and_tool_result_conversion():
    body = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "SF"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "sunny"},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"data": "..."}},
                    {"type": "text", "text": "what now?"},
                ],
            },
        ],
    }
    out = m.openai_messages_from_anthropic(body)
    assistant = out[0]
    assert assistant["role"] == "assistant"
    assert assistant["tool_calls"][0]["id"] == "toolu_1"
    assert assistant["tool_calls"][0]["function"]["name"] == "get_weather"
    import json as _json
    assert _json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"city": "SF"}

    tool_msg = out[1]
    assert tool_msg == {"role": "tool", "tool_call_id": "toolu_1", "content": "sunny"}

    last = out[2]
    assert last["role"] == "user"
    assert last["content"] == "what now?"


def _cfg(custom_models=None):
    return types.SimpleNamespace(custom_models=custom_models or [])


def test_serve_model_ids_includes_pseudo_and_enabled_custom():
    cfg = _cfg(
        custom_models=[
            {"id": "my-model", "enabled": True},
            {"id": "disabled-model", "enabled": False},
        ]
    )
    ids = m.serve_model_ids(cfg)
    assert "autoconduck" in ids
    assert "autoconduck-budget" in ids
    assert "autoconduck-expensive" in ids
    assert "my-model" in ids
    assert "disabled-model" not in ids
    assert ids == sorted(ids)


def test_custom_entry_and_litellm_params_use_base_url_and_env(monkeypatch):
    monkeypatch.setenv("MY_CUSTOM_KEY", "secret-token")
    cfg = _cfg(
        custom_models=[
            {
                "id": "my-model",
                "base_url": "https://example.com/v1",
                "api_key_env": "MY_CUSTOM_KEY",
                "enabled": True,
            }
        ]
    )
    entry = m.custom_entry(cfg, "my-model")
    assert entry is not None
    assert entry["base_url"] == "https://example.com/v1"

    params = m.litellm_params_for("my-model", cfg)
    assert params["model"] == "openai/my-model"
    assert params["api_base"] == "https://example.com/v1"
    assert params["api_key"] == "secret-token"

    default_params = m.litellm_params_for("autoconduck", cfg)
    assert default_params == {"model": "openai/autoconduck"}


def test_custom_entry_supports_pi_and_nested_litellm_params(monkeypatch):
    from autoconduck.config import Config, ModelEntry
    monkeypatch.setenv("PI_MODEL_KEY", "pi-secret")
    cfg = Config()
    cfg.pi.enabled = True
    cfg.pi.model_entries = [
        ModelEntry(
            id="gpt-5-6-luna",
            provider="openai",
            base_url="https://custom.luna-api.com/v1",
            api_key_env="PI_MODEL_KEY",
        )
    ]

    entry = m.custom_entry(cfg, "gpt-5-6-luna")
    assert entry is not None
    params = m.litellm_params_for("gpt-5-6-luna", cfg)
    assert params["model"] == "openai/gpt-5-6-luna"
    assert params["api_base"] == "https://custom.luna-api.com/v1"
    assert params["api_key"] == "pi-secret"


def test_litellm_params_for_custom_provider_and_api_base():
    from autoconduck.config import Config
    cfg = Config()
    cfg.custom_models = [
        {
            "id": "llama3-70b",
            "provider": "ollama",
            "api_base": "http://localhost:11434/v1",
            "enabled": True,
        }
    ]

    entry = m.custom_entry(cfg, "llama3-70b")
    assert entry is not None
    params = m.litellm_params_for("llama3-70b", cfg)
    assert params["model"] == "ollama/llama3-70b"
    assert params["api_base"] == "http://localhost:11434/v1"

def test_messages_kwargs_do_not_clobber_qualified_model():
    assert m.messages_litellm_kwargs("deepseek-v4-flash", {"model": "openai/deepseek-v4-flash"})["model"] == "openai/deepseek-v4-flash"


def test_openai_tools_from_anthropic():
    tools = [{"name": "do_thing", "description": "Do it", "input_schema": {"type": "object", "properties": {}}}]
    assert m.openai_tools_from_anthropic(tools) == [{
        "type": "function",
        "function": {"name": "do_thing", "description": "Do it", "parameters": tools[0]["input_schema"]},
    }]
    assert m.openai_tools_from_anthropic([]) == []
    assert m.openai_tools_from_anthropic(None) == []
    assert m.openai_tool_choice_from_anthropic({"type": "auto"}) == "auto"
    assert m.openai_tool_choice_from_anthropic({"type": "tool", "name": "do_thing"}) == {
        "type": "function", "function": {"name": "do_thing"}
    }


def test_openai_tool_choice_from_anthropic_all_variants():
    assert m.openai_tool_choice_from_anthropic(None) is None
    assert m.openai_tool_choice_from_anthropic("auto") == "auto"
    assert m.openai_tool_choice_from_anthropic("none") == "none"
    assert m.openai_tool_choice_from_anthropic("any") == "required"
    assert m.openai_tool_choice_from_anthropic({"type": "auto"}) == "auto"
    assert m.openai_tool_choice_from_anthropic({"type": "none"}) == "none"
    assert m.openai_tool_choice_from_anthropic({"type": "any"}) == "required"
    assert m.openai_tool_choice_from_anthropic({"type": "tool", "name": "X"}) == {
        "type": "function", "function": {"name": "X"}
    }


def test_translator_text_stream_events():
    translator = m.AnthropicSSETranslator("autoconduck")
    chunk1 = {"choices": [{"delta": {"role": "assistant", "content": "Hel"}, "finish_reason": None}]}
    chunk2 = {"choices": [{"delta": {"content": "lo!"}, "finish_reason": None}]}
    chunk3 = {"choices": [{"delta": {}, "finish_reason": "stop"}]}

    events1 = translator.translate(chunk1)
    events2 = translator.translate(chunk2)
    events3 = translator.translate(chunk3)

    all_events = events1 + events2 + events3
    types_seen = [e["type"] for e in all_events]

    assert types_seen[0] == "message_start"
    assert "content_block_start" in types_seen
    assert types_seen.count("content_block_delta") == 2
    assert "content_block_stop" in types_seen
    assert types_seen[-1] == "message_stop"

    message_delta = next(e for e in all_events if e["type"] == "message_delta")
    assert message_delta["delta"]["stop_reason"] == "end_turn"

    # content_block_stop must appear before message_delta/message_stop.
    stop_idx = types_seen.index("content_block_stop")
    delta_idx = types_seen.index("message_delta")
    assert stop_idx < delta_idx


def test_translator_tool_calls_events():
    translator = m.AnthropicSSETranslator("autoconduck")
    chunk1 = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {"index": 0, "id": "call_1", "function": {"name": "get_weather", "arguments": ""}}
                    ]
                },
                "finish_reason": None,
            }
        ]
    }
    chunk2 = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {"index": 0, "function": {"arguments": '{"city": "SF"}'}}
                    ]
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    events = translator.translate(chunk1) + translator.translate(chunk2)
    types_seen = [e["type"] for e in events]

    assert "content_block_start" in types_seen
    tool_start = next(e for e in events if e["type"] == "content_block_start" and e["content_block"]["type"] == "tool_use")
    assert tool_start["index"] == 0
    assert tool_start["content_block"]["name"] == "get_weather"

    input_delta = next(e for e in events if e["type"] == "content_block_delta" and e["delta"]["type"] == "input_json_delta")
    assert input_delta["delta"]["partial_json"] == '{"city": "SF"}'

    message_delta = next(e for e in events if e["type"] == "message_delta")
    assert message_delta["delta"]["stop_reason"] == "tool_use"
    assert types_seen[-1] == "message_stop"


def test_translator_text_then_tool_uses_sequential_indices():
    translator = m.AnthropicSSETranslator("autoconduck")
    events = translator.translate({"choices": [{"delta": {"content": "hi"}, "finish_reason": None}]})
    events += translator.translate({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call", "function": {"name": "x", "arguments": "{}"}}]}, "finish_reason": "tool_calls"}]})
    starts = [e for e in events if e["type"] == "content_block_start"]
    assert [e["index"] for e in starts] == [0, 1]


def test_translator_finish_idempotent():
    translator = m.AnthropicSSETranslator("autoconduck")
    translator.translate({"choices": [{"delta": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}]})
    assert translator.finish() == []


def test_count_tokens_estimate():
    assert m.count_tokens("") >= 0
    assert m.count_tokens("hello world") > 0


def test_anthropic_response_coerces_list_and_none_content():
    response = m.anthropic_response_text([
        {"type": "text", "text": "hello"},
        {"type": "image", "source": {}},
        " world",
    ])
    assert response["content"] == [{"type": "text", "text": "hello world"}]
    assert m.anthropic_response_text(None)["content"] == [{"type": "text", "text": ""}]


def test_translator_finish_creates_empty_text_block():
    translator = m.AnthropicSSETranslator("autoconduck")
    events = translator.translate({"choices": [{"delta": {}, "finish_reason": "stop"}]})
    event_types = [event["type"] for event in events]
    assert "content_block_start" in event_types
    assert event_types[-2:] == ["message_delta", "message_stop"]


@pytest.mark.asyncio
async def test_chat_stream_litellm_missing_emits_error_and_done(monkeypatch):
    async def route_target(_model, _messages):
        return "test-model", {"model": "openai/test-model"}

    class Request:
        async def is_disconnected(self):
            return False

    monkeypatch.setattr(server_streaming, "_route_target", route_target)
    monkeypatch.setattr(server_streaming, "_litellm", lambda: None)
    body = main.CompletionRequest(model="autoconduck", messages=[], stream=True)
    response = await server_streaming._cached["completions"](body, Request())
    output = "".join([chunk async for chunk in response.body_iterator])
    assert '"type": "api_error"' in output
    assert "data: [DONE]" in output


@pytest.mark.asyncio
async def test_messages_stream_first_await_failure_returns_502(monkeypatch):
    class FailingLLM:
        async def acompletion(self, **_kwargs):
            raise RuntimeError("upstream failed")

    async def route_target(_model, _messages):
        return "test-model", {"model": "openai/test-model"}

    class Request:
        async def is_disconnected(self):
            return False

    monkeypatch.setattr(server_streaming, "_route_target", route_target)
    monkeypatch.setattr(server_streaming, "_litellm", lambda: FailingLLM())

    body = main.MessagesRequest(
        model="autoconduck",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
    )
    response = await server_streaming._cached["messages_endpoint"](body, Request())
    assert response.status_code == 502
    assert response.body == b'{"type":"error","error":{"type":"api_error","message":"upstream failed"}}'


@pytest.mark.asyncio
async def test_messages_thinking_enables_litellm_drop_params(monkeypatch):
    calls = []

    class Result:
        choices = [types.SimpleNamespace(message=types.SimpleNamespace(content="Hello"))]

    class LLM:
        async def acompletion(self, **kwargs):
            calls.append(kwargs)
            return Result()

    async def route_target(_model, _messages):
        return "deepseek", {"model": "openai/deepseek"}

    monkeypatch.setattr(server_streaming, "_route_target", route_target)
    monkeypatch.setattr(server_streaming, "_litellm", lambda: LLM())

    body = main.MessagesRequest(
        model="autoconduck",
        messages=[{"role": "user", "content": "Hello"}],
        thinking={"type": "enabled", "budget_tokens": 1024},
        stream=False,
    )
    response = await server_streaming._cached["messages_endpoint"](body, object())

    assert response.status_code == 200
    assert calls[0]["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert calls[0]["drop_params"] is True


def test_sanitize_tools_converts_non_string_enums():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "pi_tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "enabled": {
                            "anyOf": [
                                {"type": "string"},
                                {"enum": [False, True]},
                            ]
                        },
                        "count": {
                            "enum": [1, 2, 3]
                        },
                        "null_option": {
                            "enum": [None]
                        }
                    },
                },
            },
        }
    ]
    sanitized = m.sanitize_tools(tools)
    props = sanitized[0]["function"]["parameters"]["properties"]
    assert props["enabled"]["anyOf"][1]["enum"] == ["false", "true"]
    assert props["count"]["enum"] == ["1", "2", "3"]
    assert props["null_option"]["enum"] == ["null"]


def test_opencodego_litellm_params_avoids_provider_prefix(monkeypatch):
    monkeypatch.setenv("OPENCODE_API_KEY", "opencode-key-123")
    monkeypatch.setattr(m, "resolve_api_key", lambda *_args: "opencode-key-123")
    cfg = _cfg(
        custom_models=[
            {
                "id": "gpt-5.6-luna",
                "provider": "opencodego",
                "base_url": "https://opencode.ai/zen/go/v1",
                "api_key": "opencode-key-123",
                "enabled": True,
            }
        ]
    )
    params = m.litellm_params_for("gpt-5.6-luna", cfg)
    assert params["model"] == "openai/gpt-5.6-luna"
    assert params["api_base"] == "https://opencode.ai/zen/go/v1"
    assert params["api_key"] == "opencode-key-123"


