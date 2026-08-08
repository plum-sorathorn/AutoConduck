from __future__ import annotations

import os
import types

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
    assert params["model"] == "my-model"
    assert params["api_base"] == "https://example.com/v1"
    assert params["api_key"] == "secret-token"

    default_params = m.litellm_params_for("autoconduck", cfg)
    assert default_params == {"model": "autoconduck"}


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
    assert tool_start["content_block"]["name"] == "get_weather"

    input_delta = next(e for e in events if e["type"] == "content_block_delta" and e["delta"]["type"] == "input_json_delta")
    assert input_delta["delta"]["partial_json"] == '{"city": "SF"}'

    message_delta = next(e for e in events if e["type"] == "message_delta")
    assert message_delta["delta"]["stop_reason"] == "tool_use"
    assert types_seen[-1] == "message_stop"


def test_translator_finish_idempotent():
    translator = m.AnthropicSSETranslator("autoconduck")
    translator.translate({"choices": [{"delta": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}]})
    assert translator.finish() == []


def test_count_tokens_estimate():
    assert m.count_tokens("") >= 0
    assert m.count_tokens("hello world") > 0
