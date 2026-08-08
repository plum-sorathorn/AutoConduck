"""Anthropic Messages API compatibility helpers (pure, no network I/O).

This module converts between the Anthropic Messages API wire format and the
OpenAI chat-completions format used internally by litellm, and translates
OpenAI-style streaming chunks into Anthropic SSE events. Nothing here touches
the network or imports fastapi/litellm, so it stays importable and testable
in isolation.
"""
from __future__ import annotations

import json
import os
import uuid
from typing import Any
from .config import normalize_api_base, qualify_model, resolve_api_key

PSEUDO_MODELS = {"autoconduck", "autoconduck-budget", "autoconduck-expensive"}

STOP_REASON_MAP: dict[Any, Any] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    None: None,
}


def _text_from_blocks(blocks: Any) -> str:
    if isinstance(blocks, str):
        return blocks
    if isinstance(blocks, list):
        parts: list[str] = []
        for b in blocks:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(b.get("text", ""))
            elif isinstance(b, str):
                parts.append(b)
        return "".join(parts)
    return ""


def openai_messages_from_anthropic(body: dict) -> list[dict]:
    """Convert an Anthropic /v1/messages request body into OpenAI messages."""
    messages: list[dict] = []

    system = body.get("system")
    if system:
        messages.append({"role": "system", "content": _text_from_blocks(system)})

    for msg in body.get("messages", []) or []:
        role = msg.get("role", "user")
        content = msg.get("content")

        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            messages.append({"role": role, "content": ""})
            continue

        text_parts: list[str] = []
        tool_calls: list[dict] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "tool_use":
                tool_calls.append(
                    {
                        "id": block.get("id"),
                        "type": "function",
                        "function": {
                            "name": block.get("name"),
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    }
                )
            elif btype == "tool_result":
                result_content = block.get("content")
                if isinstance(result_content, list):
                    result_content = _text_from_blocks(result_content)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id"),
                        "content": result_content,
                    }
                )
            # image / unknown block types are silently skipped.

        if tool_calls:
            entry: dict[str, Any] = {"role": role, "content": "".join(text_parts) or None}
            entry["tool_calls"] = tool_calls
            messages.append(entry)
        elif text_parts:
            messages.append({"role": role, "content": "".join(text_parts)})

    return messages


def serve_model_ids(cfg) -> list[str]:
    """Return the sorted list of ids served on /v1/models."""
    custom_ids = [
        m["id"]
        for m in (getattr(cfg, "custom_models", None) or [])
        if m.get("enabled", True) and m.get("id")
    ]
    return sorted(PSEUDO_MODELS | set(custom_ids))


def custom_entry(cfg, model_id: str) -> dict | None:
    for m in getattr(cfg, "custom_models", None) or []:
        if m.get("id") == model_id:
            return m
    return None


def litellm_params_for(model_id: str, cfg) -> dict:
    entry = custom_entry(cfg, model_id)
    if entry and entry.get("base_url"):
        return {
            "model": qualify_model(model_id),
            "api_base": normalize_api_base(entry["base_url"]),
            "api_key": resolve_api_key(entry),
        }
    result = {"model": qualify_model(model_id)}
    if entry and entry.get("api_key_env"):
        result["api_key"] = resolve_api_key(entry)
    return result

def messages_litellm_kwargs(model_id: str, extra: dict | None = None) -> dict:
    from .config import qualify_model
    kwargs = dict(extra or {})
    kwargs["model"] = qualify_model(model_id)
    return kwargs


def count_tokens(text: str) -> int:
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text or ""))
    except Exception:
        return max(1, len(text or "") // 4)


def anthropic_response_text(content: str, model: str, stop_reason: str = "end_turn") -> dict:
    return {
        "id": "msg_" + uuid.uuid4().hex[:12],
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": content}],
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": count_tokens(content),
            "output_tokens": count_tokens(content),
        },
    }


class AnthropicSSETranslator:
    """Stateful translator: OpenAI stream chunks -> Anthropic SSE event dicts."""

    def __init__(self, model: str, input_text: str = ""):
        self.model = model
        self.message_id = "msg_" + uuid.uuid4().hex[:12]
        self.input_tokens = count_tokens(input_text) if input_text else 0
        self.started = False
        self.blocks: dict[int, dict] = {}
        self.text_index: int | None = None
        self.finished = False
        self.output_tokens = 0

    def _ensure_message_start(self) -> list[dict]:
        if self.started:
            return []
        self.started = True
        return [
            {
                "type": "message_start",
                "message": {
                    "id": self.message_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": self.model,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": self.input_tokens, "output_tokens": 0},
                },
            }
        ]

    def translate(self, chunk: dict) -> list[dict]:
        events: list[dict] = []
        events.extend(self._ensure_message_start())

        choices = chunk.get("choices") or []
        if not choices:
            return events
        choice = choices[0]
        delta = choice.get("delta") or {}

        if delta.get("content"):
            if self.text_index is None:
                self.text_index = 0
                self.blocks[self.text_index] = {"kind": "text", "started": True}
                events.append(
                    {
                        "type": "content_block_start",
                        "index": self.text_index,
                        "content_block": {"type": "text", "text": ""},
                    }
                )
            self.output_tokens += count_tokens(delta["content"])
            events.append(
                {
                    "type": "content_block_delta",
                    "index": self.text_index,
                    "delta": {"type": "text_delta", "text": delta["content"]},
                }
            )
        elif delta.get("role") == "assistant" and self.text_index is None and not delta.get("tool_calls"):
            self.text_index = 0
            self.blocks[self.text_index] = {"kind": "text", "started": True}
            events.append(
                {
                    "type": "content_block_start",
                    "index": self.text_index,
                    "content_block": {"type": "text", "text": ""},
                }
            )

        for tc in delta.get("tool_calls") or []:
            idx = tc.get("index", 0)
            block_index = idx + 1  # offset so the text block (0) never collides
            fn = tc.get("function") or {}
            if block_index not in self.blocks:
                self.blocks[block_index] = {"kind": "tool_use", "started": True}
                events.append(
                    {
                        "type": "content_block_start",
                        "index": block_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": tc.get("id") or "toolu_" + uuid.uuid4().hex[:12],
                            "name": fn.get("name"),
                            "input": {},
                        },
                    }
                )
            if fn.get("arguments"):
                self.output_tokens += count_tokens(fn["arguments"])
                events.append(
                    {
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {"type": "input_json_delta", "partial_json": fn["arguments"]},
                    }
                )

        finish_reason = choice.get("finish_reason")
        if finish_reason:
            events.extend(self._close(finish_reason, chunk))

        return events

    def _close(self, finish_reason: Any, chunk: dict | None = None) -> list[dict]:
        events: list[dict] = []
        for idx in sorted(self.blocks.keys()):
            block = self.blocks[idx]
            if block.get("started") and not block.get("stopped"):
                block["stopped"] = True
                events.append({"type": "content_block_stop", "index": idx})
        stop_reason = STOP_REASON_MAP.get(finish_reason, finish_reason)
        usage_out = self.output_tokens
        if chunk is not None:
            usage = chunk.get("usage")
            if usage and usage.get("completion_tokens") is not None:
                usage_out = usage["completion_tokens"]
        events.append(
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason},
                "usage": {"output_tokens": usage_out},
            }
        )
        events.append({"type": "message_stop"})
        self.finished = True
        return events

    def finish(self) -> list[dict]:
        if self.finished:
            return []
        return self._close(None)
