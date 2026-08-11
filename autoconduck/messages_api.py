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
    None: "end_turn",
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


def coerce_content_text(content: Any) -> str:
    """Return plain text for the varied content shapes returned by providers."""
    return _text_from_blocks(content)


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
        content_parts: list[dict[str, Any]] = []
        tool_calls: list[dict] = []
        tool_messages: list[dict] = []
        has_image = False
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text = block.get("text", "")
                text_parts.append(text)
                content_parts.append({"type": "text", "text": text})
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
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id"),
                        "content": result_content,
                    }
                )
            elif btype == "image":
                source = block.get("source") or {}
                if source.get("type") == "base64" and source.get("data"):
                    url = f"data:{source.get('media_type', 'application/octet-stream')};base64,{source['data']}"
                elif source.get("type") == "url" and source.get("url"):
                    url = source["url"]
                else:
                    continue
                has_image = True
                content_parts.append({"type": "image_url", "image_url": {"url": url}})
            # Unknown block types are silently skipped.

        if tool_calls:
            entry: dict[str, Any] = {
                "role": role,
                "content": content_parts if has_image else "".join(text_parts) or None,
            }
            entry["tool_calls"] = tool_calls
            messages.append(entry)
        elif text_parts or has_image:
            messages.append({
                "role": role,
                "content": content_parts if has_image else "".join(text_parts),
            })
        messages.extend(tool_messages)

    return messages


def _clean_enum_value(val: Any) -> str:
    if isinstance(val, str):
        return val
    if isinstance(val, bool):
        return "true" if val else "false"
    if val is None:
        return "null"
    return str(val)


def sanitize_tools(tools: Any) -> Any:
    """Sanitize JSON Schema definitions inside tools for strict providers (e.g. Gemini).

    Gemini's Schema protobuf requires all ``enum`` entries to be strings.  Tools
    generated by agents (like Pi extensions) may contain booleans or integers
    inside ``enum`` arrays or ``anyOf`` blocks, causing 400 validation errors on
    strict API gateways.
    """
    if isinstance(tools, list):
        return [sanitize_tools(item) for item in tools]
    if not isinstance(tools, dict):
        return tools

    cleaned: dict[str, Any] = {}
    for key, value in tools.items():
        if key == "enum" and isinstance(value, list):
            cleaned[key] = [_clean_enum_value(v) for v in value]
        else:
            cleaned[key] = sanitize_tools(value)
    return cleaned


def openai_tools_from_anthropic(tools: Any) -> list[dict]:
    """Translate Anthropic tool definitions to OpenAI function tools."""
    if not tools:
        return []
    result: list[dict] = []
    for tool in tools:
        if not isinstance(tool, dict) or not tool.get("name"):
            continue
        function = {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {"type": "object"}),
        }
        result.append({"type": "function", "function": function})
    return sanitize_tools(result)



def openai_tool_choice_from_anthropic(choice: Any) -> Any:
    """Translate Anthropic tool_choice values accepted by OpenAI APIs."""
    if choice == "any":
        return "required"
    if not isinstance(choice, dict):
        return choice
    choice_type = choice.get("type")
    if choice_type == "auto":
        return "auto"
    if choice_type == "none":
        return "none"
    if choice_type == "any":
        return "required"
    if choice_type == "tool" and choice.get("name"):
        return {"type": "function", "function": {"name": choice["name"]}}
    return choice


def serve_model_ids(cfg) -> list[str]:
    """Return the sorted list of ids served on /v1/models."""
    custom_ids = [
        m["id"]
        for m in (getattr(cfg, "custom_models", None) or [])
        if m.get("enabled", True) and m.get("id")
    ]
    return sorted(PSEUDO_MODELS | set(custom_ids))


def custom_entry(cfg, model_id: str) -> dict | None:
    from .config import _configured_model_sources

    def matches(entry: dict) -> bool:
        candidate = entry.get("id") or entry.get("model_name") or entry.get("model")
        params = entry.get("litellm_params")
        if not candidate and isinstance(params, dict):
            candidate = params.get("model") or params.get("model_name")
        return bool(candidate) and str(candidate).removeprefix("openai/") == str(model_id).removeprefix("openai/")

    for entry in _configured_model_sources(cfg):
        if isinstance(entry, dict) and entry.get("enabled", True) is not False and matches(entry):
            return entry
    return None


def litellm_params_for(model_id: str, cfg) -> dict:
    from .config import provider_for

    entry = custom_entry(cfg, model_id)
    if not entry:
        return {"model": qualify_model(model_id)}

    params = (
        entry.get("litellm_params")
        if isinstance(entry.get("litellm_params"), dict)
        else entry
    )

    provider = params.get("provider") or entry.get("provider")
    base_url = (
        params.get("base_url")
        or params.get("api_base")
        or entry.get("base_url")
        or entry.get("api_base")
    )

    raw_model = (
        params.get("model")
        or params.get("id")
        or params.get("model_name")
        or entry.get("id")
        or model_id
    )
    if "/" in str(raw_model):
        qual_model = str(raw_model)
    elif provider:
        qual_model = f"{provider}/{model_id}"
    else:
        qual_model = qualify_model(model_id)

    result = {"model": qual_model}
    if base_url:
        result["api_base"] = normalize_api_base(base_url)

    api_key = resolve_api_key(params, provider_for(entry, cfg))
    if api_key:
        result["api_key"] = api_key

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


def anthropic_response_text(
    content: Any, model: str = "", stop_reason: str = "end_turn", input_text: str = ""
) -> dict:
    text = coerce_content_text(content)
    return {
        "id": "msg_" + uuid.uuid4().hex[:12],
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": count_tokens(input_text),
            "output_tokens": count_tokens(text),
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
        self.tool_indices: dict[int, int] = {}
        self.next_block_index = 0
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
                self.text_index = self.next_block_index
                self.next_block_index += 1
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
            self.text_index = self.next_block_index
            self.next_block_index += 1
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
            if idx not in self.tool_indices:
                self.tool_indices[idx] = self.next_block_index
                self.next_block_index += 1
            block_index = self.tool_indices[idx]
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
            if not self.blocks:
                self.text_index = self.next_block_index
                self.next_block_index += 1
                self.blocks[0] = {"kind": "text", "started": True}
                events.append({
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                })
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
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": usage_out},
            }
        )
        events.append({"type": "message_stop"})
        self.finished = True
        return events

    def finish(self) -> list[dict]:
        if self.finished:
            return []
        # Some providers send only a finish_reason chunk. Anthropic still
        # requires a content block in the message, even when it is empty.
        events = self._ensure_message_start()
        if not self.blocks:
            self.text_index = self.next_block_index
            self.next_block_index += 1
            self.blocks[0] = {"kind": "text", "started": True}
            events.append({
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            })
        events.extend(self._close("stop"))
        return events

    def error(self, message: str) -> list[dict]:
        return [{
            "event": "error",
            "data": json.dumps({
                "type": "error",
                "error": {"type": "api_error", "message": message},
            }),
        }]
