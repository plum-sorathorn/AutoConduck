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

from .messages_models import PSEUDO_MODELS

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


def normalize_messages_for_llm(messages: list) -> list[dict]:
    """Ensure all assistant messages have a reasoning_content key for thinking-mode API providers (e.g. Console Go)."""
    if not isinstance(messages, list):
        return []
    normalized: list[dict] = []
    for msg in messages:
        if isinstance(msg, dict):
            m = dict(msg)
        elif hasattr(msg, "model_dump"):
            m = msg.model_dump()
        else:
            m = {"role": getattr(msg, "role", "user"), "content": str(msg)}
        if m.get("role") == "assistant":
            if m.get("reasoning_content") is None:
                m["reasoning_content"] = m.get("thinking") or m.get("reasoning") or ""
        normalized.append(m)
    return normalized


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
            entry: dict[str, Any] = {"role": role, "content": content}
            if role == "assistant":
                entry["reasoning_content"] = msg.get("reasoning_content") or msg.get("thinking") or ""
            messages.append(entry)
            continue
        if not isinstance(content, list):
            entry = {"role": role, "content": ""}
            if role == "assistant":
                entry["reasoning_content"] = msg.get("reasoning_content") or msg.get("thinking") or ""
            messages.append(entry)
            continue

        text_parts: list[str] = []
        content_parts: list[dict[str, Any]] = []
        tool_calls: list[dict] = []
        tool_messages: list[dict] = []
        thinking_parts: list[str] = []
        has_image = False
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text = block.get("text", "")
                text_parts.append(text)
                content_parts.append({"type": "text", "text": text})
            elif btype == "thinking":
                thinking_text = block.get("thinking", "")
                if thinking_text:
                    thinking_parts.append(thinking_text)
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

        thinking_str = "".join(thinking_parts) if thinking_parts else (msg.get("reasoning_content") or msg.get("thinking") or "")
        if tool_calls:
            entry = {
                "role": role,
                "content": content_parts if has_image else "".join(text_parts) or None,
            }
            entry["tool_calls"] = tool_calls
            if role == "assistant":
                entry["reasoning_content"] = thinking_str
            messages.append(entry)
        elif text_parts or has_image or thinking_parts:
            entry = {
                "role": role,
                "content": content_parts if has_image else "".join(text_parts),
            }
            if role == "assistant":
                entry["reasoning_content"] = thinking_str
            messages.append(entry)
        messages.extend(tool_messages)

    return normalize_messages_for_llm(messages)


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
        is_known_provider = False
        try:
            from litellm import provider_list

            if provider in provider_list:
                is_known_provider = True
        except Exception:
            pass
        if is_known_provider:
            qual_model = f"{provider}/{model_id}"
        else:
            qual_model = str(model_id)
    else:
        qual_model = str(model_id)

    qual_model = qualify_model(qual_model)

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



from .messages_sse import AnthropicSSETranslator, anthropic_response_text, count_tokens
