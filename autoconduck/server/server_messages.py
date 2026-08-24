"""Anthropic /v1/messages translation endpoint and thinking delta streaming."""

from __future__ import annotations

import json
import uuid
from typing import Any

import autoconduck.config as config_module
from autoconduck.server.server_models import MessagesRequest


async def handle_messages(
    body: MessagesRequest,
    request: Any,
    *,
    route_target_fn: Any,
    openai_messages_from_anthropic: Any,
    openai_tools_from_anthropic: Any,
    openai_tool_choice_from_anthropic: Any,
    count_tokens: Any,
    AnthropicSSETranslator: Any,
    anthropic_response_text: Any,
    coerce_content_text: Any,
    messages_litellm_kwargs: Any,
    normalize_messages_for_llm: Any,
    StreamingResponse: Any,
    JSONResponse: Any,
) -> Any:
    """Translate Anthropic /v1/messages request to OpenAI format with thinking deltas."""
    try:
        oai_messages = normalize_messages_for_llm(
            openai_messages_from_anthropic(body.model_dump(exclude_none=True))
        )
        try:
            from autoconduck.orchestrator.session_guard import SessionGuard

            oai_messages = SessionGuard().guard_context(oai_messages).messages
        except Exception:
            pass
    except Exception as exc:
        return JSONResponse(
            {
                "type": "error",
                "error": {"type": "invalid_request_error", "message": str(exc)},
            },
            status_code=400,
        )
    try:
        target, extra = await route_target_fn(
            body.model, oai_messages, request, client_type="claude"
        )
    except Exception as exc:
        return JSONResponse(
            {"type": "error", "error": {"type": "api_error", "message": str(exc)}},
            status_code=500,
        )
    answer = extra.get("__answer__")
    if answer is not None:
        content = (
            answer.get("content", "") if isinstance(answer, dict) else str(answer)
        )
        tool_calls = (
            answer.get("tool_calls")
            if isinstance(answer, dict)
            else getattr(answer, "tool_calls", None)
        )
        if tool_calls:
            requested_tool_names = {
                (t.get("name") if isinstance(t, dict) else getattr(t, "name", None))
                for t in (body.tools or [])
            }
            valid_tool_calls = [
                tc
                for tc in tool_calls
                if ((tc.get("function") or {}).get("name") in requested_tool_names)
            ]
            tool_calls = valid_tool_calls or None

        if body.stream:

            async def answer_stream():
                translator = AnthropicSSETranslator(
                    target or body.model, input_text=json.dumps(oai_messages)
                )
                for ev in translator._ensure_message_start():
                    yield f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n"
                delta: dict[str, Any] = {"role": "assistant", "content": content}
                if tool_calls:
                    delta["tool_calls"] = tool_calls
                for ev in translator.translate(
                    {
                        "choices": [
                            {
                                "delta": delta,
                                "finish_reason": "tool_calls"
                                if tool_calls
                                else "stop",
                            }
                        ]
                    }
                ):
                    yield f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n"

            return StreamingResponse(
                answer_stream(), media_type="text/event-stream"
            )
        if not tool_calls:
            return JSONResponse(
                anthropic_response_text(
                    content,
                    target or body.model,
                    input_text=json.dumps(oai_messages),
                )
            )

        content_blocks: list[dict[str, Any]] = []
        if content:
            content_blocks.append({"type": "text", "text": content})
        for tc in tool_calls:
            fn = tc.get("function") or {}
            raw_args = fn.get("arguments", "{}")
            parsed_args = (
                json.loads(raw_args)
                if isinstance(raw_args, str)
                else (raw_args or {})
            )
            content_blocks.append(
                {
                    "type": "tool_use",
                    "id": tc.get("id") or ("toolu_" + uuid.uuid4().hex[:12]),
                    "name": fn.get("name"),
                    "input": parsed_args,
                }
            )
        return JSONResponse(
            {
                "id": "msg_" + uuid.uuid4().hex[:12],
                "type": "message",
                "role": "assistant",
                "content": content_blocks,
                "model": target or body.model,
                "stop_reason": "tool_use",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": count_tokens(json.dumps(oai_messages)),
                    "output_tokens": count_tokens(content)
                    + count_tokens(json.dumps(content_blocks)),
                },
            }
        )
    if extra.get("_path") == "FAST":
        from autoconduck.digest import maybe_digest_messages

        digest = await maybe_digest_messages(
            oai_messages, config_module.get_config(), request=request
        )
        if digest:
            oai_messages = oai_messages + digest
    kwargs = messages_litellm_kwargs(target, extra)
    kwargs.update(
        _path=extra.get("_path", "unknown"),
        _pseudo=extra.get("_pseudo", body.model),
    )
    for name, value in (
        ("tools", openai_tools_from_anthropic(body.tools)),
        ("tool_choice", openai_tool_choice_from_anthropic(body.tool_choice)),
        ("max_tokens", body.max_tokens),
        ("stop", body.stop_sequences),
        ("temperature", body.temperature),
        ("top_p", body.top_p),
        ("thinking", body.thinking),
        ("metadata", body.metadata),
        ("cache_control", body.cache_control),
    ):
        if value is not None:
            kwargs[name] = value
    if extra.get("_plan_context"):
        plan_ctx = extra.pop("_plan_context")
        oai_messages = list(oai_messages) + [{
            "role": "user",
            "content": f"[AutoConduck Task Plan & Context]\n{plan_ctx}\n\nExecute the above plan immediately using your available tools.",
        }]
    from autoconduck.server.server_streaming import _litellm

    llm = _litellm()
    if llm is None:
        return JSONResponse(
            {
                "type": "error",
                "error": {"type": "api_error", "message": "litellm unavailable"},
            },
            status_code=502 if body.stream else 503,
        )
    if body.stream:
        try:
            response = await llm.acompletion(
                messages=oai_messages, stream=True, drop_params=True, **kwargs
            )
        except Exception as exc:
            from autoconduck.routing.pricing import record_error
            record_error(target)
            return JSONResponse(
                {
                    "type": "error",
                    "error": {"type": "api_error", "message": str(exc)},
                },
                status_code=502,
            )

        async def relay():
            translator = AnthropicSSETranslator(
                target, input_text=json.dumps(oai_messages)
            )
            try:
                for ev in translator._ensure_message_start():
                    yield f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n"
                async for chunk in response:
                    if await request.is_disconnected():
                        break
                    payload = (
                        chunk.model_dump()
                        if hasattr(chunk, "model_dump")
                        else chunk
                    )
                    for ev in translator.translate(payload):
                        yield f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n"
                for ev in translator.finish():
                    yield f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n"
            except Exception as exc:
                from autoconduck.routing.pricing import record_error
                record_error(target)
                exc_str = str(exc)
                if (
                    "Error building chunks" in exc_str
                    or "stream_chunk_builder" in exc_str
                    or "list index out of range" in exc_str
                ):
                    for ev in translator.finish():
                        yield f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n"
                else:
                    for ev in translator.error(str(exc)):
                        yield f"event: {ev['event']}\ndata: {ev['data']}\n\n"

        return StreamingResponse(relay(), media_type="text/event-stream")
    try:
        result = await llm.acompletion(
            messages=oai_messages, stream=False, drop_params=True, **kwargs
        )
        text = (
            result.choices[0].message.content
            if hasattr(result, "choices")
            else None
        )
    except Exception as exc:
        from autoconduck.routing.pricing import record_error
        record_error(target)
        return JSONResponse(
            {"type": "error", "error": {"type": "api_error", "message": str(exc)}},
            status_code=502,
        )
    return JSONResponse(
        anthropic_response_text(
            coerce_content_text(text), target, input_text=json.dumps(oai_messages)
        )
    )


async def handle_messages_count_tokens(
    request: Any,
    *,
    openai_messages_from_anthropic: Any,
    count_tokens: Any,
) -> dict[str, int]:
    """Count token input size for Anthropic client compatibility."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    return {
        "input_tokens": count_tokens(
            json.dumps(openai_messages_from_anthropic(body))
        )
    }
