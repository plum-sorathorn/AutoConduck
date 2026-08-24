"""OpenAI-compatible /v1/chat/completions endpoint handler and SSE streaming."""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

import autoconduck.config as config_module
from autoconduck.server.server_models import CompletionRequest
from autoconduck.stats import record


async def handle_chat_completions(
    body: CompletionRequest,
    request: Any,
    *,
    PSEUDO_MODELS: set[str],
    route_target_fn: Any,
    call_litellm_fn: Any,
    sanitize_tools: Any,
    normalize_messages_for_llm: Any,
    StreamingResponse: Any,
    JSONResponse: Any,
) -> Any:
    """Handle OpenAI /v1/chat/completions request with reasoning streams and compaction."""
    try:
        from autoconduck.orchestrator.session_guard import SessionGuard

        body.messages = SessionGuard().guard_context(body.messages).messages
    except Exception:
        pass
    body.messages = normalize_messages_for_llm(body.messages)
    cfg = config_module.get_config()
    configured_progress = bool(
        getattr(getattr(cfg, "selection", None), "slow_stream_progress", True)
    )
    env_progress = os.environ.get("AUTOCONDUCK_STREAM_PROGRESS")
    if env_progress is None:
        progress_setting = configured_progress
    else:
        normalized_progress = env_progress.strip().lower()
        if normalized_progress in {"1", "true", "yes"}:
            progress_setting = True
        elif normalized_progress in {"0", "false", "no"}:
            progress_setting = False
        else:
            progress_setting = configured_progress
    progress_enabled = (
        body.stream and progress_setting and body.model in PSEUDO_MODELS
    )

    if progress_enabled:

        async def progress_stream():
            progress_q: asyncio.Queue = asyncio.Queue()
            first_event = asyncio.get_running_loop().create_future()

            def on_progress(event):
                try:
                    progress_q.put_nowait(event)
                    if not first_event.done():
                        first_event.set_result(event)
                except Exception:
                    pass

            task = asyncio.create_task(
                route_target_fn(body.model, body.messages, request, on_progress)
            )
            try:
                first = await first_event
                first_path = (
                    getattr(first, "path", None)
                    or (first.get("path") if isinstance(first, dict) else "")
                    or ""
                )
                is_slow = str(first_path).upper() == "SLOW"
                target, extra = (await task) if not is_slow else (None, None)
                if is_slow:
                    labels = {
                        "recon": "recon",
                        "recon_subagent_pool": "reading files",
                        "planner": "planner",
                        "subagent_pool": "subagents",
                        "compactor": "compactor",
                        "executor": "executor",
                    }
                    created = int(time.time())
                    sent_role = False
                    while True:
                        if task.done() and progress_q.empty():
                            break
                        try:
                            event = await asyncio.wait_for(
                                progress_q.get(), timeout=0.1
                            )
                        except asyncio.TimeoutError:
                            if task.done() and progress_q.empty():
                                break
                            continue
                        delta_text = None
                        if isinstance(event, str):
                            delta_text, node = (
                                event if event.endswith("\n") else f"{event}\n"
                            ), "progress"
                        elif isinstance(event, dict):
                            if event.get("kind") == "route":
                                continue
                            node = event.get("node", "progress")
                            detail = event.get("step_detail") or node
                            label = labels.get(node, node)
                            delta_text = f"[{label}] {detail}\n"
                        else:
                            node = getattr(
                                event, "name", getattr(event, "node", "progress")
                            )
                            detail = getattr(
                                event,
                                "detail",
                                getattr(event, "step_detail", str(event)),
                            )
                            delta_text = f"[{node}] {detail}\n"
                        if delta_text is None:
                            continue
                        delta: dict[str, Any] = {"reasoning_content": delta_text}
                        if not sent_role:
                            delta["role"] = "assistant"
                            sent_role = True
                        yield (
                            "data: "
                            + json.dumps(
                                {
                                    "id": "autoconduck",
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": body.model,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": delta,
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                            )
                            + "\n\n"
                        )
                        if node == "idle":
                            break
                        if task.done() and progress_q.empty():
                            break
                    target, extra = await task
                answer = extra.get("__answer__") if extra else None
                if answer is not None:
                    content = (
                        answer.get("content", "")
                        if isinstance(answer, dict)
                        else str(answer)
                    )
                    tool_calls = (
                        answer.get("tool_calls")
                        if isinstance(answer, dict)
                        else getattr(answer, "tool_calls", None)
                    )
                    finish_reason = "tool_calls" if tool_calls else "stop"
                    created = int(time.time())
                    delta = {"role": "assistant"}
                    if content:
                        delta["content"] = content
                    if tool_calls:
                        delta["tool_calls"] = tool_calls
                    yield (
                        "data: "
                        + json.dumps(
                            {
                                "id": "autoconduck",
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": target or body.model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": delta,
                                        "finish_reason": None,
                                    }
                                ],
                            }
                        )
                        + "\n\n"
                    )
                    yield (
                        "data: "
                        + json.dumps(
                            {
                                "id": "autoconduck",
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": target or body.model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {},
                                        "finish_reason": finish_reason,
                                    }
                                ],
                            }
                        )
                        + "\n\n"
                    )
                    yield "data: [DONE]\n\n"
                else:
                    async for chunk in relay_for(target, extra, body.messages):
                        yield chunk
            except asyncio.CancelledError:
                task.cancel()
                raise
            finally:
                if await request.is_disconnected() and not task.done():
                    task.cancel()

        async def relay_for(target, extra, messages):
            from autoconduck.server.server_streaming import _litellm

            llm = _litellm()
            if llm is None:
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "error": {
                                "message": "litellm unavailable",
                                "type": "api_error",
                            }
                        }
                    )
                    + "\n\n"
                )
                yield "data: [DONE]\n\n"
                return
            kwargs = body.model_dump(exclude_none=True)
            kwargs["messages"] = normalize_messages_for_llm(messages)
            if kwargs.get("tools"):
                kwargs["tools"] = sanitize_tools(kwargs["tools"])
            kwargs.update(model=target, drop_params=True)
            kwargs.update(extra or {})
            response = await llm.acompletion(**kwargs)
            async for chunk in response:
                if await request.is_disconnected():
                    return
                payload = (
                    chunk.model_dump()
                    if hasattr(chunk, "model_dump")
                    else chunk
                )
                yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(progress_stream(), media_type="text/event-stream")

    target, extra = await route_target_fn(body.model, body.messages, request)
    answer = extra.get("__answer__")
    if answer is not None:
        record(
            extra.get("_path", "SLOW"),
            extra.get("_pseudo", body.model),
            target or "unknown",
            0,
            0,
            complexity=extra.get("_complexity"),
            route=extra.get("_route"),
            tier=extra.get("_tier"),
            plan=extra.get("_plan"),
        )
        created = int(time.time())
        content = (
            answer.get("content", "")
            if isinstance(answer, dict)
            else str(answer)
        )
        tool_calls = (
            answer.get("tool_calls")
            if isinstance(answer, dict)
            else getattr(answer, "tool_calls", None)
        )
        finish_reason = "tool_calls" if tool_calls else "stop"
        msg: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        if body.stream:

            async def answer_stream():
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "id": "autoconduck",
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": target or body.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": msg,
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
                    + "\n\n"
                )
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "id": "autoconduck",
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": target or body.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": finish_reason,
                                }
                            ],
                        }
                    )
                    + "\n\n"
                )
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                answer_stream(), media_type="text/event-stream"
            )
        return {
            "id": "autoconduck",
            "object": "chat.completion",
            "created": created,
            "model": target or body.model,
            "choices": [{"message": msg, "finish_reason": finish_reason}],
        }

    messages = normalize_messages_for_llm(body.messages)
    if extra.get("_path") == "FAST":
        from autoconduck.digest import maybe_digest_messages

        digest = await maybe_digest_messages(
            messages, config_module.get_config(), request=request
        )
        if digest:
            messages = messages + digest
    if body.stream:

        async def relay():
            from autoconduck.server.server_streaming import _litellm

            llm = _litellm()
            if llm is None:
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "error": {
                                "message": "litellm unavailable",
                                "type": "api_error",
                            }
                        }
                    )
                    + "\n\n"
                )
                yield "data: [DONE]\n\n"
                return
            kwargs = body.model_dump(exclude_none=True)
            kwargs["messages"] = normalize_messages_for_llm(messages)
            if kwargs.get("tools"):
                kwargs["tools"] = sanitize_tools(kwargs["tools"])
            kwargs.update(model=target, drop_params=True)
            kwargs.update(extra)
            response = await llm.acompletion(**kwargs)
            async for chunk in response:
                if await request.is_disconnected():
                    break
                payload = (
                    chunk.model_dump()
                    if hasattr(chunk, "model_dump")
                    else chunk
                )
                yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(relay(), media_type="text/event-stream")
    body.model = target
    return JSONResponse(
        await call_litellm_fn(
            target,
            body,
            extra.get("_path"),
            extra.get("_pseudo"),
            messages=messages,
        )
    )
