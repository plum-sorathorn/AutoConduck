"""FastAPI route installation for the lazily loaded server."""

import json
import logging
import os
import time
import asyncio
from typing import Any


def install_routes(app, Request, JSONResponse, StreamingResponse, BaseModel, Field,
                   helpers, serve_model_ids, PSEUDO_MODELS, cache):
    (openai_messages_from_anthropic, openai_tools_from_anthropic,
     openai_tool_choice_from_anthropic, litellm_params_for, count_tokens,
     AnthropicSSETranslator, anthropic_response_text, coerce_content_text,
     sanitize_tools, messages_litellm_kwargs, *extra_helpers) = helpers
    normalize_messages_for_llm = extra_helpers[0] if extra_helpers else None
    if normalize_messages_for_llm is None:
        from .messages_api import normalize_messages_for_llm
    from fastapi import FastAPI
    from .config import get_config
    from .stats import aggregate, load_records, record

    class CompletionRequest(BaseModel):
        model: str
        messages: list[dict[str, Any]] = Field(default_factory=list)
        stream: bool = False
        temperature: float | None = None
        max_tokens: int | None = None
        tools: list[dict[str, Any]] | None = None
        tool_choice: Any | None = None

    class MessagesRequest(BaseModel):
        model: str
        messages: list[dict[str, Any]] = Field(default_factory=list)
        system: Any | None = None
        max_tokens: int | None = None
        temperature: float | None = None
        top_p: float | None = None
        stop_sequences: list[str] | None = None
        stream: bool = True
        tools: list[dict[str, Any]] | None = None
        tool_choice: Any | None = None
        thinking: Any | None = None
        metadata: dict[str, Any] | None = None
        cache_control: Any | None = None

    if app is None:
        app = FastAPI(title="AutoConduck")
    decisions = []

    async def _call(model, body, path=None, pseudo=None, messages=None):
        from .server_streaming import _litellm
        llm = _litellm()
        if llm is None:
            raise RuntimeError("litellm unavailable")
        kwargs = body.model_dump(exclude_none=True)
        if messages is not None:
            kwargs["messages"] = normalize_messages_for_llm(messages)
        if kwargs.get("tools"):
            kwargs["tools"] = sanitize_tools(kwargs["tools"])
        kwargs.update(model=model, drop_params=True)
        kwargs.pop("stream", None)
        kwargs.update(litellm_params_for(model, get_config()))
        kwargs["_path"] = path if path is not None else "unknown"
        kwargs["_pseudo"] = pseudo if pseudo is not None else "unknown"
        result = await llm.acompletion(**kwargs)
        return result.model_dump() if hasattr(result, "model_dump") else result

    async def _route_target(body_model, messages, request=None, on_progress=None):
        started = time.perf_counter()
        cfg = get_config()
        messages = normalize_messages_for_llm(messages)
        target, path = body_model, "direct"
        if body_model in PSEUDO_MODELS:
            try:
                from .routing.dispatcher import route

                history = decisions[-5:] if decisions else []
                decision = route(messages, history, pseudo_model=body_model, config=cfg)
                path = getattr(decision, "path", "FAST").upper()
                model = getattr(decision, "model", None)
            except Exception:
                decision, path, model = None, "FAST", None
            # Guard: only invoke the full LangGraph orchestrator when complexity
            # is genuinely high enough to justify 3-5 extra LLM calls.
            task_complexity = float(getattr(decision, "complexity", 0.5))
            min_orch = float(
                getattr(getattr(cfg, "selection", None), "min_orchestrator_complexity", 0.62)
            )
            if path == "SLOW" and task_complexity < min_orch:
                path = "FAST"
                model = model or None  # will be resolved below
            if on_progress is not None:
                try:
                    on_progress({"kind": "route", "path": path})
                except Exception:
                    pass
            decisions.append({"path": path, "model": model or body_model, "time": time.time()})
            logging.getLogger("autoconduck").info("route=%s model=%s ms=%.1f", path, model or body_model, (time.perf_counter() - started) * 1000)
            if path == "SLOW" and not (request is not None and await request.is_disconnected()):
                try:
                    from .orchestrator import run
                    result = await run(messages, [], pseudo_model=body_model,
                                       task_value=float(getattr(decision, "complexity", .5)), request=request,
                                       on_progress=on_progress)
                    if result is not None:
                        tool_calls = getattr(result, "tool_calls", None) or (result.get("tool_calls") if isinstance(result, dict) else None)
                        content = str(result) if not isinstance(result, dict) else result.get("content", str(result))
                        ans: dict[str, Any] = {"content": content}
                        if tool_calls:
                            ans["tool_calls"] = tool_calls
                        return None, {"__answer__": ans, "_path": path, "_pseudo": body_model}
                except Exception as exc:
                    logging.getLogger("autoconduck").warning("Orchestrator execution failed: %s", exc)
            if not model:
                try:
                    from .routing.pricing import pool_ids, select_closest
                    from .config import resolve_orchestrator_model
                    selected = select_closest(pool_ids(cfg), .15, cfg, pseudo_model=body_model)
                    model = selected or resolve_orchestrator_model(cfg)
                    if not selected:
                        logging.getLogger("autoconduck").warning(
                            "Model pool is empty - no models configured; falling back to %s",
                            model,
                        )
                except Exception:
                    from .config import resolve_orchestrator_model
                    model = resolve_orchestrator_model(cfg)
            if not model:
                logging.getLogger("autoconduck").warning("No model available for request")
            target = model
        extra = litellm_params_for(target, cfg)
        extra.update(_path=path if body_model in PSEUDO_MODELS else "direct", _pseudo=body_model)
        return target, extra

    def healthz():
        return {"status": "ok"}

    async def models():
        return {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "autoconduck"}
                for m in serve_model_ids(get_config())]}

    async def get_stats():
        usage = aggregate(load_records())
        return {"counts": decisions, "cost_saved_metered": 0.0, "cost_saved_subscription": 0.0,
                "cache_hit_ratio": 0.0, "usage": usage["totals"], "models": usage["models"],
                "path_counts": usage["paths"], "pseudo_counts": usage["pseudos"]}

    async def completions(body: CompletionRequest, request: Request):
        body.messages = normalize_messages_for_llm(body.messages)
        cfg = get_config()
        configured_progress = bool(getattr(getattr(cfg, "selection", None), "slow_stream_progress", True))
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
        progress_enabled = body.stream and progress_setting and body.model in PSEUDO_MODELS
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

                task = asyncio.create_task(_route_target(body.model, body.messages, request, on_progress))
                try:
                    first = await first_event
                    first_path = (
                        getattr(first, "path", None)
                        or (first.get("path") if isinstance(first, dict) else "")
                        or ""
                    )
                    is_slow = str(first_path).upper() == "SLOW"
                    target, extra = await task if not is_slow else (None, None)
                    if is_slow:
                        labels = {"recon": "recon", "recon_subagent_pool": "reading files",
                                  "planner": "planner", "subagent_pool": "subagents",
                                  "compactor": "compactor", "executor": "executor"}
                        created = int(time.time())
                        sent_role = False
                        while True:
                            event = await progress_q.get()
                            delta_text = None
                            if isinstance(event, str):
                                delta_text, node = event, "progress"
                            elif isinstance(event, dict):
                                if event.get("kind") == "route":
                                    continue
                                node = event.get("node", "progress")
                                detail = event.get("step_detail") or node
                                label = labels.get(node, node)
                                delta_text = f"[{label}] {detail}\n"
                            else:
                                from .progress import ProgressFormatter
                                formatted = ProgressFormatter(cfg).format(event)
                                node = getattr(event, "name", "progress")
                                if not formatted:
                                    continue
                                delta_text = formatted + "\n"
                            if delta_text is None:
                                continue
                            delta = {"content": delta_text}
                            if not sent_role:
                                delta["role"] = "assistant"
                                sent_role = True
                            yield "data: " + json.dumps({"id": "autoconduck", "object": "chat.completion.chunk",
                                "created": created, "model": body.model,
                                "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}) + "\n\n"
                            if node == "idle":
                                break
                            if task.done() and progress_q.empty():
                                break
                        target, extra = await task
                    answer = extra.get("__answer__") if extra else None
                    if answer is not None:
                        content = answer.get("content", "") if isinstance(answer, dict) else str(answer)
                        tool_calls = answer.get("tool_calls") if isinstance(answer, dict) else getattr(answer, "tool_calls", None)
                        finish_reason = "tool_calls" if tool_calls else "stop"
                        created = int(time.time())
                        delta = {"role": "assistant"}
                        if content:
                            delta["content"] = content
                        if tool_calls:
                            delta["tool_calls"] = tool_calls
                        yield "data: " + json.dumps({"id": "autoconduck", "object": "chat.completion.chunk",
                            "created": created, "model": target or body.model,
                            "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}) + "\n\n"
                        yield "data: " + json.dumps({"id": "autoconduck", "object": "chat.completion.chunk",
                            "created": created, "model": target or body.model,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]}) + "\n\n"
                        yield "data: [DONE]\n\n"
                    else:
                        # FAST (or an orchestrator fallback): relay the normal provider stream.
                        async for chunk in relay_for(target, extra, body.messages):
                            yield chunk
                except asyncio.CancelledError:
                    task.cancel()
                    raise
                finally:
                    if await request.is_disconnected() and not task.done():
                        task.cancel()

            async def relay_for(target, extra, messages):
                from .server_streaming import _litellm
                llm = _litellm()
                if llm is None:
                    yield 'data: ' + json.dumps({"error": {"message": "litellm unavailable", "type": "api_error"}}) + "\n\n"
                    yield "data: [DONE]\n\n"
                    return
                kwargs = body.model_dump(exclude_none=True)
                kwargs["messages"] = normalize_messages_for_llm(messages)
                if kwargs.get("tools"): kwargs["tools"] = sanitize_tools(kwargs["tools"])
                kwargs.update(model=target, drop_params=True)
                kwargs.update(extra or {})
                response = await llm.acompletion(**kwargs)
                async for chunk in response:
                    if await request.is_disconnected(): return
                    payload = chunk.model_dump() if hasattr(chunk, "model_dump") else chunk
                    yield f"data: {json.dumps(payload)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(progress_stream(), media_type="text/event-stream")
        target, extra = await _route_target(body.model, body.messages, request)
        answer = extra.get("__answer__")
        if answer is not None:
            record(extra.get("_path", "SLOW"), extra.get("_pseudo", body.model), target or "unknown", 0, 0)
            created = int(time.time())
            content = answer.get("content", "") if isinstance(answer, dict) else str(answer)
            tool_calls = answer.get("tool_calls") if isinstance(answer, dict) else getattr(answer, "tool_calls", None)
            finish_reason = "tool_calls" if tool_calls else "stop"
            msg: dict[str, Any] = {"role": "assistant", "content": content}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            if body.stream:
                async def answer_stream():
                    yield "data: " + json.dumps({"id": "autoconduck", "object": "chat.completion.chunk", "created": created,
                        "model": target or body.model, "choices": [{"index": 0, "delta": msg, "finish_reason": finish_reason}]}) + "\n\n"
                    yield "data: [DONE]\n\n"
                return StreamingResponse(answer_stream(), media_type="text/event-stream")
            return {"id": "autoconduck", "object": "chat.completion", "created": created, "model": target or body.model,
                    "choices": [{"message": msg, "finish_reason": finish_reason}]}
        messages = normalize_messages_for_llm(body.messages)
        if extra.get("_path") == "FAST":
            from .digest import maybe_digest_messages
            digest = await maybe_digest_messages(messages, get_config(), request=request)
            if digest:
                messages = messages + digest
        if body.stream:
            async def relay():
                from .server_streaming import _litellm
                llm = _litellm()
                if llm is None:
                    yield 'data: ' + json.dumps({"error": {"message": "litellm unavailable", "type": "api_error"}}) + "\n\n"
                    yield "data: [DONE]\n\n"; return
                kwargs = body.model_dump(exclude_none=True)
                kwargs["messages"] = normalize_messages_for_llm(messages)
                if kwargs.get("tools"): kwargs["tools"] = sanitize_tools(kwargs["tools"])
                kwargs.update(model=target, drop_params=True)
                kwargs.update(extra)
                response = await llm.acompletion(**kwargs)
                async for chunk in response:
                    if await request.is_disconnected(): break
                    payload = chunk.model_dump() if hasattr(chunk, "model_dump") else chunk
                    yield f"data: {json.dumps(payload)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(relay(), media_type="text/event-stream")
        body.model = target
        return JSONResponse(await _call(target, body, extra.get("_path"), extra.get("_pseudo"), messages=messages))

    async def messages_endpoint(body: MessagesRequest, request: Request):
        try: oai_messages = normalize_messages_for_llm(openai_messages_from_anthropic(body.model_dump(exclude_none=True)))
        except Exception as exc: return JSONResponse({"type": "error", "error": {"type": "invalid_request_error", "message": str(exc)}}, status_code=400)
        try: target, extra = await _route_target(body.model, oai_messages, request)
        except Exception as exc: return JSONResponse({"type": "error", "error": {"type": "api_error", "message": str(exc)}}, status_code=500)
        answer = extra.get("__answer__")
        if answer is not None:
            content = answer.get("content", "") if isinstance(answer, dict) else str(answer)
            tool_calls = answer.get("tool_calls") if isinstance(answer, dict) else getattr(answer, "tool_calls", None)
            if body.stream:
                async def answer_stream():
                    translator = AnthropicSSETranslator(target or body.model, input_text=json.dumps(oai_messages))
                    for ev in translator._ensure_message_start(): yield f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n"
                    delta: dict[str, Any] = {"role": "assistant", "content": content}
                    if tool_calls: delta["tool_calls"] = tool_calls
                    for ev in translator.translate({"choices": [{"delta": delta, "finish_reason": "tool_calls" if tool_calls else "stop"}]}): yield f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n"
                return StreamingResponse(answer_stream(), media_type="text/event-stream")
            if not tool_calls:
                return JSONResponse(anthropic_response_text(content, target or body.model, input_text=json.dumps(oai_messages)))
            import uuid
            content_blocks: list[dict[str, Any]] = []
            if content:
                content_blocks.append({"type": "text", "text": content})
            for tc in tool_calls:
                fn = tc.get("function") or {}
                raw_args = fn.get("arguments", "{}")
                parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id") or ("toolu_" + uuid.uuid4().hex[:12]),
                    "name": fn.get("name"),
                    "input": parsed_args,
                })
            return JSONResponse({
                "id": "msg_" + uuid.uuid4().hex[:12],
                "type": "message",
                "role": "assistant",
                "content": content_blocks,
                "model": target or body.model,
                "stop_reason": "tool_use",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": count_tokens(json.dumps(oai_messages)),
                    "output_tokens": count_tokens(content) + count_tokens(json.dumps(content_blocks)),
                },
            })
        if extra.get("_path") == "FAST":
            from .digest import maybe_digest_messages
            digest = await maybe_digest_messages(oai_messages, get_config(), request=request)
            if digest:
                oai_messages = oai_messages + digest
        kwargs = messages_litellm_kwargs(target, extra)
        kwargs.update(_path=extra.get("_path", "unknown"), _pseudo=extra.get("_pseudo", body.model))
        for name, value in (("tools", openai_tools_from_anthropic(body.tools)), ("tool_choice", openai_tool_choice_from_anthropic(body.tool_choice)),
                            ("max_tokens", body.max_tokens), ("stop", body.stop_sequences), ("temperature", body.temperature),
                            ("top_p", body.top_p), ("thinking", body.thinking), ("metadata", body.metadata), ("cache_control", body.cache_control)):
            if value is not None: kwargs[name] = value
        from .server_streaming import _litellm
        llm = _litellm()
        if llm is None: return JSONResponse({"type": "error", "error": {"type": "api_error", "message": "litellm unavailable"}}, status_code=502 if body.stream else 503)
        if body.stream:
            try: response = await llm.acompletion(messages=oai_messages, stream=True, drop_params=True, **kwargs)
            except Exception as exc: return JSONResponse({"type": "error", "error": {"type": "api_error", "message": str(exc)}}, status_code=502)
            async def relay():
                translator = AnthropicSSETranslator(target, input_text=json.dumps(oai_messages))
                try:
                    for ev in translator._ensure_message_start(): yield f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n"
                    async for chunk in response:
                        if await request.is_disconnected(): break
                        payload = chunk.model_dump() if hasattr(chunk, "model_dump") else chunk
                        for ev in translator.translate(payload): yield f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n"
                    for ev in translator.finish(): yield f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n"
                except Exception as exc:
                    exc_str = str(exc)
                    if "Error building chunks" in exc_str or "stream_chunk_builder" in exc_str or "list index out of range" in exc_str:
                        for ev in translator.finish(): yield f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n"
                    else:
                        for ev in translator.error(str(exc)): yield f"event: {ev['event']}\ndata: {ev['data']}\n\n"
            return StreamingResponse(relay(), media_type="text/event-stream")
        try:
            result = await llm.acompletion(messages=oai_messages, stream=False, drop_params=True, **kwargs)
            text = result.choices[0].message.content if hasattr(result, "choices") else None
        except Exception as exc: return JSONResponse({"type": "error", "error": {"type": "api_error", "message": str(exc)}}, status_code=500)
        return JSONResponse(anthropic_response_text(coerce_content_text(text), target, input_text=json.dumps(oai_messages)))

    async def messages_count_tokens(request: Request):
        try: body = await request.json()
        except Exception: body = {}
        return {"input_tokens": count_tokens(json.dumps(openai_messages_from_anthropic(body)))}

    app.get("/healthz")(healthz); app.get("/v1/models")(models); app.get("/stats")(get_stats)
    app.post("/v1/chat/completions")(completions); app.post("/v1/messages")(messages_endpoint)
    app.post("/v1/messages/count_tokens")(messages_count_tokens)
    if cache is not None:
        cache.update(CompletionRequest=CompletionRequest, MessagesRequest=MessagesRequest,
                     _route_target=_route_target, _call=_call, completions=completions,
                     messages_endpoint=messages_endpoint, healthz=healthz, models=models, get_stats=get_stats)
    return app
