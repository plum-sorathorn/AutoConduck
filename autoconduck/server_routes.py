"""FastAPI route installation for the lazily loaded server."""

import json
import logging
import time
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

    async def _route_target(body_model, messages, request=None):
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
            decisions.append({"path": path, "model": model or body_model, "time": time.time()})
            logging.getLogger("autoconduck").info("route=%s model=%s ms=%.1f", path, model or body_model, (time.perf_counter() - started) * 1000)
            if path == "SLOW" and not (request is not None and await request.is_disconnected()):
                try:
                    from .orchestrator import run
                    result = await run(messages, [], pseudo_model=body_model,
                                       task_value=float(getattr(decision, "complexity", .5)), request=request)
                    if result is not None:
                        return None, {"__answer__": result, "_path": path, "_pseudo": body_model}
                except Exception:
                    pass
            if not model:
                try:
                    from .routing.pricing import pool_ids, select_closest
                    model = select_closest(pool_ids(cfg), .15, cfg, pseudo_model=body_model)
                except Exception:
                    from .config import resolve_orchestrator_model
                    model = resolve_orchestrator_model(cfg)
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
        target, extra = await _route_target(body.model, body.messages, request)
        answer = extra.get("__answer__")
        if answer is not None:
            record(extra.get("_path", "SLOW"), extra.get("_pseudo", body.model), target or "unknown", 0, 0)
            created = int(time.time())
            if body.stream:
                async def answer_stream():
                    yield "data: " + json.dumps({"id": "autoconduck", "object": "chat.completion.chunk", "created": created,
                        "model": target or body.model, "choices": [{"index": 0, "delta": {"role": "assistant", "content": answer}, "finish_reason": "stop"}]}) + "\n\n"
                    yield "data: [DONE]\n\n"
                return StreamingResponse(answer_stream(), media_type="text/event-stream")
            return {"id": "autoconduck", "object": "chat.completion", "created": created, "model": target or body.model,
                    "choices": [{"message": {"role": "assistant", "content": answer}}]}
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
            if body.stream:
                async def answer_stream():
                    translator = AnthropicSSETranslator(target or body.model, input_text=json.dumps(oai_messages))
                    for ev in translator._ensure_message_start(): yield f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n"
                    for ev in translator.translate({"choices": [{"delta": {"role": "assistant", "content": answer}, "finish_reason": "stop"}]}): yield f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n"
                return StreamingResponse(answer_stream(), media_type="text/event-stream")
            return JSONResponse(anthropic_response_text(answer, target or body.model, input_text=json.dumps(oai_messages)))
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
