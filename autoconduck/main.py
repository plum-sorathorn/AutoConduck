"""AutoConduck CLI and pragmatic FastAPI/LiteLLM hybrid surface."""
from __future__ import annotations
import argparse, asyncio, json, os, sys, time, subprocess
from contextlib import asynccontextmanager
from typing import Any
from .config import get_config, load_config, save_config, home_dir

DEFAULT_PORT = 11434
def _find_free_port(start: int, tries: int = 11) -> int: return start

def _run_proxy(port: int, log_level: str = "info", host: str = "127.0.0.1"):
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level=log_level.lower(), access_log=False)

def cmd_start(args):
    cfg = load_config(); port = args.port or cfg.port or DEFAULT_PORT
    if not getattr(args, "headless", False) and not home_dir().exists():
        _run_proxy(_find_free_port(port), cfg.log_level)
        return
    if getattr(args, "headless", False):
        if getattr(args, "daemon", False):
            log = (home_dir() / "run" / "server.log"); log.parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, "-m", "autoconduck", "start", "--headless", "--port", str(port), "--host", args.host]
            with log.open("ab") as stream:
                flags = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP) if sys.platform == "win32" else 0
                subprocess.Popen(cmd, stdout=stream, stderr=stream, start_new_session=sys.platform != "win32", creationflags=flags, close_fds=True)
            return
        _run_proxy(port, cfg.log_level, args.host) if args.host != "127.0.0.1" else _run_proxy(port, cfg.log_level)
    else:
        try:
            from .tui.app import AutoConduckApp
            AutoConduckApp(configured=bool(getattr(cfg, "model_list", []))).run()
        except (ImportError, RuntimeError):
            _run_proxy(port, cfg.log_level, args.host) if args.host != "127.0.0.1" else _run_proxy(port, cfg.log_level)

def cmd_edit(args):
    from .tui.app import AutoConduckApp
    AutoConduckApp(configured=True).run()

def cmd_uninstall(args):
    if not args.force and input("Uninstall AutoConduck and restore agent configs? [y/N] ").lower() not in ("y", "yes"): return
    from .agents import all_adapters
    from . import launcher
    for adapter in all_adapters():
        try: adapter.revert()
        except Exception as exc: print(f"failed {adapter.display_name}: {exc}")
    path = home_dir() / "config.yaml"
    if path.exists(): path.unlink()
    launcher.uninstall_shims(); launcher.remove_path_entry()

def cmd_ensure(args):
    from . import launcher
    launcher.ensure_server(args.port)

def cmd_release(args):
    from . import launcher
    launcher.release_server(args.port)

def cmd_stop(args):
    from . import launcher
    launcher.stop_server(args.port)

def cmd_install(args):
    from . import launcher
    from .agents import all_adapters
    adapters = all_adapters(); selected = args.agents or [a.id for a in adapters if launcher.real_binary_path(a.id)]
    for aid in selected:
        adapter = next((a for a in adapters if a.id == aid), None)
        if adapter is None: continue
        try: adapter.patch(load_config())
        except Exception as exc: print(f"failed {aid}: {exc}")
    paths = launcher.install_shims(selected)
    modified = launcher.ensure_path_entry()
    for aid, path in paths.items(): print(f"{aid}: {path}")
    if modified: print(f"PATH: {modified}")

def _litellm():
    try:
        import litellm
        return litellm
    except ImportError: return None

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel, Field
    from .messages_api import (
        openai_messages_from_anthropic,
        serve_model_ids,
        custom_entry,
        litellm_params_for,
        count_tokens,
        AnthropicSSETranslator,
        anthropic_response_text,
        PSEUDO_MODELS,
    )
    class CompletionRequest(BaseModel):
        model: str
        messages: list[dict[str, Any]] = Field(default_factory=list)
        stream: bool = False
        temperature: float | None = None
        max_tokens: int | None = None
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
    class _Stats:
        decisions: list[dict[str, Any]] = []
    @asynccontextmanager
    async def lifespan(_app):
        try:
            import litellm  # warm the local import before the first request
        except ImportError:
            pass
        yield
    stats = _Stats(); app = FastAPI(title="AutoConduck", lifespan=lifespan)
    PSEUDO = PSEUDO_MODELS
    async def _call(model, body):
        llm = _litellm()
        if llm is None: return {"id": "autoconduck", "choices": [{"message": {"role": "assistant", "content": ""}}]}
        kwargs = body.model_dump(exclude_none=True); kwargs["model"] = model; kwargs.pop("stream", None)
        kwargs.update(litellm_params_for(model, get_config()))
        result = await llm.acompletion(**kwargs)
        return result.model_dump() if hasattr(result, "model_dump") else result
    async def _route_target(body_model: str, messages: list) -> tuple[str | None, dict]:
        """Resolve a pseudo/custom model to a concrete target + litellm kwargs.

        Returns (target, extra) where extra may contain "__answer__" if the
        orchestrator already produced a full answer (SLOW path short-circuit).
        """
        cfg = get_config()
        target = body_model
        if body_model in PSEUDO:
            try:
                from .dispatcher import route
                decision = await asyncio.to_thread(route, messages, [], pseudo_model=body_model, config=cfg)
                path = getattr(decision, "path", "FAST").upper(); model = getattr(decision, "model", None)
            except Exception: path, model = "FAST", None
            stats.decisions.append({"path": path, "model": model or body_model, "time": time.time()})
            if path == "SLOW":
                try:
                    from .orchestrator import run
                    result = await run(messages, [], pseudo_model=body_model)
                    if result is not None:
                        return None, {"__answer__": result}
                except Exception: pass
            if not model:
                try:
                    from .pricing import select
                    model = select(getattr(cfg, "model_list", []), body_model, cfg)
                except Exception:
                    from .config import resolve_orchestrator_model
                    model = resolve_orchestrator_model(cfg)
            target = model
        kwargs = litellm_params_for(target, cfg)
        return target, kwargs
    @app.get("/healthz")
    async def healthz(): return {"status": "ok"}
    @app.get("/v1/models")
    async def models(): return {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "autoconduck"} for m in serve_model_ids(get_config())]}
    @app.get("/stats")
    async def get_stats(): return {"counts": stats.decisions, "cost_saved_metered": 0.0, "cost_saved_subscription": 0.0, "cache_hit_ratio": 0.0}
    @app.post("/v1/chat/completions")
    async def completions(body: CompletionRequest, request: Request):
        target, extra = await _route_target(body.model, body.messages)
        if extra.get("__answer__") is not None:
            return {"id": "autoconduck", "object": "chat.completion", "choices": [{"message": {"role": "assistant", "content": extra["__answer__"]}}]}
        if body.stream:
            async def relay():
                llm = _litellm()
                if llm is None: return
                kwargs = body.model_dump(exclude_none=True); kwargs["model"] = target
                kwargs.update(extra)
                response = await llm.acompletion(**kwargs)
                async for chunk in response:
                    if await request.is_disconnected(): break
                    payload = chunk.model_dump() if hasattr(chunk, "model_dump") else chunk
                    yield f"data: {json.dumps(payload)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(relay(), media_type="text/event-stream")
        body.model = target
        return JSONResponse(await _call(target, body))
    @app.post("/v1/messages")
    async def messages_endpoint(body: MessagesRequest, request: Request):
        cfg = get_config()
        try:
            oai_messages = openai_messages_from_anthropic(body.model_dump(exclude_none=True))
        except Exception as exc:
            return JSONResponse({"type": "error", "error": {"type": "invalid_request_error", "message": str(exc)}}, status_code=400)
        try:
            target, extra = await _route_target(body.model, oai_messages)
        except Exception as exc:
            return JSONResponse({"type": "error", "error": {"type": "api_error", "message": str(exc)}}, status_code=500)
        if extra.get("__answer__") is not None:
            answer = extra["__answer__"]
            if body.stream:
                async def relay_answer():
                    translator = AnthropicSSETranslator(target or body.model)
                    chunk = {"choices": [{"delta": {"role": "assistant", "content": answer}, "finish_reason": "stop"}]}
                    for ev in translator.translate(chunk):
                        yield f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n"
                return StreamingResponse(relay_answer(), media_type="text/event-stream")
            return JSONResponse(anthropic_response_text(answer, target or body.model))
        from .messages_api import messages_litellm_kwargs
        kwargs = messages_litellm_kwargs(target, extra)
        if body.stream:
            async def relay():
                translator = AnthropicSSETranslator(target, input_text=json.dumps(oai_messages))
                stopped = False
                try:
                    llm = _litellm()
                    if llm is None:
                        raise RuntimeError("litellm unavailable")
                    response = await llm.acompletion(messages=oai_messages, stream=True, **kwargs)
                    last_emit = time.monotonic()
                    async for chunk in response:
                        if await request.is_disconnected(): break
                        payload = chunk.model_dump() if hasattr(chunk, "model_dump") else chunk
                        for ev in translator.translate(payload):
                            if ev["type"] == "message_stop":
                                stopped = True
                            yield f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n"
                        now = time.monotonic()
                        if now - last_emit > 5:
                            yield 'event: ping\ndata: {"type": "ping"}\n\n'
                        last_emit = now
                    if not stopped:
                        for ev in translator.finish():
                            if ev["type"] == "message_stop":
                                stopped = True
                            yield f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n"
                except Exception as exc:
                    yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'type': 'api_error', 'message': str(exc)}})}\n\n"
                    if not stopped:
                        yield 'event: message_stop\ndata: {"type": "message_stop"}\n\n'
            return StreamingResponse(relay(), media_type="text/event-stream")
        try:
            llm = _litellm()
            if llm is None:
                return JSONResponse(anthropic_response_text("", target))
            result = await llm.acompletion(messages=oai_messages, stream=False, **kwargs)
            text = result.choices[0].message.content if hasattr(result, "choices") else ""
        except Exception as exc:
            return JSONResponse({"type": "error", "error": {"type": "api_error", "message": str(exc)}}, status_code=500)
        return JSONResponse(anthropic_response_text(text or "", target))
    @app.post("/v1/messages/count_tokens")
    async def messages_count_tokens(request: Request):
        try:
            body = await request.json()
        except Exception:
            body = {}
        oai_messages = openai_messages_from_anthropic(body)
        text = json.dumps(oai_messages)
        return {"input_tokens": count_tokens(text)}
except ImportError:
    app = None

def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(prog="autoconduck"); parser.add_argument("--version", action="store_true")
    sub = parser.add_subparsers(dest="cmd"); start = sub.add_parser("start"); start.add_argument("--headless", action="store_true"); start.add_argument("--daemon", action="store_true"); start.add_argument("--port", type=int); start.add_argument("--host", default="127.0.0.1")
    for name, func in (("ensure", cmd_ensure), ("release", cmd_release), ("stop", cmd_stop)):
        p = sub.add_parser(name); p.add_argument("--port", type=int); p.set_defaults(handler=func)
    install = sub.add_parser("install"); install.add_argument("agents", nargs="*"); install.set_defaults(handler=cmd_install)
    sub.add_parser("edit"); uninstall = sub.add_parser("uninstall"); uninstall.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.version: from . import __version__; print(__version__)
    elif args.cmd == "start": cmd_start(args)
    elif args.cmd == "edit": cmd_edit(args)
    elif args.cmd == "uninstall": cmd_uninstall(args)
    elif hasattr(args, "handler"): args.handler(args)
    else: cmd_start(argparse.Namespace(headless=False, port=None, host="127.0.0.1"))
if __name__ == "__main__": main()
