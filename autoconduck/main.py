"""AutoConduck CLI and pragmatic FastAPI/LiteLLM hybrid surface."""
from __future__ import annotations
import argparse, json, sys, time, subprocess
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
    class CompletionRequest(BaseModel):
        model: str
        messages: list[dict[str, Any]] = Field(default_factory=list)
        stream: bool = False
        temperature: float | None = None
        max_tokens: int | None = None
    class _Stats:
        decisions: list[dict[str, Any]] = []
    stats = _Stats(); app = FastAPI(title="AutoConduck")
    PSEUDO = {"autoconduck", "autoconduck-budget", "autoconduck-expensive"}
    async def _call(model, body):
        llm = _litellm()
        if llm is None: return {"id": "autoconduck", "choices": [{"message": {"role": "assistant", "content": ""}}]}
        kwargs = body.model_dump(exclude_none=True); kwargs["model"] = model; kwargs.pop("stream", None)
        result = await llm.acompletion(**kwargs)
        return result.model_dump() if hasattr(result, "model_dump") else result
    @app.get("/healthz")
    async def healthz(): return {"status": "ok"}
    @app.get("/v1/models")
    async def models(): return {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "autoconduck"} for m in PSEUDO]}
    @app.get("/stats")
    async def get_stats(): return {"counts": stats.decisions, "cost_saved_metered": 0.0, "cost_saved_subscription": 0.0, "cache_hit_ratio": 0.0}
    @app.post("/v1/chat/completions")
    async def completions(body: CompletionRequest, request: Request):
        target = body.model
        if target in PSEUDO:
            try:
                from .dispatcher import route
                decision = route(body.messages, [], pseudo_model=target)
                path = getattr(decision, "path", "FAST").upper(); model = getattr(decision, "model", None)
            except Exception: path, model = "FAST", None
            stats.decisions.append({"path": path, "model": model or target, "time": time.time()})
            if path == "SLOW":
                try:
                    from .orchestrator import run
                    result = await run(body.messages, [], pseudo_model=target)
                    if result is not None:
                        return {"id": "autoconduck", "object": "chat.completion", "choices": [{"message": {"role": "assistant", "content": result}}]}
                except Exception: pass
            if not model:
                try:
                    from .pricing import select
                    cfg = get_config(); model = select(getattr(cfg, "model_list", []), target, cfg)
                except Exception: model = target
            target = model
        if body.stream:
            async def relay():
                llm = _litellm()
                if llm is None: return
                kwargs = body.model_dump(exclude_none=True); kwargs["model"] = target
                response = await llm.acompletion(**kwargs)
                async for chunk in response:
                    if await request.is_disconnected(): break
                    payload = chunk.model_dump() if hasattr(chunk, "model_dump") else chunk
                    yield f"data: {json.dumps(payload)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(relay(), media_type="text/event-stream")
        return JSONResponse(await _call(target, body))
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
