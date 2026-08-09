"""AutoConduck CLI and pragmatic FastAPI/LiteLLM hybrid surface."""
from __future__ import annotations
import argparse, asyncio, json, logging, os, sys, time, subprocess, shutil
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any
from .config import get_config, load_config, save_config, home_dir


# ---------- Lightweight modules always imported -------------------------------------------
# Heavy deps (fastapi, pydantic-core, litellm, textual, uvicorn) are deferred until a
# server/CLI command actually needs them.


DEFAULT_PORT = 11434
def _find_free_port(start: int, tries: int = 11) -> int: return start

# ---- Lazy-loaded application object ----------------------------------------------------
# The actual app object + its route-handler functions live inside ``_build()``.  We expose
# them through ``sys.modules[__name__].__getattr__`` so that existing tests which do
# ``monkeypatch.setattr(main, "something", mock)`` work even though nothing has been built
# yet.  The first time an API route is needed, _build() fires up fastapi / litellm etc.,
# then stores the real objects on the module namespace for subsequent direct access.


app = None   # type: ignore[assignment]

# Cache so that getattr once-built returns the real thing immediately.
_cached = {}  # key -> value  (populated by _build())


def __getattr__(name):
    """Dynamic attribute accessor for lazily-bootstrapped symbols.

    When _build() has already run the symbol is looked up in ``_cached`` first.
    If the requested name is only used by the FastAPI test harness it triggers
    the full build on demand.
    """
    if name in _cached:
        return _cached[name]
    # Only trigger the build for names we know will be used after the server starts
    if name in ("CompletionRequest", "MessagesRequest", "_route_target", "_call", "completions", "messages_endpoint"):
        _build()
        if name in _cached:
            return _cached[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _build():
    """Construct the FastAPI app (once) and populate *_cached*.

    This is the single point where heavy imports fire.
    """
    global app
    if app is not None:
        return
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel, Field
    from .messages_api import (
        openai_messages_from_anthropic,
        openai_tools_from_anthropic,
        openai_tool_choice_from_anthropic,
        serve_model_ids,
        custom_entry,
        litellm_params_for,
        count_tokens,
        AnthropicSSETranslator,
        anthropic_response_text,
        coerce_content_text,
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
        thinking: Any | None = None
        metadata: dict[str, Any] | None = None
        cache_control: Any | None = None

    class _Stats:
        decisions: list[dict[str, Any]] = []

    @asynccontextmanager
    async def lifespan(_a):
        try:
            import litellm  # warm the local import before the first request
        except ImportError:
            pass
        yield

    stats = _Stats()
    app = FastAPI(title="AutoConduck", lifespan=lifespan)
    PSEUDO = PSEUDO_MODELS

    async def _call(model, body):
        llm = _litellm()
        if llm is None: raise RuntimeError("litellm unavailable")
        kwargs = body.model_dump(exclude_none=True)
        kwargs["model"] = model
        kwargs.pop("stream", None)
        kwargs.update(litellm_params_for(model, get_config()))
        result = await llm.acompletion(**kwargs)
        return result.model_dump() if hasattr(result, "model_dump") else result

    async def _route_target(body_model, messages):
        """Resolve a pseudo/custom model to a concrete target + litellm kwargs."""
        started = time.perf_counter()
        cfg = get_config()
        target = body_model
        if body_model in PSEUDO:
            try:
                from .dispatcher import route
                decision = await asyncio.to_thread(
                    route, messages, [], pseudo_model=body_model, config=cfg,
                )
                path = getattr(decision, "path", "FAST").upper()
                model = getattr(decision, "model", None)
            except Exception:
                path, model = "FAST", None
            stats.decisions.append({"path": path, "model": model or body_model, "time": time.time()})
            logging.getLogger("autoconduck").info(
                "route=%s model=%s ms=%.1f", path, model or body_model,
                (time.perf_counter() - started) * 1000,
            )
            if path == "SLOW":
                try:
                    from .orchestrator import run
                    result = await run(messages, [], pseudo_model=body_model)
                    if result is not None:
                        return None, {"__answer__": result}
                except Exception:
                    pass
            if not model:
                try:
                    from .pricing import select
                    model = select(getattr(cfg, "model_list", []), body_model, cfg)
                except Exception:
                    from .config import resolve_orchestrator_model
                    model = resolve_orchestrator_model(cfg)
            target = model
        return target, litellm_params_for(target, cfg)

    def healthz():
        return {"status": "ok"}

    async def models():
        return {
            "object": "list",
            "data": [{"id": m, "object": "model", "owned_by": "autoconduck"} for m in serve_model_ids(get_config())],
        }

    async def get_stats():
        return {
            "counts": stats.decisions,
            "cost_saved_metered": 0.0,
            "cost_saved_subscription": 0.0,
            "cache_hit_ratio": 0.0,
        }

    async def completions(body: CompletionRequest, request: Request):
        target, extra = await _route_target(body.model, body.messages)
        if extra.get("__answer__") is not None:
            return {
                "id": "autoconduck", "object": "chat.completion",
                "choices": [{"message": {"role": "assistant", "content": extra["__answer__"]}}],
            }
        if body.stream:
            async def relay():
                llm = _litellm()
                if llm is None:
                    yield "data: " + json.dumps({"error": {"message": "litellm unavailable", "type": "api_error"}}) + "\n\n"
                    yield "data: [DONE]\n\n"
                    return
                kwargs = body.model_dump(exclude_none=True)
                kwargs["model"] = target
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
            return JSONResponse(anthropic_response_text(coerce_content_text(answer), target or body.model, input_text=json.dumps(oai_messages)))
        from .messages_api import messages_litellm_kwargs
        kwargs = messages_litellm_kwargs(target, extra)
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
        if body.stream:
            llm = _litellm()
            if llm is None:
                return JSONResponse({"type": "error", "error": {"type": "api_error", "message": "litellm unavailable"}}, status_code=502)
            try:
                response = await llm.acompletion(messages=oai_messages, stream=True, drop_params=True, **kwargs)
            except Exception as exc:
                return JSONResponse({"type": "error", "error": {"type": "api_error", "message": str(exc)}}, status_code=502)
            async def relay():
                translator = AnthropicSSETranslator(target, input_text=json.dumps(oai_messages))
                stopped = False
                try:
                    for ev in translator._ensure_message_start():
                        yield f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n"
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
                    for ev in translator.error(str(exc)):
                        yield f"event: {ev['event']}\ndata: {ev['data']}\n\n"
                    if not stopped:
                        yield 'event: message_stop\ndata: {"type": "message_stop"}\n\n'
            return StreamingResponse(relay(), media_type="text/event-stream")
        try:
            llm = _litellm()
            if llm is None:
                return JSONResponse({"type": "error", "error": {"type": "api_error", "message": "litellm unavailable"}}, status_code=503)
            result = await llm.acompletion(messages=oai_messages, stream=False, drop_params=True, **kwargs)
            text = result.choices[0].message.content if hasattr(result, "choices") else None
        except Exception as exc:
            return JSONResponse({"type": "error", "error": {"type": "api_error", "message": str(exc)}}, status_code=500)
        return JSONResponse(anthropic_response_text(coerce_content_text(text), target, input_text=json.dumps(oai_messages)))

    app.get("/healthz")(healthz)
    app.get("/v1/models")(models)
    app.get("/stats")(get_stats)
    app.post("/v1/chat/completions")(completions)
    app.post("/v1/messages")(messages_endpoint)
    @app.post("/v1/messages/count_tokens")
    async def messages_count_tokens(request: Request):
        try:
            body = await request.json()
        except Exception:
            body = {}
        oai_messages = openai_messages_from_anthropic(body)
        text = json.dumps(oai_messages)
        return {"input_tokens": count_tokens(text)}

    # Populate cache so subsequent getattr hits directly.
    _cached["CompletionRequest"] = CompletionRequest
    _cached["MessagesRequest"] = MessagesRequest
    _cached["_route_target"] = _route_target
    _cached["_call"] = _call
    _cached["healthz"] = healthz
    _cached["models"] = models
    _cached["get_stats"] = get_stats
    _cached["completions"] = completions
    _cached["messages_endpoint"] = messages_endpoint


# ---------- Public helpers ------------------------------------------------------------------

def _run_proxy(port: int, log_level: str = "info", host: str = "127.0.0.1"):
    """Start the FastAPI server via uvicorn."""
    configured_level = os.environ.get("AUTOCONDUCK_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, configured_level, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    import uvicorn
    uvicorn.run(_get_app(), host=host, port=port, log_level=log_level.lower(), access_log=False)

def _check_port_available(port: int) -> None:
    from .launcher import find_process_on_port, kill_process, prompt_kill_port
    pid = find_process_on_port(port)
    if pid is None:
        return
    if prompt_kill_port(port, pid) and kill_process(pid):
        print(f"Killed process {pid} using port {port}", file=sys.stderr)
        return
    print(f"Port {port} is in use by PID {pid}; kill it and retry", file=sys.stderr)
    raise SystemExit(1)

def _litellm():
    try:
        import litellm
        return litellm
    except ImportError:
        return None


def cmd_start(args):
    cfg = load_config()
    port = args.port or cfg.port or DEFAULT_PORT
    if not getattr(args, "headless", False) and not home_dir().exists():
        port = _find_free_port(port)
        _check_port_available(port)
        _run_proxy(port, cfg.log_level)
        return
    if getattr(args, "headless", False):
        if getattr(args, "daemon", False):
            _check_port_available(port)
            log = (home_dir() / "run" / "server.log")
            log.parent.mkdir(parents=True, exist_ok=True)
            from .launcher import daemon_python
            cmd = [
                daemon_python(), "-m", "autoconduck", "start",
                "--headless", "--daemon", "--port", str(port), "--host", args.host,
            ]
            with log.open("ab") as stream:
                flags = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW) if sys.platform == "win32" else 0
                subprocess.Popen(cmd, stdout=stream, stderr=stream, start_new_session=sys.platform != "win32", creationflags=flags, close_fds=True)
            return
        _check_port_available(port)
        _run_proxy(port, cfg.log_level, args.host) if args.host != "127.0.0.1" else _run_proxy(port, cfg.log_level)
    else:
        _check_port_available(port)
        try:
            from .tui.app import AutoConduckApp
            AutoConduckApp(configured=bool(getattr(cfg, "model_list", []))).run()
        except (ImportError, RuntimeError):
            _check_port_available(port)
            _run_proxy(port, cfg.log_level, args.host) if args.host != "127.0.0.1" else _run_proxy(port, cfg.log_level)


def cmd_edit(args):
    _get_app()  # ensure app object exists for dashboard usage
    from .tui.app import AutoConduckApp
    AutoConduckApp(configured=False).run()


def cmd_uninstall(args):
    if not args.force and input(
        "Uninstall AutoConduck, stop the daemon, and delete ALL state under the autoconduck home (config, backups, logs, run, cache, shims, state.json) as if freshly installed? [y/N] "
    ).lower() not in ("y", "yes"):
        return
    cfg = load_config()
    from .agents import all_adapters
    from . import launcher, update
    try:
        launcher.stop_server(getattr(cfg, "port", None) or DEFAULT_PORT)
    except Exception as exc:
        print(f"warning: could not stop daemon: {exc}")
    for adapter in all_adapters():
        try:
            adapter.revert()
        except Exception as exc:
            print(f"failed {adapter.display_name}: {exc}")
    launcher.uninstall_shims()
    launcher.remove_path_entry()
    purge_home_dir(home_dir())
    print("AutoConduck state purged; package remains installed.")
    hint = update.uninstall_hint(update.detect_install_method())
    if hint:
        print(f"Package still installed — remove it with: {hint}")


def purge_home_dir(home: Path) -> None:
    """Remove state, refusing obvious catastrophic paths."""
    import shutil
    try:
        resolved = home.resolve()
        if not home.exists() or not home.is_dir():
            return
        if resolved == Path.cwd().resolve() or resolved.parent == resolved:
            print(f"error: refusing to purge unsafe home directory: {resolved}")
            return
        shutil.rmtree(resolved, ignore_errors=True)
    except OSError as exc:
        print(f"error: could not purge home directory: {exc}")


def cmd_update(args):
    from . import __version__, update
    method = update.detect_install_method()
    command = update.upgrade_command(method)
    print(f"Current version: {__version__}")
    if command is None:
        if method == "uv-tool-editable":
            print("Editable checkout detected; update it manually (git pull), then reinstall with: uv tool install --editable .")
        elif method == "pip-editable":
            print("Editable checkout detected; update it manually (git pull), then reinstall with: pip install -e .")
        else:
            print("No managed installation detected; update the checkout manually (git pull) and reinstall.")
        return
    if args.dry_run:
        print(f"Would run: {command}")
        return
    tool = shutil.which(command.split()[0])
    if not tool:
        print(f"Error: required package manager '{command.split()[0]}' was not found on PATH.")
        return
    subprocess.call([tool, *command.split()[1:]])
    try:
        import importlib.metadata
        print(f"New version: {importlib.metadata.version('autoconduck')}")
    except importlib.metadata.PackageNotFoundError:
        print("Upgrade finished; run autoconduck --version to confirm the new version.")


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
    adapters = all_adapters()
    selected = args.agents or [a.id for a in adapters if launcher.real_binary_path(a.id)]
    for aid in selected:
        adapter = next((a for a in adapters if a.id == aid), None)
        if adapter is None:
            continue
        try:
            adapter.patch(load_config())
        except Exception as exc:
            print(f"failed {aid}: {exc}")
    paths = launcher.install_shims(selected)
    modified = launcher.ensure_path_entry()
    for aid, path in paths.items():
        print(f"{aid}: {path}")
    if modified:
        print(f"PATH: {modified}")


def cmd_launch_agent(agent_id: str, port: int | None = None) -> int:
    """Launch an agent by ID through the AutoConduck proxy server.

    Strategy: always kill any existing process on the port, start a fresh
    daemon, wait for readiness with exponential-backoff polling, then launch
    the agent binary.  Heavy imports (fastapi, litellm, textual) are deferred
    into the daemon child so this parent process stays lean.
    """
    from . import launcher
    # Only import the adapter we actually need (lazy)
    from .agents.claude_code import ClaudeCodeAdapter
    adapter_cls = ClaudeCodeAdapter

    cfg = load_config()
    port = port or getattr(cfg, "port", None) or DEFAULT_PORT

    # Kill anything currently listening on the port (always fresh start)
    launcher.kill_existing_on_port(port)

    log = (home_dir() / "run" / "server.log")
    log.parent.mkdir(parents=True, exist_ok=True)

    python_bin = launcher.daemon_python()
    cmd = [python_bin, "-m", "autoconduck", "start", "--headless", "--daemon", "--port", str(port)]

    import subprocess as _sp
    flags = (_sp.DETACHED_PROCESS | _sp.CREATE_NEW_PROCESS_GROUP | _sp.CREATE_NO_WINDOW) if sys.platform == "win32" else 0
    proc = _sp.Popen(cmd, stdout=log, stderr=log, start_new_session=sys.platform != "win32", creationflags=flags, close_fds=True)

    # Exponential-backoff health poll (max ~6 s instead of flat 10 s)
    server_ready = False
    for attempt in range(25):
        try:
            from urllib.request import urlopen
            with urlopen(f"http://127.0.0.1:{port}/healthz", timeout=0.5):
                server_ready = True
                break
        except Exception:
            pass
        time.sleep(min(0.15 * (1.5 ** attempt), 0.8))

    if not server_ready:
        print(f"error: server did not become ready within 6 s (daemon PID {proc.pid})", file=sys.stderr)
        return 1

    # Patch the adapter
    adapter = adapter_cls()
    adapter.patch(cfg)

    real_bin = launcher.real_binary_path(agent_id)
    if not real_bin:
        real_bin = shutil.which(adapter.binary_name)

    if not real_bin:
        print(f"agent '{agent_id}' not found on PATH; run: autoconduck install {agent_id}")
        launcher.release_server(port)
        return 1

    env = os.environ.copy()
    if agent_id == "claude_code":
        env.update(launcher._claude_env(port, getattr(cfg, "pseudo_model", "autoconduck")))

    print(f"AutoConduck ready at http://127.0.0.1:{port} — launching {adapter.binary_name}")
    try:
        return _sp.run([real_bin], env=env).returncode
    finally:
        launcher.release_server(port)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(prog="autoconduck")
    parser.add_argument("--version", action="store_true")
    parser.add_argument("--claude", action="store_true")
    parser.add_argument("--opencode", action="store_true")
    sub = parser.add_subparsers(dest="cmd")
    start = sub.add_parser("start")
    start.add_argument("--headless", action="store_true")
    start.add_argument("--daemon", action="store_true")
    start.add_argument("--port", type=int)
    start.add_argument("--host", default="127.0.0.1")
    for name, func in (("ensure", cmd_ensure), ("release", cmd_release), ("stop", cmd_stop)):
        p = sub.add_parser(name)
        p.add_argument("--port", type=int)
        p.set_defaults(handler=func)
    install = sub.add_parser("install")
    install.add_argument("agents", nargs="*")
    install.set_defaults(handler=cmd_install)
    upd = sub.add_parser("update")
    upd.add_argument("--dry-run", action="store_true")
    upd.set_defaults(handler=cmd_update)
    sub.add_parser("edit")
    uninstall = sub.add_parser("uninstall")
    uninstall.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.version:
        from . import __version__
        print(__version__)
    elif args.claude and args.opencode:
        print("--claude and --opencode cannot be used together", file=sys.stderr)
        raise SystemExit(2)
    elif args.claude:
        raise SystemExit(cmd_launch_agent("claude_code"))
    elif args.opencode:
        raise SystemExit(cmd_launch_agent("opencode"))
    elif args.cmd == "start":
        cmd_start(args)
    elif args.cmd == "edit":
        cmd_edit(args)
    elif args.cmd == "uninstall":
        cmd_uninstall(args)
    elif hasattr(args, "handler"):
        args.handler(args)
    else:
        cmd_start(argparse.Namespace(headless=False, port=None, host="127.0.0.1"))


if __name__ == "__main__":
    main()
