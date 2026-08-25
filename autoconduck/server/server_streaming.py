"""Lazy-loaded FastAPI/LiteLLM server implementation."""

import argparse, asyncio, ctypes, json, logging, os, sys, time, subprocess, shutil, signal, traceback
from contextlib import asynccontextmanager
from typing import Any
from autoconduck import config
from autoconduck.config import get_config, home_dir


DEFAULT_PORT = 11434
logger = logging.getLogger("autoconduck")


def _write_crash_report(exc):
    crash_path = home_dir() / "run" / "server.crash"
    crash_path.parent.mkdir(parents=True, exist_ok=True)
    with crash_path.open("a", encoding="utf-8") as stream:
        stream.write(f"\n--- AutoConduck crash at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        stream.write(traceback.format_exc())


def _find_free_port(start: int, tries: int = 11) -> int:
    return start# ---- Lazy-loaded application object ----------------------------------------------------
# The actual app object + its route-handler functions live inside ``_build()``.  We expose
# them through ``sys.modules[__name__].__getattr__`` so that existing tests which do
# ``monkeypatch.setattr(main, "something", mock)`` work even though nothing has been built
# yet.  The first time an API route is needed, _build() fires up fastapi / litellm etc.,
# then stores the real objects on the module namespace for subsequent direct access.


app = None  # type: ignore[assignment]

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
    if name in (
        "CompletionRequest",
        "MessagesRequest",
        "_route_target",
        "_call",
        "completions",
        "messages_endpoint",
    ):
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
    import litellm
    litellm.suppress_debug_info = True
    litellm.set_verbose = False
    from autoconduck.stats import install_recorder

    install_recorder(litellm)
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
        sanitize_tools,
        messages_litellm_kwargs,
        normalize_messages_for_llm,
        PSEUDO_MODELS,
    )

    from .server_routes import install_routes
    app = install_routes(
        app, Request, JSONResponse, StreamingResponse, BaseModel, Field,
        (openai_messages_from_anthropic, openai_tools_from_anthropic, openai_tool_choice_from_anthropic, litellm_params_for, count_tokens, AnthropicSSETranslator, anthropic_response_text, coerce_content_text, sanitize_tools, messages_litellm_kwargs, normalize_messages_for_llm),
        serve_model_ids, PSEUDO_MODELS, _cached,
    )



def _get_app():
    _build()
    return app


# ---------- Public helpers ------------------------------------------------------------------


def _run_proxy(port: int, log_level: str = "info", host: str = "127.0.0.1"):
    """Start the FastAPI server via uvicorn."""
    configured_level = os.environ.get("AUTOCONDUCK_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, configured_level, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        import uvicorn

        # Write a readiness sentinel so cmd_launch_agent detects startup via filesystem
        # rather than HTTP polling (faster on machines where the first TCP connect is slow).
        def _write_ready():
            try:
                marker = home_dir() / "run" / f"server_{port}.ready"
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text("ready")
            except Exception:
                pass

        config = uvicorn.Config(
            _get_app(),
            host=host,
            port=port,
            log_level=log_level.lower(),
            access_log=False,
        )
        server = uvicorn.Server(config)
        # Hook into uvicorn's startup lifecycle to write the marker as soon as
        # the server is listening (before the first request is processed).
        _orig_startup = server.startup

        async def _patched_startup(sockets=None):
            await _orig_startup(sockets=sockets)
            _write_ready()
            logging.getLogger("autoconduck").info(
                "AutoConduck proxy ready at http://%s:%d (Press CTRL+C to quit)", host, port
            )

        server.startup = _patched_startup
        # On Windows, uvicorn adds SIGBREAK (signal 21 / Ctrl+Break) to its
        # HANDLED_SIGNALS list.  When a sibling console-group process (e.g. OMP)
        # exits or is interrupted, Windows broadcasts CTRL_BREAK_EVENT to every
        # process sharing the same console session — including a foreground
        # `conduck start --headless` invocation.  Uvicorn catches it, sets
        # should_exit, and then re-raises it, which terminates the server with
        # no warning or crash report.
        #
        # Fix: in headless mode ignore SIGBREAK so that only an explicit
        # SIGTERM / SIGINT (i.e. a deliberate "conduck stop" or Ctrl+C in *this*
        # terminal) can shut the server down.  We restore the previous handler
        # after server.run() returns so the process behaves normally again.
        _prev_sigbreak = None
        if sys.platform == "win32" and hasattr(signal, "SIGBREAK"):
            _prev_sigbreak = signal.signal(signal.SIGBREAK, signal.SIG_IGN)
        try:
            server.run()
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            if _prev_sigbreak is not None and hasattr(signal, "SIGBREAK"):
                try:
                    signal.signal(signal.SIGBREAK, _prev_sigbreak)
                except Exception:
                    pass
    except Exception as exc:
        logger.exception("AutoConduck proxy crashed")
        _write_crash_report(exc)
        raise


SUPERVISOR_MAX_RAPID_FAILURES = 5
SUPERVISOR_FAILURE_WINDOW = 60.0
SUPERVISOR_INITIAL_BACKOFF = 1.0
SUPERVISOR_MAX_BACKOFF = 30.0


def _run_supervisor(
    port: int, log_level: str = "info", host: str = "127.0.0.1", child_cmd=None
):
    """Keep the in-process proxy in a child, restarting only crash exits.

    Five failures within one minute are treated as a persistent startup fault;
    giving up lets the normal ensure_server watchdog perform the next revive.
    """
    from autoconduck.launcher import _create_kill_on_close_job, daemon_python

    log_path = home_dir() / "run" / "server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pidfile = config.run_dir() / "server.pid"
    pidfile.parent.mkdir(parents=True, exist_ok=True)
    pidfile.write_text(str(os.getpid()))
    stopping = False
    child = None
    job = _create_kill_on_close_job()
    child_path = config.run_dir() / "child.pid"

    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:
            child.terminate()

    old_handlers = {}
    if hasattr(signal, "SIGTERM"):
        old_handlers[signal.SIGTERM] = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, request_stop)
    if hasattr(signal, "SIGINT"):
        old_handlers[signal.SIGINT] = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, request_stop)

    def restore_handlers():
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)

    failures = []
    backoff = SUPERVISOR_INITIAL_BACKOFF
    try:
        while not stopping:
            command = child_cmd or [
                daemon_python(),
                "-m",
                "autoconduck",
                "start",
                "--headless",
                "--port",
                str(port),
                "--host",
                host,
            ]
            child_env = os.environ.copy()
            child_env["AUTOCONDUCK_SUPERVISED"] = "1"
            started = time.monotonic()
            with log_path.open("ab") as stream:
                flags = 0
                if os.name == "nt":
                    flags = (
                        subprocess.CREATE_NEW_PROCESS_GROUP
                        | subprocess.CREATE_NO_WINDOW
                    )
                child = subprocess.Popen(
                    command,
                    stdout=stream,
                    stderr=stream,
                    close_fds=True,
                    env=child_env,
                    creationflags=flags,
                )
            child_path.parent.mkdir(parents=True, exist_ok=True)
            child_path.write_text(str(child.pid))
            if job is not None and os.name == "nt":
                try:
                    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
                    if not kernel32.AssignProcessToJobObject(job, int(child._handle)):
                        error = ctypes.get_last_error()
                        if error == 5:
                            logging.getLogger(__name__).warning(
                                "could not assign supervised child to job: access denied"
                            )
                        else:
                            logging.getLogger(__name__).warning(
                                "could not assign supervised child to job"
                            )
                except Exception:
                    logging.getLogger(__name__).warning(
                        "could not assign supervised child to job"
                    )
            exit_code = child.wait()
            if stopping:
                return
            now = time.monotonic()
            if now - started > SUPERVISOR_FAILURE_WINDOW:
                failures = []
                backoff = SUPERVISOR_INITIAL_BACKOFF
            failures = [
                when for when in failures if now - when <= SUPERVISOR_FAILURE_WINDOW
            ]
            failures.append(now)
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    f"supervised server exited with code {exit_code}; restart {len(failures)}/{SUPERVISOR_MAX_RAPID_FAILURES}\n"
                )
            logger.warning(
                "supervised server exited with code %s; restart %d/%d",
                exit_code,
                len(failures),
                SUPERVISOR_MAX_RAPID_FAILURES,
            )
            crash_path = home_dir() / "run" / "server.crash"
            crash_path.parent.mkdir(parents=True, exist_ok=True)
            with crash_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    f"supervised child exited with code {exit_code} at "
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')}; see run/server.log for child stderr\n"
                )
            if len(failures) >= SUPERVISOR_MAX_RAPID_FAILURES:
                with log_path.open("a", encoding="utf-8") as stream:
                    stream.write("supervisor giving up after repeated rapid failures\n")
                logger.error(
                    "supervisor giving up after %d failures in %.0fs — run 'autoconduck start' to relaunch",
                    SUPERVISOR_MAX_RAPID_FAILURES,
                    SUPERVISOR_FAILURE_WINDOW,
                )
                return
            time.sleep(backoff)
            backoff = min(backoff * 2, SUPERVISOR_MAX_BACKOFF)
    finally:
        restore_handlers()
        try:
            pidfile.unlink()
        except OSError:
            pass
        try:
            child_path.unlink()
        except OSError:
            pass
        if job is not None:
            try:
                ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(job)
            except Exception:
                pass


def _check_port_available(port: int, host: str = "127.0.0.1") -> None:
    from autoconduck.launcher import (
        find_process_on_port, prompt_kill_port, wait_for_port_free,
        is_port_bindable, stop_server,
    )

    # When running under the internal supervisor, do not kill anything —
    # the supervisor orchestrates restarts externally.
    if os.environ.get("AUTOCONDUCK_SUPERVISED") == "1":
        pid = find_process_on_port(port)
        if pid is not None:
            print(
                f"Port {port} is still in use by PID {pid}; supervised child will not kill it",
                file=sys.stderr,
            )
            raise SystemExit(1)
        # No process found; just give the OS a moment to release the socket.
        if not is_port_bindable(port, host):
            wait_for_port_free(port, host, timeout=1.0)
        return

    pid = find_process_on_port(port)
    if pid is None:
        # Nothing found — give the kernel a brief moment to release resources.
        if not is_port_bindable(port, host):
            wait_for_port_free(port, host, timeout=1.0)
        return

    # Ask the user before tearing down the existing listener.
    if not prompt_kill_port(port, pid):
        raise SystemExit(1)

    # Use the proven stop_server routine (port-scan + kill + 5 s wait).
    stop_server(port)

    # Verify the port is truly free before the caller binds.
    if not wait_for_port_free(port, host, timeout=5.0):
        raise SystemExit(
            f"Port {port} still unavailable after kill; aborting"
        )


def _litellm():
    try:
        import litellm

        return litellm
    except ImportError:
        return None




