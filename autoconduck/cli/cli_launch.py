"""Agent installation and launch CLI commands."""
from __future__ import annotations
import os, sys, subprocess, shutil, json, signal, time
from pathlib import Path
from autoconduck import config
from autoconduck.config import get_config, load_config, save_config, home_dir
from autoconduck.server import DEFAULT_PORT
AGENT_ALIASES: dict[str, str] = {
    "claude": "claude_code",
    "claude-code": "claude_code",
    "claude_code": "claude_code",
    "opencode": "opencode",
    "open-code": "opencode",
    "pi": "pi",
}


def resolve_agent_ids(requested: list[str] | None) -> list[str]:
    from autoconduck import launcher
    from autoconduck.harnesses import all_adapters

    adapters = all_adapters()
    if not requested:
        return [a.id for a in adapters if launcher.real_binary_path(a.id) or a.id in ("claude_code", "opencode", "pi")]

    resolved: list[str] = []
    for raw in requested:
        key = raw.strip().lower()
        if key in ("all", "*"):
            return [a.id for a in adapters]
        aid = AGENT_ALIASES.get(key, key)
        if any(a.id == aid for a in adapters):
            if aid not in resolved:
                resolved.append(aid)
        else:
            print(f"warning: unknown coding agent '{raw}'", file=sys.stderr)
    return resolved


def cmd_install(args):
    from autoconduck import launcher
    from autoconduck.harnesses import all_adapters

    adapters = all_adapters()
    selected = resolve_agent_ids(getattr(args, "agents", None))
    cfg = load_config()
    port = getattr(cfg, "port", DEFAULT_PORT)

    for aid in selected:
        adapter = next((a for a in adapters if a.id == aid), None)
        if adapter is None:
            continue
        try:
            adapter.patch(cfg, port=port)
            installed_feats = adapter.install_features()
            if installed_feats:
                print(f"{aid}: installed features: {', '.join(installed_feats)}")
        except Exception as exc:
            print(f"failed {aid}: {exc}", file=sys.stderr)

    paths = launcher.install_shims(selected)
    modified = launcher.ensure_path_entry()
    for aid, path in paths.items():
        print(f"{aid}: {path}")
    if modified:
        print(f"PATH: {modified}")

def _open_new_terminal(cmd: list[str], log: "Path") -> None:  # noqa: F821
    """Open the proxy daemon in a new visible terminal window (best effort).

    Tries Windows Terminal, then conhost, then falls back to a windowless
    background process with output redirected to *log*.
    """
    if sys.platform == "win32":
        import subprocess as _sp

        # Prefer Windows Terminal (wt.exe) if available — provides a proper tab.
        if shutil.which("wt"):
            try:
                _sp.Popen(
                    ["wt", "new-tab", "--title", "AutoConduck", "--"] + cmd,
                    creationflags=_sp.CREATE_NEW_PROCESS_GROUP,
                    close_fds=True,
                )
                return
            except Exception:
                pass
        # Fall back to a plain cmd.exe window.
        try:
            _sp.Popen(
                ["cmd", "/c", "start", "AutoConduck proxy"] + cmd,
                creationflags=_sp.CREATE_NEW_PROCESS_GROUP,
                close_fds=True,
            )
            return
        except Exception:
            pass
    elif sys.platform == "darwin":
        import subprocess as _sp

        try:
            script = "tell application \"Terminal\" to do script \"" + " ".join(cmd) + "\""
            _sp.Popen(["osascript", "-e", script], close_fds=True)
            return
        except Exception:
            pass
    else:
        import subprocess as _sp

        for term in ("gnome-terminal", "xterm", "konsole", "xfce4-terminal"):
            if shutil.which(term):
                try:
                    _sp.Popen(
                        [term, "--"] + cmd,
                        close_fds=True,
                    )
                    return
                except Exception:
                    continue
    # Fallback: background process, log redirected (same as normal mode)
    import subprocess as _sp
    with log.open("ab") as stream:
        flags = (
            (_sp.DETACHED_PROCESS | _sp.CREATE_NEW_PROCESS_GROUP | _sp.CREATE_NO_WINDOW)
            if sys.platform == "win32"
            else 0
        )
        _sp.Popen(
            cmd,
            stdout=stream,
            stderr=stream,
            start_new_session=sys.platform != "win32",
            creationflags=flags,
            close_fds=True,
        )

def cmd_launch_agent(agent_id: str, port: int | None = None, new_terminal: bool | None = None) -> int:
    """Launch an agent by ID through the AutoConduck proxy server.

    Strategy: always kill any existing process on the port, start a fresh
    daemon, wait for readiness with exponential-backoff polling, then launch
    the agent binary.  Heavy imports (fastapi, litellm, textual) are deferred
    into the daemon child so this parent process stays lean.

    When *new_terminal* is True (or config.launch_in_new_terminal is True),
    the proxy daemon is launched in a new visible terminal window so the
    caller's terminal stays clean for the agent.
    """
    from autoconduck import launcher
    from autoconduck.harnesses import all_adapters

    # Find the adapter by ID
    adapter = next((a for a in all_adapters() if a.id == agent_id), None)
    if adapter is None:
        print(f"unknown agent '{agent_id}'", file=sys.stderr)
        return 1

    cfg = load_config()
    port = int(port or getattr(cfg, "port", None) or DEFAULT_PORT)

    # Resolve new-terminal setting: CLI flag > config value
    use_new_terminal = (
        new_terminal if new_terminal is not None else bool(getattr(cfg, "launch_in_new_terminal", False))
    )

    # Reuse a healthy manual daemon; otherwise trust the port check below
    # to find-and-kill whatever is listening on the port.
    reused = launcher.server_alive(port)
    if reused:
        launcher._write_claim(False)

    log = home_dir() / "run" / "server.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    # Readiness sentinel: proxy writes this file when it is serving.
    ready_marker = home_dir() / "run" / f"server_{port}.ready"
    try:
        ready_marker.unlink(missing_ok=True)
    except Exception:
        pass

    python_bin = launcher.daemon_python()
    cmd = [python_bin, "-m", "autoconduck", "start", "--headless", "--port", str(port)]

    if use_new_terminal:
        print(f"AutoConduck: launching proxy in a new terminal window (port {port})…")
        _open_new_terminal(cmd, log)
        proc = None
    else:
        import subprocess as _sp

        flags = (
            (_sp.DETACHED_PROCESS | _sp.CREATE_NEW_PROCESS_GROUP | _sp.CREATE_NO_WINDOW)
            if sys.platform == "win32"
            else 0
        )
        with log.open("ab") as stream:
            proc = _sp.Popen(
                cmd,
                stdout=stream,
                stderr=stream,
                start_new_session=sys.platform != "win32",
                creationflags=flags,
                close_fds=True,
            )

    # Exponential-backoff health poll with a configurable cold-start budget.
    # On first install, LiteLLM's model registry import can take 30–90 s.
    # We first check the cheap marker file, then fall back to HTTP polling.
    server_ready = False
    try:
        ready_budget = max(
            30.0, float(os.environ.get("AUTOCONDUCK_READY_TIMEOUT", "90.0"))
        )
    except ValueError:
        ready_budget = 90.0
    deadline = time.monotonic() + ready_budget
    attempt = 0
    while time.monotonic() < deadline:
        # Fast check: sentinel file written by the proxy at startup
        if ready_marker.exists():
            server_ready = True
            break
        # HTTP fallback: covers cases where the proxy started but did not write the file
        try:
            from urllib.request import urlopen

            with urlopen(
                f"http://127.0.0.1:{port}/healthz",
                timeout=min(0.5, max(0.01, deadline - time.monotonic())),
            ):
                server_ready = True
                break
        except Exception:
            pass
        if proc is not None and proc.poll() is not None:
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(0.15 * (1.5**attempt), 0.8, remaining))
        attempt += 1

    if not server_ready:
        # Do not leave a detached cold-start daemon behind when readiness fails.
        if proc is not None:
            import subprocess as _sp

            if sys.platform == "win32":
                _sp.run(
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    check=False,
                    capture_output=True,
                    creationflags=_sp.CREATE_NO_WINDOW,
                )
            else:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except (OSError, AttributeError):
                    proc.terminate()
        print(
            f"error: server did not become ready within {ready_budget:.0f} s — "
            f"check {log} for details",
            file=sys.stderr,
        )
        return 1

    if not reused:
        launcher._write_claim(True)

    # Patch the adapter
    try:
        # Adapters may optionally accept the launch port (Pi does); the base
        # adapter contract predates that optional argument.
        adapter.patch(cfg, port=port)  # type: ignore[call-arg]
    except TypeError:
        # Fallback for adapters that don't accept port argument (e.g., ClaudeCodeAdapter)
        adapter.patch(cfg)

    real_bin = launcher.real_binary_path(agent_id)
    binary_name = getattr(adapter, "binary_name", None)
    if not real_bin and binary_name:
        real_bin = shutil.which(binary_name)

    if not real_bin:
        print(
            f"agent '{agent_id}' not found on PATH; run: autoconduck install {agent_id}"
        )
        launcher.release_server(port)
        return 1

    env = os.environ.copy()
    pseudo = getattr(cfg, "pseudo_model", "autoconduck") or "autoconduck"
    if agent_id == "claude_code":
        env.update(
            launcher._claude_env(port, pseudo)
        )
    elif agent_id == "opencode":
        env["OPENAI_BASE_URL"] = f"http://127.0.0.1:{port}/v1"
        env["OPENAI_API_KEY"] = "autoconduck-local"
        env["OPENCODE_MODEL"] = f"autoconduck/{pseudo}"
    elif agent_id == "pi":
        env["AI_AGENT"] = "pi"
        env["PI_CODING_AGENT"] = "true"
    else:
        env["OPENAI_BASE_URL"] = f"http://127.0.0.1:{port}/v1"
        env["OPENAI_API_BASE"] = f"http://127.0.0.1:{port}/v1"
        env["OPENAI_API_KEY"] = "autoconduck-local"

    import subprocess as _sp

    print(
        f"AutoConduck ready at http://127.0.0.1:{port} — launching {adapter.binary_name}"
    )
    try:
        return _sp.run([real_bin], env=env).returncode
    finally:
        launcher.release_server(port)

def cmd_tune(args):
    """Launch tuning UI, with a useful deterministic fallback."""
    try:
        from autoconduck.tui.app import AutoConduckApp

        mode = getattr(args, "mode", None) or "select"
        app = AutoConduckApp(configured=True, tune_mode=mode)
        getattr(app, "run")()
    except (ImportError, RuntimeError):
        cfg = get_config()
        print("AutoConduck tuning is unavailable without Textual.")
        print(cfg.selection.model_dump())
