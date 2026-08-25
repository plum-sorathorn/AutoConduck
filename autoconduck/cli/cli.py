"""AutoConduck command-line commands."""
"""AutoConduck CLI and pragmatic FastAPI/LiteLLM hybrid surface."""
import argparse, asyncio, ctypes, json, logging, os, sys, time, subprocess, shutil, signal
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any
from autoconduck import config
from autoconduck.config import get_config, load_config, save_config, home_dir
# ---------- Lightweight modules always imported -------------------------------------------
# Heavy deps (fastapi, pydantic-core, litellm, textual, uvicorn) are deferred until a
# server/CLI command actually needs them.
from autoconduck.server import DEFAULT_PORT, _check_port_available, _find_free_port, _run_proxy, _run_supervisor
from .cli_launch import cmd_launch_agent, cmd_install, _open_new_terminal


def cmd_omp_link(args):
    from autoconduck import launcher
    from autoconduck.harnesses import OmpAdapter

    adapter = OmpAdapter()
    if not adapter.detect():
        print("Oh My Pi was not detected", file=sys.stderr)
        return 1
    cfg = load_config()
    try:
        adapter.patch(cfg, port=getattr(cfg, "port", DEFAULT_PORT))
        launcher.install_shims(["omp"])
        launcher.ensure_path_entry()
    except Exception as exc:
        print(f"failed omp: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_omp_unlink(args):
    from autoconduck import launcher
    from autoconduck.harnesses import OmpAdapter

    try:
        OmpAdapter().revert()
        launcher.uninstall_shims(["omp"])
        cfg = load_config()
        if not cfg.shims:
            launcher.remove_path_entry()
    except Exception as exc:
        print(f"failed omp unlink: {exc}", file=sys.stderr)
        return 1
    return 0

def _invoke_check_port(port: int, host: str = "127.0.0.1") -> None:
    fn = getattr(sys.modules.get(__name__), "_check_port_available", _check_port_available)
    try:
        return fn(port, host)
    except TypeError:
        return fn(port)


def cmd_start(args):
    flags = [
        getattr(args, "claude", False),
        getattr(args, "opencode", False),
        getattr(args, "pi", False),
    ]
    if sum(1 for f in flags if f) > 1:
        print("--claude, --opencode, and --pi cannot be used together", file=sys.stderr)
        raise SystemExit(2)
    new_terminal = getattr(args, "new_terminal", None)
    launch_fn = getattr(sys.modules.get(__name__), "cmd_launch_agent", cmd_launch_agent)
    if getattr(args, "claude", False):
        raise SystemExit(launch_fn("claude_code", new_terminal=new_terminal))
    if getattr(args, "opencode", False):
        raise SystemExit(launch_fn("opencode", new_terminal=new_terminal))
    if getattr(args, "pi", False):
        raise SystemExit(launch_fn("pi", new_terminal=new_terminal))
    cfg = load_config()
    port = args.port or cfg.port or DEFAULT_PORT
    # Agent configuration and shim installation are intentionally opt-in.  They
    # happen only from ``install`` or the explicit onboarding integration step,
    # never as a side effect of starting the API or opening the TUI.
    if getattr(args, "headless", False):
        if getattr(args, "daemon", False):
            _invoke_check_port(port, getattr(args, "host", "127.0.0.1"))
            log = home_dir() / "run" / "server.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            from autoconduck.launcher import daemon_python
            cmd = [
                daemon_python(),
                "-m",
                "autoconduck",
                "start",
                "--headless",
                "--supervisor",
                "--port",
                str(port),
                "--host",
                args.host,
            ]
            with log.open("ab") as stream:
                flags = (
                    (
                        subprocess.DETACHED_PROCESS
                        | subprocess.CREATE_NEW_PROCESS_GROUP
                        | subprocess.CREATE_NO_WINDOW
                    )
                    if sys.platform == "win32"
                    else 0
                )
                child = subprocess.Popen(
                    cmd,
                    stdout=stream,
                    stderr=stream,
                    start_new_session=sys.platform != "win32",
                    creationflags=flags,
                    close_fds=True,
                )
            from autoconduck import launcher
            # Cold starts can include the one-time LiteLLM registry import.
            # Keep the historical 30s floor while allowing slower machines a
            # useful budget instead of reporting a misleading timeout.
            try:
                ready_budget = max(
                    30.0, float(os.environ.get("AUTOCONDUCK_READY_TIMEOUT", "60.0"))
                )
            except ValueError:
                ready_budget = 60.0
            deadline = time.monotonic() + ready_budget
            while time.monotonic() < deadline and not launcher.server_alive(port):
                if child.poll() is not None:
                    break
                time.sleep(0.1)
            if not launcher.server_alive(port):
                if sys.platform == "win32":
                    subprocess.run(
                        ["taskkill", "/PID", str(child.pid), "/T", "/F"],
                        check=False,
                        capture_output=True,
                        creationflags=subprocess.CREATE_NO_WINDOW,
                    )
                else:
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except (OSError, AttributeError):
                        try:
                            child.terminate()
                        except OSError:
                            pass
                try:
                    child.wait(timeout=5)
                except (subprocess.TimeoutExpired, AttributeError):
                    try:
                        child.kill()
                    except OSError:
                        pass
                print(
                    f"AutoConduck daemon failed to become ready within {ready_budget:.0f}s on port {port}",
                    file=sys.stderr,
                )
                return 1
            pidfile, _, _ = launcher._files()
            pidfile.parent.mkdir(parents=True, exist_ok=True)
            pidfile.write_text(str(child.pid))
            # The owner marker makes a manual daemon persistent across shim releases.
            launcher._write_claim(False, owner=True, pid=child.pid)
            return
        if getattr(args, "supervisor", False):
            _run_supervisor(port, cfg.log_level, args.host)
            return
        _invoke_check_port(port, getattr(args, "host", "127.0.0.1"))
        _run_proxy(
            port, cfg.log_level, args.host
        ) if args.host != "127.0.0.1" else _run_proxy(port, cfg.log_level)
    else:
        _invoke_check_port(port, getattr(args, "host", "127.0.0.1"))
        try:
            from autoconduck.tui.app import AutoConduckApp
            is_configured = bool(
                any(
                    isinstance(e, dict) and e.get("enabled", True)
                    for e in config._configured_model_sources(cfg)
                )
                or getattr(cfg, "model_list", [])
                or getattr(cfg, "custom_models", [])
            )
            res = AutoConduckApp(configured=is_configured).run()
            if isinstance(res, str) and res.startswith("launch:"):
                agent_id = res.split("launch:", 1)[1]
                cmd_launch_agent(agent_id)
        except (ImportError, RuntimeError):
            _invoke_check_port(port, getattr(args, "host", "127.0.0.1"))
            _run_proxy(
                port, cfg.log_level, args.host
            ) if args.host != "127.0.0.1" else _run_proxy(port, cfg.log_level)
def cmd_edit(args):
    from autoconduck.tui.app import AutoConduckApp
    cfg = load_config()
    is_configured = bool(
        any(
            isinstance(e, dict) and e.get("enabled", True)
            for e in config._configured_model_sources(cfg)
        )
        or getattr(cfg, "model_list", [])
        or getattr(cfg, "custom_models", [])
    )
    getattr(AutoConduckApp(configured=is_configured, initial_screen="edit"), "run")()
def cmd_reset(args):
    if not getattr(args, "force", False) and input(
        "Reset AutoConduck, stop the daemon, revert coding agent configurations, and delete state under autoconduck home? [y/N] "
    ).lower() not in ("y", "yes"):
        return
    cfg = load_config()
    from autoconduck.harnesses import all_adapters
    from autoconduck import launcher, update
    try:
        launcher.stop_server(getattr(cfg, "port", None) or DEFAULT_PORT)
    except Exception as exc:
        print(f"warning: could not stop daemon: {exc}")
    reverted = []
    for adapter in all_adapters():
        try:
            paths = [p for p in adapter.config_paths() if p.exists()]
            adapter.revert()
            reverted.append(
                f"  [OK] Reverted {adapter.display_name}"
                + (f" ({', '.join(str(p) for p in paths)})" if paths else "")
            )
        except Exception as exc:
            print(f"  [FAIL] Failed {adapter.display_name}: {exc}")
    launcher.uninstall_shims()
    launcher.remove_path_entry()
    purge_home_dir(home_dir())
    print("\nCoding agents reverted:")
    for msg in reverted:
        print(msg)
    
    is_uninstall = getattr(args, "cmd", "") == "uninstall"
    if not is_uninstall:
        print("\nAutoConduck state purged; package remains installed.")
        hint = update.uninstall_hint(update.detect_install_method())
        if hint:
            print(f"Package still installed — remove it with: {hint}")
    else:
        print("\nAutoConduck state purged.")
def _run_detached_self_destruct(command_args, cwd=None):
    import sys, os, subprocess
    if sys.platform == "win32":
        cmd_str = " ".join(command_args)
        script = f"ping 127.0.0.1 -n 2 > nul & {cmd_str}"
        subprocess.Popen(
            ["cmd.exe", "/c", script],
            creationflags=0x08000000,
            cwd=cwd
        )
        sys.exit(0)
    else:
        os.execvp(command_args[0], command_args)
def cmd_uninstall(args):
    # Capture port before cmd_reset purges the home dir.
    try:
        _pre_cfg = load_config()
        _port = getattr(_pre_cfg, "port", None) or DEFAULT_PORT
    except Exception:
        _port = DEFAULT_PORT
    cmd_reset(args)
    from autoconduck import update, launcher
    # Kill any daemon still alive on the port (pidfile may already be gone).
    try:
        launcher.stop_server(_port)
        launcher.kill_existing_on_port(_port)
    except Exception:
        pass
    method = update.detect_install_method()
    command = update.uninstall_hint(method)
    if not command:
        return
    print(f"Uninstalling package via: {command}")
    tool = shutil.which(command.split()[0])
    full_cmd = [tool or command.split()[0], *command.split()[1:]]
    # When installed as a uv tool the current process is loaded from the very
    # directory uv wants to delete.  Running the uninstall synchronously
    # therefore fails with "Access is denied".  Detach a small shell snippet
    # that waits for us to exit before invoking uv, so the Lib/ directory is
    # released in time.
    if method.startswith("uv-tool") and tool:
        _run_detached_self_destruct(full_cmd)
        # _run_detached_self_destruct calls sys.exit() on Windows and
        # os.execvp() on POSIX — this line is never reached.
        return
    if tool:
        subprocess.run(full_cmd, check=False)
    else:
        subprocess.run(command, shell=True, check=False)
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
    from autoconduck import __version__, update, launcher
    method = update.detect_install_method()
    command = update.upgrade_command(method)
    print(f"Current version: {__version__}")
    if command is None:
        print(
            "No managed installation detected; update the checkout manually (git pull) and reinstall."
        )
        return
    if args.dry_run:
        print(f"Would run: {command}")
        return
    tool = shutil.which(command.split()[0])
    if not tool:
        print(
            f"Error: required package manager '{command.split()[0]}' was not found on PATH."
        )
        return
    cwd = None
    if "editable" in method:
        source_dir = update._module_path().parent.parent
        if (source_dir / "pyproject.toml").exists():
            cwd = str(source_dir)
        elif (Path.cwd() / "pyproject.toml").exists():
            cwd = str(Path.cwd())
    cfg = load_config()
    port = getattr(cfg, "port", None) or DEFAULT_PORT
    try:
        launcher.stop_server(port)
        launcher.kill_existing_on_port(port)
    except Exception:
        pass
        
    print(f"Running upgrade: {command}")
    import os, sys
    if os.name == "nt":
        # Windows locks .pyd files loaded by the current process.
        # We use os.execv to replace the current Python process with 'uv'
        # This safely releases all loaded .pyd file locks while keeping the output in the same console.
        print(f"Running upgrade in-place: {command}")
        uv_path = shutil.which(tool)
        if uv_path:
            os.execv(uv_path, [tool] + command.split()[1:])
        sys.exit(0)
        
    res = subprocess.run([tool, *command.split()[1:]], cwd=cwd, check=False)
    if res.returncode == 0:
        print("Upgrade completed successfully. Run autoconduck --version to confirm.")
    else:
        print(f"Upgrade finished with exit code {res.returncode}.")

def cmd_ensure(args):
    from autoconduck import launcher
    launcher.ensure_server(args.port, getattr(args, "client_id", None))
def cmd_release(args):
    from autoconduck import launcher
    launcher.release_server(args.port, getattr(args, "client_id", None))
def cmd_stop(args):
    from autoconduck import launcher
    launcher.stop_server(args.port)
    from autoconduck.harnesses.claude_code import ClaudeCodeAdapter
    ClaudeCodeAdapter().revert()
def cmd_stats(args):
    from autoconduck import stats
    records = stats.load_records()
    if args.days is not None:
        cutoff = time.time() - args.days * 86400
        records = [r for r in records if _timestamp(r.get("ts")) >= cutoff]
    if args.reset:
        if not args.force:
            print("Refusing to reset stats without --force")
            return
        try:
            stats.stats_path().unlink()
        except FileNotFoundError:
            pass
        return
    agg = stats.aggregate(records)
    if args.json:
        print(stats.render_json(agg))
        return
    first, last = (
        (records[0].get("ts"), records[-1].get("ts")) if records else ("n/a", "n/a")
    )
    print(f"Usage stats: {first} to {last} ({agg['totals']['calls']} calls)")
    print(stats.render_table(agg))
def _timestamp(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError, OverflowError):
        return 0
def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(prog="autoconduck")
    parser.add_argument("--version", action="store_true")
    sub = parser.add_subparsers(dest="cmd")
    start = sub.add_parser("start")
    start.add_argument("--headless", action="store_true")
    start.add_argument("--daemon", action="store_true")
    start.add_argument("--supervisor", action="store_true", help=argparse.SUPPRESS)
    start.add_argument("--port", type=int)
    start.add_argument("--host", default="127.0.0.1")
    start.add_argument("--claude", action="store_true")
    start.add_argument("--opencode", action="store_true")
    start.add_argument("--pi", action="store_true")
    start.add_argument(
        "--new-terminal",
        action="store_true",
        default=None,
        help="Launch proxy in a new terminal window (overrides config launch_in_new_terminal)",
    )
    for name, func in (
        ("ensure", cmd_ensure),
        ("release", cmd_release),
        ("stop", cmd_stop),
    ):
        p = sub.add_parser(name)
        p.add_argument("--port", type=int)
        p.add_argument("--client-id", type=str, default=None)
        p.set_defaults(handler=func)
    stats_parser = sub.add_parser("stats")
    stats_parser.add_argument("--json", action="store_true")
    stats_parser.add_argument("--days", type=int)
    stats_parser.add_argument("--reset", action="store_true")
    stats_parser.add_argument("--force", action="store_true")
    stats_parser.set_defaults(handler=cmd_stats)
    install = sub.add_parser("install")
    install.add_argument("agents", nargs="*")
    install.set_defaults(handler=cmd_install)
    omp = sub.add_parser("omp")
    omp_sub = omp.add_subparsers(dest="omp_cmd", required=True)
    omp_link = omp_sub.add_parser("link")
    omp_link.set_defaults(handler=cmd_omp_link)
    omp_unlink = omp_sub.add_parser("unlink")
    omp_unlink.set_defaults(handler=cmd_omp_unlink)
    upd = sub.add_parser("update")
    upd.add_argument("--dry-run", action="store_true")
    upd.set_defaults(handler=cmd_update)
    sub.add_parser("edit")
    reset_parser = sub.add_parser("reset")
    reset_parser.add_argument("--force", action="store_true")
    reset_parser.set_defaults(handler=cmd_reset)
    uninstall = sub.add_parser("uninstall")
    uninstall.add_argument("--force", action="store_true")
    uninstall.set_defaults(handler=cmd_uninstall)
    try:
        args = parser.parse_args(argv)
        if args.version:
            from autoconduck import __version__
            print(__version__)
        elif args.cmd == "start":
            cmd_start(args)
        elif args.cmd == "edit":
            cmd_edit(args)
        elif args.cmd == "reset":
            cmd_reset(args)
        elif args.cmd == "uninstall":
            cmd_uninstall(args)
        elif hasattr(args, "handler"):
            args.handler(args)
        else:
            cmd_start(argparse.Namespace(headless=False, port=None, host="127.0.0.1"))
    except (KeyboardInterrupt, asyncio.CancelledError):
        return 0
from autoconduck.server import DEFAULT_PORT, _check_port_available, _find_free_port, _run_proxy, _run_supervisor
from .cli_launch import cmd_launch_agent, cmd_install, _open_new_terminal
