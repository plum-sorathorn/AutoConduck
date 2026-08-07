from __future__ import annotations

import argparse
import os
import signal
import socket
import sys
import time
from pathlib import Path

from .config import get_config, load_config, save_config, Config, ensure_home, home_dir
from . import state as state_mod

DEFAULT_PORT = 11434


def _find_free_port(start: int, tries: int = 11) -> int:
    for p in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    return start


def _run_proxy(port: int, log_level: str = "info"):
    import uvicorn
    from .proxy import create_app

    cfg = get_config()
    # ensure port matches
    if port != cfg.port:
        cfg.port = port
        try:
            save_config(cfg)
        except Exception:
            pass
    app = create_app(cfg)
    # flush state on sigterm
    def _handle_sigterm(signum, frame):
        try:
            state_mod.flush()
        except Exception:
            pass
        sys.exit(0)

    try:
        signal.signal(signal.SIGTERM, _handle_sigterm)
        signal.signal(signal.SIGINT, _handle_sigterm)
    except Exception:
        pass

    uvicorn.run(app, host="127.0.0.1", port=port, log_level=log_level, access_log=False)


def cmd_start(args):
    cfg = load_config()
    port = args.port or cfg.port or DEFAULT_PORT
    # port conflict detection
    free = _find_free_port(port)
    if free != port:
        print(f"[autoconduck] port {port} in use, using {free}")
        port = free
    if getattr(args, "headless", False):
        print(f"[autoconduck] starting proxy headless on 127.0.0.1:{port}")
        _run_proxy(port, cfg.log_level)
    else:
        # interactive: launch TUI which spawns proxy
        try:
            from .tui import run_tui

            run_tui(cfg, port)
        except ImportError as e:
            print(f"[autoconduck] TUI not available ({e}), starting headless")
            _run_proxy(port, cfg.log_level)


def cmd_edit(args):
    try:
        from .tui import run_edit

        cfg = load_config()
        run_edit(cfg)
    except ImportError as e:
        print(f"[autoconduck] TUI not available: {e}")
        sys.exit(1)


def cmd_uninstall(args):
    from .agents import all_adapters

    force = getattr(args, "force", False)
    if not force:
        ans = input("Uninstall AutoConduck and restore agent configs? [y/N] ")
        if ans.strip().lower() not in ("y", "yes"):
            print("aborted")
            return
    for ad in all_adapters():
        try:
            ad.revert()
            print(f"  reverted {ad.display_name}")
        except Exception as e:
            print(f"  failed {ad.display_name}: {e}")
    # remove config
    try:
        cfg_path = home_dir() / "config.yaml"
        if cfg_path.exists():
            cfg_path.unlink()
            print(f"removed {cfg_path}")
    except Exception as e:
        print(f"remove config failed: {e}")
    print("[autoconduck] uninstall complete")


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(prog="autoconduck", description="AutoConduck — local model router + orchestrator")
    parser.add_argument("--version", action="store_true", help="show version")
    sub = parser.add_subparsers(dest="cmd")

    # start
    p_start = sub.add_parser("start", help="start proxy")
    p_start.add_argument("--headless", action="store_true", help="headless mode")
    p_start.add_argument("--port", type=int, default=None)
    p_start.add_argument("--host", type=str, default="127.0.0.1")

    # edit
    p_edit = sub.add_parser("edit", help="re-open model selection")

    # uninstall
    p_uninstall = sub.add_parser("uninstall", help="restore configs and remove autoconduck config")
    p_uninstall.add_argument("--force", action="store_true")

    args = parser.parse_args(argv)

    if args.version:
        from . import __version__

        print(__version__)
        return

    if args.cmd == "start":
        cmd_start(args)
    elif args.cmd == "edit":
        cmd_edit(args)
    elif args.cmd == "uninstall":
        cmd_uninstall(args)
    else:
        # default interactive: if no config, onboarding, else dashboard
        cfg_path = home_dir() / "config.yaml"
        cfg = load_config()
        has_models = bool(cfg.models)
        if not cfg_path.exists() or not has_models:
            # onboarding
            try:
                from .tui import run_tui

                run_tui(cfg, cfg.port)
            except ImportError:
                print("[autoconduck] No config yet. Run with models configured or install textual for TUI.")
                print("  Falling back to headless on", cfg.port)
                free = _find_free_port(cfg.port)
                _run_proxy(free, cfg.log_level)
        else:
            try:
                from .tui import run_tui

                run_tui(cfg, cfg.port)
            except ImportError:
                free = _find_free_port(cfg.port)
                _run_proxy(free, cfg.log_level)


if __name__ == "__main__":
    main()
