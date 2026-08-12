"""Compatibility facade for the AutoConduck CLI and server."""
from . import cli as _cli
from . import server as _server
from .server import DEFAULT_PORT, app, _run_proxy, _run_supervisor, _check_port_available, _find_free_port, _litellm
from .cli import main, cmd_start, cmd_edit, cmd_reset, cmd_uninstall, cmd_update, cmd_ensure, cmd_release, cmd_stop, cmd_stats, cmd_install, cmd_launch_agent, cmd_tune, purge_home_dir, _timestamp, _open_new_terminal, _run_detached_self_destruct

from .config import load_config, home_dir
import subprocess, shutil, sys, time

_cached = _server._impl._cached
SUPERVISOR_MAX_RAPID_FAILURES = _server._impl.SUPERVISOR_MAX_RAPID_FAILURES
SUPERVISOR_FAILURE_WINDOW = _server._impl.SUPERVISOR_FAILURE_WINDOW
SUPERVISOR_INITIAL_BACKOFF = _server._impl.SUPERVISOR_INITIAL_BACKOFF
SUPERVISOR_MAX_BACKOFF = _server._impl.SUPERVISOR_MAX_BACKOFF


def _build():
    global app
    if app is None:
        _server._impl.app = None
    result = _server._build()
    app = _server.app
    globals().update(_server._impl._cached)
    return result


def _get_app():
    result = _server._get_app()
    global app
    app = result
    globals().update(_server._impl._cached)
    return result


def __getattr__(name):
    return getattr(_server._impl, name)


def _run_supervisor(port, log_level="info", host="127.0.0.1", child_cmd=None):
    for name in ("SUPERVISOR_MAX_RAPID_FAILURES", "SUPERVISOR_FAILURE_WINDOW", "SUPERVISOR_INITIAL_BACKOFF", "SUPERVISOR_MAX_BACKOFF"):
        setattr(_server._impl, name, globals()[name])
    return _server._run_supervisor(port, log_level, host, child_cmd)


async def completions(body, request):
    return await _server._impl._cached["completions"](body, request)


async def messages_endpoint(body, request):
    return await _server._impl._cached["messages_endpoint"](body, request)


def cmd_start(args):
    for name in ("load_config", "home_dir", "_check_port_available", "subprocess", "sys", "time"):
        if name in globals():
            setattr(_cli, name, globals()[name])
    return _cli.cmd_start(args)


def main(argv=None):
    # Keep the historical patch surface working: callers (and integrations) patch
    # names on this facade, while command implementations live in cli.py.
    for name in (
        "load_config", "home_dir", "_find_free_port", "_check_port_available",
        "_run_proxy", "cmd_edit", "cmd_reset", "cmd_uninstall", "cmd_update",
        "cmd_ensure", "cmd_release", "cmd_stop", "cmd_stats", "cmd_launch_agent",
        "cmd_install", "cmd_tune", "_open_new_terminal", "subprocess", "shutil",
    ):
        if name in globals():
            setattr(_cli, name, globals()[name])
    return _cli.main(argv)
