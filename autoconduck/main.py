"""Main entrypoint and compatibility layer for the AutoConduck CLI and server."""

import shutil
import subprocess
import sys
import time
from typing import Any

from . import cli as _cli
from . import server as _server
from .cli import (
    _open_new_terminal,
    _run_detached_self_destruct,
    _timestamp,
    cmd_edit,
    cmd_ensure,
    cmd_install,
    cmd_launch_agent,
    cmd_release,
    cmd_reset,
    cmd_start,
    cmd_stats,
    cmd_stop,
    cmd_tune,
    cmd_uninstall,
    cmd_update,
    main,
    purge_home_dir,
)
from .config import home_dir, load_config
from .server import (
    DEFAULT_PORT,
    _check_port_available,
    _find_free_port,
    _litellm,
    _run_proxy,
    _run_supervisor,
    app,
)

SUPERVISOR_MAX_RAPID_FAILURES = 5
SUPERVISOR_FAILURE_WINDOW = 60.0
SUPERVISOR_INITIAL_BACKOFF = 1.0
SUPERVISOR_MAX_BACKOFF = 30.0


def _build():
    global app
    _server._impl.app = None
    result = _server._build()
    app = _server.app or _server._impl.app
    globals().update(_server._impl._cached)
    return result


def _get_app():
    global app
    result = _server._get_app()
    app = _server.app or _server._impl.app
    globals().update(_server._impl._cached)
    return result


def __getattr__(name):
    return getattr(_server._impl, name)


def _run_supervisor(port, log_level="info", host="127.0.0.1", child_cmd=None):
    for name in (
        "SUPERVISOR_MAX_RAPID_FAILURES",
        "SUPERVISOR_FAILURE_WINDOW",
        "SUPERVISOR_INITIAL_BACKOFF",
        "SUPERVISOR_MAX_BACKOFF",
    ):
        setattr(_server._impl, name, globals()[name])
    return _server._run_supervisor(port, log_level, host, child_cmd)


async def completions(body, request):
    return await _server._impl._cached["completions"](body, request)


async def messages_endpoint(body, request):
    return await _server._impl._cached["messages_endpoint"](body, request)


def cmd_start(args):
    for name in (
        "load_config",
        "home_dir",
        "_check_port_available",
        "subprocess",
        "sys",
        "time",
    ):
        if name in globals():
            setattr(_cli, name, globals()[name])
            if hasattr(_cli, "cli"):
                setattr(_cli.cli, name, globals()[name])
    return _cli.cmd_start(args)


def main(argv=None):
    for name in (
        "load_config",
        "home_dir",
        "_find_free_port",
        "_check_port_available",
        "_run_proxy",
        "cmd_edit",
        "cmd_reset",
        "cmd_uninstall",
        "cmd_update",
        "cmd_ensure",
        "cmd_release",
        "cmd_stop",
        "cmd_stats",
        "cmd_launch_agent",
        "cmd_install",
        "cmd_tune",
        "_open_new_terminal",
        "subprocess",
        "shutil",
    ):
        if name in globals():
            setattr(_cli, name, globals()[name])
            if hasattr(_cli, "cli"):
                setattr(_cli.cli, name, globals()[name])
    return _cli.main(argv)


if __name__ == "__main__":
    sys.exit(main())
