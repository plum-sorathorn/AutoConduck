"""Server lifecycle and application facade."""

from __future__ import annotations

from typing import Any
from autoconduck.server import server_streaming as _impl

DEFAULT_PORT = _impl.DEFAULT_PORT
app = None


def _build() -> Any:
    global app
    result = _impl._build()
    app = _impl.app
    return result


def _get_app() -> Any:
    global app
    app = _impl._get_app()
    return app


def _run_proxy(port: int, log_level: str = "info", host: str = "127.0.0.1") -> None:
    return _impl._run_proxy(port, log_level, host)


def _run_supervisor(
    port: int,
    log_level: str = "info",
    host: str = "127.0.0.1",
    child_cmd: list[str] | None = None,
) -> None:
    return _impl._run_supervisor(port, log_level, host, child_cmd)


def _check_port_available(port: int, host: str = "127.0.0.1") -> Any:
    return _impl._check_port_available(port, host)


def _find_free_port(start: int, tries: int = 11) -> int:
    return _impl._find_free_port(start, tries)


def _litellm() -> Any:
    return _impl._litellm()


def __getattr__(name: str) -> Any:
    if name == "app":
        return _impl.app
    return getattr(_impl, name)
