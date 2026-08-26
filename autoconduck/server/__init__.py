"""Lazy server package facade."""

from . import server
from .server import (
    DEFAULT_PORT, _build, _check_port_available, _find_free_port, _get_app,
    _impl, _litellm, _run_proxy, _run_supervisor,
)
from . import server_streaming


def __getattr__(name):
    if name == "app":
        return server._impl.app or server.app
    from . import server_streaming
    if hasattr(server_streaming, name):
        return getattr(server_streaming, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
