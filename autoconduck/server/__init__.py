from .server import *
from .server import (
    _impl, _build, _get_app, _run_proxy, _run_supervisor,
    _check_port_available, _find_free_port, _litellm, DEFAULT_PORT
)
from .server_routes import *
from .server_streaming import *
from .messages_api import *
from .messages_models import *
from .messages_sse import *
from . import server, server_routes, server_streaming, messages_api, messages_models, messages_sse
import autoconduck.server.server as _server_mod
import autoconduck.server.server_streaming as _streaming_mod
import autoconduck.server.server_routes as _routes_mod
import autoconduck.server.messages_api as _messages_mod
import autoconduck.server.messages_models as _models_mod
import autoconduck.server.messages_sse as _sse_mod

def __getattr__(name):
    if name == "app":
        return _streaming_mod.app or _server_mod.app
    for mod in (_server_mod, _streaming_mod, _routes_mod, _messages_mod, _models_mod, _sse_mod):
        if hasattr(mod, name):
            return getattr(mod, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
