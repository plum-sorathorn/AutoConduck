"""Server compatibility facade."""
from autoconduck.server import server_streaming as _impl
DEFAULT_PORT = _impl.DEFAULT_PORT
app = None
def _build():
 global app
 result = _impl._build()
 app = _impl.app
 return result
def _get_app():
 global app
 app = _impl._get_app()
 return app
def _run_proxy(port, log_level="info", host="127.0.0.1"): return _impl._run_proxy(port, log_level, host)
def _run_supervisor(port, log_level="info", host="127.0.0.1", child_cmd=None): return _impl._run_supervisor(port, log_level, host, child_cmd)
def _check_port_available(port): return _impl._check_port_available(port)
def _find_free_port(start, tries=11): return _impl._find_free_port(start, tries)
def _litellm(): return _impl._litellm()
def __getattr__(name):
 if name == "app":
  return _impl.app
 return getattr(_impl, name)
