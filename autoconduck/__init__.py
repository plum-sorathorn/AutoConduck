__version__ = "0.3.4"

import importlib

_LAZY_MODULES = {
    "_compat": "._compat", "auth": ".auth", "cli": ".cli",
    "harnesses": ".harnesses", "launcher": ".launcher",
    "orchestrator": ".orchestrator", "presets": ".presets",
    "routing": ".routing", "server": ".server", "tui": ".tui",
    "server_streaming": ".server.server_streaming", "messages_api": ".server.messages_api",
    "messages_models": ".server.messages_models", "messages_sse": ".server.messages_sse",
    "providers": ".auth.providers",
    "dispatcher": ".routing.dispatcher",
}


import sys
import types


def __getattr__(name):
    if name == "agents":
        return __getattr__("harnesses")
    module_name = _LAZY_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name, __name__)
    globals()[name] = module
    sys.modules[f"{__name__}.{name}"] = module
    return module


def _register_lazy_aliases():
    """Pre-register placeholder sys.modules entries for legacy FLAT aliases whose
    dotted target is nested (e.g. "server_streaming" -> ".server.server_streaming")
    so `import autoconduck.<name>` / `from autoconduck.<name> import X` resolve
    without eagerly importing the (possibly heavy) real submodule. The import
    machinery checks sys.modules for the dotted name right after importing this
    parent package and before consulting any finder, so the alias must exist
    here, not only via __getattr__.

    Real on-disk subpackages (e.g. "auth", "routing", "server") are intentionally
    skipped: they already resolve via the normal filesystem-based import system,
    and pre-registering a placeholder for them would shadow the real package in
    sys.modules and break its own submodule resolution (e.g. autoconduck.server
    importing autoconduck.server.server_streaming).
    """
    for alias_name, target_name in _LAZY_MODULES.items():
        if target_name == f".{alias_name}":
            continue  # real subpackage; normal import already works
        full_alias = f"{__name__}.{alias_name}"
        placeholder = types.ModuleType(full_alias)

        def _placeholder_getattr(attr, _alias_name=alias_name, _target_name=target_name, _full_alias=full_alias):
            real = importlib.import_module(_target_name, __name__)
            sys.modules[_full_alias] = real
            globals()[_alias_name] = real
            return getattr(real, attr)

        placeholder.__getattr__ = _placeholder_getattr
        sys.modules[full_alias] = placeholder


_register_lazy_aliases()
