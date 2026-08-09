"""Installation detection and package-manager commands for the CLI."""
from __future__ import annotations

import importlib.metadata
import os
import sys
from pathlib import Path


def _module_path() -> Path:
    import autoconduck
    return Path(autoconduck.__file__).resolve()


def _is_editable() -> bool:
    try:
        return not _module_path().is_relative_to(Path(sys.prefix).resolve())
    except (ImportError, OSError, ValueError):
        return False


def _is_uv_tool() -> bool:
    return os.sep.join(("uv", "tools")) in str(Path(sys.executable))


def detect_install_method() -> str:
    if os.environ.get("AUTOCONDUCK_WHEEL_DIR"):
        return "npm"
    try:
        uv = _is_uv_tool()
        editable = _is_editable()
    except (ImportError, OSError, ValueError):
        editable = False
        uv = False
    if uv:
        return "uv-tool-editable" if editable else "uv-tool"
    try:
        importlib.metadata.distribution("autoconduck")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"
    return "pip-editable" if editable else "pip"


def upgrade_command(method: str) -> str | None:
    return {
        "uv-tool": "uv tool upgrade autoconduck",
        "npm": "npm install -g autoconduck@latest",
        "pip": "pip install --upgrade autoconduck",
    }.get(method)


def uninstall_hint(method: str) -> str | None:
    if method.startswith("uv-tool"):
        return "uv tool uninstall autoconduck"
    if method.startswith("pip"):
        return "pip uninstall autoconduck"
    if method == "npm":
        return "npm uninstall -g autoconduck"
    return None
