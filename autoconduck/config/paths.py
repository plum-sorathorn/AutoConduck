"""Filesystem and directory path resolution utilities for AutoConduck."""

from __future__ import annotations

import os
from pathlib import Path


def home_dir() -> Path:
    """Return the base AutoConduck home directory (defaults to ~/.autoconduck)."""
    return Path(os.environ.get("AUTOCONDUCK_HOME", Path.home() / ".autoconduck"))


def config_path(path: str | Path | None = None) -> Path:
    """Return the active path to config.yaml."""
    if path:
        return Path(path)
    structured = home_dir() / "config" / "config.yaml"
    if structured.exists():
        return structured
    return home_dir() / "config.yaml"


def backups_dir(agent: str | None = None) -> Path:
    """Return the backup directory for configs or agent shims."""
    path = home_dir() / "backups"
    return path / agent if agent else path


def logs_path() -> Path:
    """Return the active log file path."""
    structured = home_dir() / "logs" / "autoconduck.log"
    if structured.exists() or (home_dir() / "logs").exists():
        return structured
    return home_dir() / "autoconduck.log"


def run_dir() -> Path:
    """Return the runtime runtime process/PID directory."""
    return home_dir() / "run"
