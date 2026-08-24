"""Bounded, workspace-scoped tools for the executor model."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


class ToolError(Exception):
    pass


class ScopeViolation(ToolError):
    pass


def _schema(name: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "parameters": {"type": "object", "properties": properties, "required": required}}}


_string = {"type": "string"}
TOOL_SCHEMAS = [
    _schema("read", {"path": _string}, ["path"]),
    _schema("grep", {"pattern": _string, "path": _string}, ["pattern", "path"]),
    _schema("glob", {"pattern": _string}, ["pattern"]),
    _schema("list", {"path": _string}, ["path"]),
    _schema("edit", {"path": _string, "old_string": _string, "new_string": _string}, ["path", "old_string", "new_string"]),
    _schema("write", {"path": _string, "content": _string}, ["path", "content"]),
    _schema("bash", {"command": _string}, ["command"]),
]

READ_ONLY_TOOLS = frozenset({"read", "grep", "glob", "list"})


def is_read_only_tool(name: str) -> bool:
    """Return whether a tool only inspects the workspace."""
    return name in READ_ONLY_TOOLS


def tool_model(name: str, current_model: str, cfg) -> str:
    """Choose the model for a tool continuation.

    Workspace inspection does not need the executor's potentially expensive
    model.  Reuse the normal FAST selector so provider and catalog rules stay
    in one place; mutations remain on the executor model.
    """
    if not is_read_only_tool(name):
        return current_model
    try:
        from autoconduck.routing.dispatcher import pick_fast_model

        return pick_fast_model("autoconduck", cfg)
    except Exception:
        return current_model


def _resolve_safe(workspace_root: Path, rel_path: str) -> Path:
    root = workspace_root.resolve()
    target = (root / rel_path).resolve()
    if not target.is_relative_to(root):
        raise ScopeViolation(f"path outside workspace: {rel_path}")
    return target


def _check_scope(workspace_root, allowed_scope: list[str], rel_path: str) -> None:
    if not allowed_scope:
        raise ScopeViolation("no declared scope for edits")
    target = _resolve_safe(Path(workspace_root), rel_path)
    root = Path(workspace_root).resolve()
    allowed = [_resolve_safe(root, entry) for entry in allowed_scope]
    if not any(target == entry or target.is_relative_to(entry) for entry in allowed):
        raise ScopeViolation(f"path outside allowed scope: {rel_path}")


def tool_read(workspace_root: Path, path: str, *, max_bytes: int) -> str:
    target = _resolve_safe(workspace_root, path)
    try:
        data = target.read_bytes()
    except OSError as exc:
        raise ToolError(str(exc)) from exc
    if len(data) > max_bytes:
        return data[:max_bytes].decode("utf-8") + "\n[...truncated]"
    return data.decode("utf-8")


def tool_grep(workspace_root: Path, pattern: str, path: str = ".") -> str:
    target = _resolve_safe(workspace_root, path)
    try:
        regex = re.compile(pattern)
        files = [target] if target.is_file() else (p for p in target.rglob("*") if p.is_file())
        results = []
        for file in files:
            for number, line in enumerate(file.read_text(encoding="utf-8").splitlines(), 1):
                if regex.search(line):
                    results.append(f"{file.relative_to(workspace_root.resolve())}:{number}")
                    if len(results) >= 200:
                        return "\n".join(results)
        return "\n".join(results)
    except OSError as exc:
        raise ToolError(str(exc)) from exc


def tool_glob(workspace_root: Path, pattern: str) -> str:
    root = workspace_root.resolve()
    return "\n".join(str(p.relative_to(root)) for p in list(root.glob(pattern))[:200])


def tool_list(workspace_root: Path, path: str = ".") -> str:
    target = _resolve_safe(workspace_root, path)
    try:
        return "\n".join(sorted(p.name for p in target.iterdir()))
    except OSError as exc:
        raise ToolError(str(exc)) from exc


def tool_edit(workspace_root: Path, allowed_scope: list[str], path: str, old_string: str, new_string: str) -> str:
    _check_scope(workspace_root, allowed_scope, path)
    target = _resolve_safe(workspace_root, path)
    try:
        text = target.read_text(encoding="utf-8")
        if text.count(old_string) != 1:
            raise ToolError("old_string not found or not unique")
        target.write_text(text.replace(old_string, new_string), encoding="utf-8")
    except OSError as exc:
        raise ToolError(str(exc)) from exc
    return f"edited {path}"


def tool_write(workspace_root: Path, allowed_scope: list[str], path: str, content: str) -> str:
    if not allowed_scope:
        raise ScopeViolation("no declared scope for edits")
    target = _resolve_safe(workspace_root, path)
    if target.exists():
        _check_scope(workspace_root, allowed_scope, path)
    else:
        parent = target.parent
        if not any(parent == _resolve_safe(workspace_root, entry) or parent.is_relative_to(_resolve_safe(workspace_root, entry)) for entry in allowed_scope):
            raise ScopeViolation(f"path outside allowed scope: {path}")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise ToolError(str(exc)) from exc
    return f"wrote {path}"


def tool_bash(workspace_root: Path, command: str, *, enabled: bool) -> str:
    if not enabled:
        return "ERROR: bash tool disabled"
    try:
        result = subprocess.run(command, shell=True, cwd=workspace_root, timeout=30.0, capture_output=True, text=True)
    except subprocess.TimeoutExpired as exc:
        raise ToolError(str(exc)) from exc
    output = result.stdout + (result.stderr if result.returncode else "")
    return output[:10000]


def execute_tool(name: str, args: dict, *, workspace_root, allowed_scope, cfg) -> str:
    try:
        selection = getattr(cfg, "selection", None)
        dispatch = {
            "read": lambda: tool_read(workspace_root, args["path"], max_bytes=getattr(selection, "executor_max_read_bytes", 200_000)),
            "grep": lambda: tool_grep(workspace_root, args["pattern"], args.get("path", ".")),
            "glob": lambda: tool_glob(workspace_root, args["pattern"]),
            "list": lambda: tool_list(workspace_root, args.get("path", ".")),
            "edit": lambda: tool_edit(workspace_root, allowed_scope, args["path"], args["old_string"], args["new_string"]),
            "write": lambda: tool_write(workspace_root, allowed_scope, args["path"], args["content"]),
            "bash": lambda: tool_bash(workspace_root, args["command"], enabled=getattr(selection, "executor_enable_bash", False)),
        }
        return dispatch[name]()
    except Exception as exc:
        return f"ERROR: {exc}"
