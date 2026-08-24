"""TUI dashboard rendering helpers, box formatting, and log visualizers."""

from __future__ import annotations

import re
from typing import Any


def move_cursor(cursor: int, delta: int, length: int) -> int:
    """Clamp navigation cursor within [0, length - 1]."""
    return 0 if length <= 0 else max(0, min(length - 1, cursor + delta))


def render_log_rows(records: list[dict[str, Any]], cursor: int) -> str:
    """Render compact rich-formatted routing history rows."""
    if not records:
        return "(no routing decisions yet)"
    lines = []
    for index, record in enumerate(records):
        stamp = record.get("time", record.get("timestamp", "--"))
        route = record.get("route", "fast")
        model = record.get("model", record.get("model_used", "--"))
        prompt = (
            str(record.get("prompt", ""))
            .replace("\n", " ")[:40]
            .replace("[", "\\[")
        )
        confidence = record.get("confidence", "--")
        line = (
            f"› {stamp} {route} {model} {prompt} ({confidence})"
            if index == cursor
            else f"  {stamp} {route} {model} {prompt} ({confidence})"
        )
        lines.append(f"[reverse]{line}[/reverse]" if index == cursor else line)
    return "\n".join(lines)


def _cell_len(s: str) -> int:
    """Compute visual terminal cell width of markup string."""
    try:
        from rich.text import Text

        return Text.from_markup(s).cell_len
    except Exception:
        return len(re.sub(r"\[/?[a-zA-Z0-9_ =#,-]+\]", "", s))


def _format_box_lines(title: str, lines: list[str], width: int = 76) -> list[str]:
    """Render a structured border box with header title."""
    title_len = _cell_len(title)
    dashes = max(0, width - 6 - title_len)
    top = f"+-- {title} {'-' * dashes}+"
    bottom = f"+{'-' * (width - 2)}+"

    result = [top]
    for line in lines:
        if line.startswith("-") and set(line) == {"-"}:
            result.append(f"+{'-' * (width - 2)}+")
            continue
        l_len = _cell_len(line)
        pad = max(0, width - 4 - l_len)
        result.append(f"| {line}{' ' * pad} |")
    result.append(bottom)
    return result
