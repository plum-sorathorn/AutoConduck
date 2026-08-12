"""Onboarding helpers."""
from __future__ import annotations
import shutil
from pathlib import Path
from autoconduck.config import get_config, save_config, resolve_api_key

def move_cursor(cursor: int, delta: int, length: int) -> int:
    return 0 if length <= 0 else max(0, min(length - 1, cursor + delta))


def render_agent_rows(names, detected, selected, cursor):
    return "\n".join(
        ("[reverse]" if i == cursor else "")
        + f"{'›' if i == cursor else ' '} {'✓' if n in selected else ' '} {n:16} {detected.get(n) or 'not found'}"
        + ("[/reverse]" if i == cursor else "")
        for i, n in enumerate(names)
    )


def render_source_rows(sources, selected, cursor, models=None):
    rows = ["┌─ Model Sources ─┐"] + [
        (
            f"[reverse]› {'✓' if i in selected else ' '} {s}[/reverse]"
            if i == cursor
            else f"  {'✓' if i in selected else ' '} {s}"
        )
        for i, s in enumerate(sources)
    ]
    return "\n".join(
        rows
        + ["", "Selected models for routing pool:"]
        + [f"  {m}" for m in (models or [sources[i] for i in sorted(selected)])]
        or ["  none"]
    )


def render_model_rows(models, enabled, cursor):
    def price(m):
        return f" · in ${format_price(m.get('price_in', 0))} · out ${format_price(m.get('price_out', 0))} /1M"

    return (
        "\n".join(
            (
                f"[reverse]› {'✓' if m['id'] in enabled else ' '} {m['id']} ({m.get('tier', 'balanced')}){price(m)}[/reverse]"
                if i == cursor
                else f"  {'✓' if m['id'] in enabled else ' '} {m['id']} ({m.get('tier', 'balanced')}){price(m)}"
            )
            for i, m in enumerate(models)
        )
        or "No models available for this provider."
    )


def format_price(value) -> str:
    value = float(value or 0)
    return f"{value:.3f}" if 0 < value < 1 else f"{value:.2f}"


def model_option_label(row) -> str:
    return f"{row['id']} (${format_price(row['price_in'])} / ${format_price(row['price_out'])} per 1M)"


def render_provider_rows(providers, cursor):
    def _name(p):
        return p.get("provider", "") if isinstance(p, dict) else str(p)

    def _on(p):
        return bool(p.get("enabled", True)) if isinstance(p, dict) else True

    return (
        "\n".join(
            (
                f"[reverse]› {'✓' if _on(p) else ' '} {_name(p)}[/reverse]"
                if i == cursor
                else f"  {'✓' if _on(p) else ' '} {_name(p)}"
            )
            for i, p in enumerate(providers)
        )
        or "No custom providers. Press n to add one."
    )


def render_check_rows(agents, checked, cursor):
    return (
        "\n".join(
            (
                f"[reverse]› {'✓' if a in checked else ' '} {a}[/reverse]"
                if i == cursor
                else f"  {'✓' if a in checked else ' '} {a}"
            )
            for i, a in enumerate(agents)
        )
        or "No eligible agents selected."
    )


def _models_value(widget):
    return getattr(widget, "text", None) or getattr(widget, "value", "")


MODELS_PLACEHOLDER = (
    "One model ID per line (newline-separated)\nExample:\n  gpt-4o\n  gpt-4o-mini"
)


def render_models_placeholder(has_models):
    if has_models:
        return ""
    return f"[dim]┌─ {MODELS_PLACEHOLDER}[/dim]"


try:
    from textual.app import ComposeResult
    from textual.containers import Vertical
    from textual.screen import Screen
    from textual.binding import Binding
    from textual.widgets import Static, Input, Label

    try:
        from textual.widgets import TextArea
    except ImportError:
        TextArea = Input
    _TEXTUAL = True
except ImportError:
    _TEXTUAL = False

    class Screen:
        pass

    ComposeResult = object
    Binding = None


def _require_textual():
    if not _TEXTUAL:
        raise RuntimeError("Textual is required to use the AutoConduck TUI")


AGENTS = ("claude_code", "opencode", "pi")


def detect_agents():
    home = Path.home()
    locations = {
        "claude_code": [home / ".claude" / "settings.json"],
        "opencode": [home / ".config" / "opencode" / "config.json"],
        "pi": [home / ".pi" / "agent" / "settings.json"],
    }
    commands = {"claude_code": "claude", "opencode": "opencode", "pi": "pi"}
    return {
        n: next(
            (str(p) for p in locations[n] if p.exists()),
            shutil.which(commands[n]) if n in commands else None,
        )
        for n in AGENTS
    }


def is_agent_configured(agent_id: str) -> bool:
    from ...agents import all_adapters

    adapter = next(
        (
            a
            for a in all_adapters()
            if a.id == agent_id or getattr(a, "binary_name", None) == agent_id
        ),
        None,
    )
    if not adapter:
        return False
    for p in adapter.config_paths():
        if p.exists():
            try:
                txt = p.read_text(encoding="utf-8").lower()
                if "autoconduck" in txt or "# begin autoconduck" in txt:
                    return True
            except Exception:
                pass
    return False



try:
    import textual as _TEXTUAL
except ImportError:
    _TEXTUAL = None

def _require_textual():
    if _TEXTUAL is None:
        raise RuntimeError("Textual is required for the onboarding UI")

