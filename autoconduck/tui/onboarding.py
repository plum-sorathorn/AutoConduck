"""Onboarding screens.  Textual remains an optional dependency."""
from __future__ import annotations
import os
import shutil
from pathlib import Path
from .keymap import FOOTER_HINT
try:
    from autoconduck.pricing import is_subscription
except ImportError:
    def is_subscription(model: str) -> bool:
        return False

def move_cursor(cursor: int, delta: int, length: int) -> int:
    return 0 if length <= 0 else max(0, min(length - 1, cursor + delta))

def render_agent_rows(names: list[str], detected: dict, selected: set[str], cursor: int) -> str:
    rows = []
    for index, name in enumerate(names):
        check = "✓" if name in selected else " "
        row = f"› {check} {name:16} {detected.get(name) or 'not found'}" if index == cursor else f"  {check} {name:16} {detected.get(name) or 'not found'}"
        rows.append(f"[reverse]{row}[/reverse]" if index == cursor else row)
    return "\n".join(rows)

def render_source_rows(sources: list[str], selected: set[int], cursor: int, models: list[str] | None = None) -> str:
    rows = ["┌─ Model Sources ─┐"]
    for index, source in enumerate(sources):
        check = "✓" if index in selected else " "
        row = f"› {check} {source}" if index == cursor else f"  {check} {source}"
        rows.append(f"[reverse]{row}[/reverse]" if index == cursor else row)
    rows.extend(["", "Selected models for routing pool:"])
    chosen = models or [sources[index] for index in sorted(selected) if 0 <= index < len(sources)]
    if chosen:
        for model in chosen:
            glyph = "~" if is_subscription(model) else "$"
            rows.append(f"  {glyph} {model}")
    else:
        rows.append("  none")
    return "\n".join(rows)

try:
    from autoconduck import launcher
except ImportError:
    launcher = None

try:
    from textual.app import ComposeResult
    from textual.containers import Vertical, Horizontal
    from textual.screen import Screen
    from textual.widgets import Static, Input
    _TEXTUAL = True
except ImportError:
    _TEXTUAL = False
    class Screen: pass
    ComposeResult = object

def _require_textual() -> None:
    if not _TEXTUAL:
        raise RuntimeError("Textual is required to use the AutoConduck TUI")

AGENTS = ("claude_code", "opencode", "aider", "continue_dev", "kilocode", "cursor", "generic_openai")
def detect_agents() -> dict[str, str | None]:
    home = Path.home()
    locations = {
        "claude_code": [home / ".claude" / "settings.json"],
        "opencode": [home / ".config" / "opencode" / "config.json"],
        "aider": [home / ".aider.conf.yml"], "continue_dev": [home / ".continue"],
        "kilocode": [home / ".kilocode"], "cursor": [home / ".cursor"],
        "generic_openai": [],
    }
    commands = {"claude_code": "claude", "opencode": "opencode", "aider": "aider", "kilocode": "kilocode"}
    return {name: next((str(p) for p in locations[name] if p.exists()), shutil.which(commands[name]) if name in commands else None) for name in AGENTS}

if _TEXTUAL:
    class OnboardingScreen(Screen):
        BINDINGS = [("a", "all", "select all"), ("tab", "next", "continue"), ("right", "next", "continue"), ("left", "back", "back")]
        def __init__(self, app_controller=None):
            super().__init__(); self.selected: set[str] = set(); self.detected = detect_agents(); self.controller = app_controller; self.cursor = 0
        def compose(self) -> ComposeResult:
            yield Vertical(Static("┌─ AutoConduck · Detected agents ─┐", id="title"), Static(render_agent_rows(list(AGENTS), self.detected, self.selected, self.cursor), id="agents", markup=True), Static("[j/k] move  [space] toggle  [a] select all  [→] continue  [ctrl+c] quit", id="footer"))
        def _text(self):
            return render_agent_rows(list(AGENTS), self.detected, self.selected, self.cursor)
        def _update(self): self.query_one("#agents").update(self._text())
        def action_all(self): self.selected = set(AGENTS); self._update()
        def on_key(self, event):
            if event.key in ("j", "down"): self.cursor = move_cursor(self.cursor, 1, len(AGENTS)); self._update()
            elif event.key in ("k", "up"): self.cursor = move_cursor(self.cursor, -1, len(AGENTS)); self._update()
            elif event.key in ("space", "enter"):
                self.selected.symmetric_difference_update({list(AGENTS)[self.cursor]}); self._update()
            elif event.key in ("right", "tab") and self.controller: self.controller.push_screen(ModelSourceScreen(self.controller, self.selected))
else:
    class OnboardingScreen(Screen):
        def __init__(self, *args, **kwargs): _require_textual()

if _TEXTUAL:
    class ModelSourceScreen(Screen):
        SOURCES = ["Anthropic", "OpenAI", "Google", "DevPass (LLM Gateway)", "Custom endpoint…"]
        def __init__(self, app_controller=None, selected=None):
            super().__init__(); self.controller = app_controller; self.agent_selected = set(selected or ()); self.selected: set[int] = set(); self.cursor = 0; self.models: list[str] = []
        def compose(self) -> ComposeResult:
            yield Vertical(Static(render_source_rows(self.SOURCES, self.selected, self.cursor, self.models), id="sources", markup=True), Static("[j/k] move  [space/enter/→] select  [←] back  [ctrl+c] quit", id="footer"))
        def _update(self): self.query_one("#sources").update(render_source_rows(self.SOURCES, self.selected, self.cursor, self.models))
        def _choose(self):
            self.selected.add(self.cursor)
            if self.SOURCES[self.cursor] not in self.models: self.models.append(self.SOURCES[self.cursor])
            self._update()
        def on_key(self, event):
            if event.key in ("j", "down"): self.cursor = move_cursor(self.cursor, 1, len(self.SOURCES)); self._update()
            elif event.key in ("k", "up"): self.cursor = move_cursor(self.cursor, -1, len(self.SOURCES)); self._update()
            elif event.key in ("space", "enter"): self._choose()
            elif event.key == "right" and self.controller:
                self._choose()
                eligible = self.agent_selected & {"claude_code", "opencode", "aider", "kilocode"}
                if eligible:
                    self.controller.push_screen(LauncherIntegrationScreen(self.controller, self.agent_selected))
                else:
                    self._finish()
            elif event.key == "left" and self.controller: self.controller.pop_screen()
        def _finish(self):
            if self.controller:
                from .dashboard import DashboardScreen
                self.controller.switch_screen(DashboardScreen())
else:
    class ModelSourceScreen(Screen):
        def __init__(self, *args, **kwargs): _require_textual()

if _TEXTUAL:
    class LauncherIntegrationScreen(Screen):
        ELIGIBLE = {"claude_code", "opencode", "aider", "kilocode"}
        def __init__(self, app_controller=None, selected=None):
            super().__init__(); self.controller = app_controller; self.selected = set(selected or ())
            self.result = ""
        def compose(self) -> ComposeResult:
            gui = self.selected & {"cursor", "continue_dev"}
            body = ("Start/stop the AutoConduck server automatically whenever you launch your coding agents? "
                    "This installs launcher shims into ~/.autoconduck/bin and adds that dir to your PATH.\n\n"
                    "Installable: " + ", ".join(sorted(self.selected & self.ELIGIBLE)) )
            if gui:
                body += "\n\nCursor/Continue are GUI apps — they can't be wrapped; start the server manually (`autoconduck start --headless`) or keep it running."
            if self.result:
                body += "\n\n" + self.result
            yield Vertical(Static("Launcher integration", id="title"), Static(body, id="body"), Static(FOOTER_HINT("enter", "esc", "ctrl+c"), id="footer"))
        def on_key(self, event):
            if event.key == "enter": self._install()
            elif event.key == "esc": self._finish()
        def _install(self):
            try:
                if launcher is None: raise ImportError("launcher integration is unavailable")
                shims = launcher.install_shims(sorted(self.selected & self.ELIGIBLE))
                path_file = launcher.ensure_path_entry()
                paths = ", ".join(str(path) for path in shims.values()) or "none"
                self.result = f"Installed shims: {paths}. PATH updated: {path_file or 'unchanged'}."
            except Exception as exc:
                self.result = f"Launcher integration could not be installed: {exc}"
            self.refresh()
        def _finish(self):
            if self.controller:
                from .dashboard import DashboardScreen
                self.controller.switch_screen(DashboardScreen())
else:
    class LauncherIntegrationScreen(Screen):
        def __init__(self, *args, **kwargs): _require_textual()
