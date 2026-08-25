"""Textual application shell with optional-dependency guard."""
from __future__ import annotations

try:
    from textual.app import App
    from textual.widgets import Footer
    _TEXTUAL = True
except ImportError:
    _TEXTUAL = False
    class App: pass

if _TEXTUAL:
    from .onboarding import OnboardingScreen, ModelSourceScreen
    from .dashboard import DashboardScreen, MainMenuScreen

    class AutoConduckApp(App):
        BINDINGS = [("ctrl+c", "quit", "quit"), ("ctrl+q", "ignore_quit", "disabled")]
        CSS = "Screen { padding: 1; } #footer { color: $text-muted; } #header { color: $success; } .focused { background: $boost; color: $text; }"

        def __init__(self, configured=False, initial_screen=None):
            super().__init__()
            self.configured = configured
            self.initial_screen = initial_screen
            self.paused = False

        def on_mount(self):
            try:
                from autoconduck.launcher import ensure_server
                from autoconduck.config import get_config
                cfg = get_config()
                port = getattr(cfg, "port", None) or 11434
                ensure_server(port)
            except Exception:
                pass
            if self.initial_screen in ("edit", "models"):
                self.push_screen(ModelSourceScreen(self))
            elif not self.configured:
                self.push_screen(ModelSourceScreen(self))
            else:
                self.push_screen(MainMenuScreen())

        def action_quit(self):
            try:
                from autoconduck.launcher import stop_server
                from autoconduck.config import get_config

                cfg = get_config()
                port = getattr(cfg, "port", None) or 11434
                stop_server(port)
            except Exception:
                pass
            self.exit()

        def action_pause(self): self.paused = not self.paused
        def action_edit(self): self.push_screen(ModelSourceScreen(self))
        def action_ignore_quit(self): pass
        def action_help(self): self.notify("up/down move  enter open  d stats  m models  s settings  a launch agent  ctrl+c quit")
else:
    class AutoConduckApp(App):
        def __init__(self, *args, **kwargs): raise RuntimeError("Textual is required to use the AutoConduck TUI")
