"""Textual application shell with optional-dependency guard."""
from __future__ import annotations
try:
    from textual.app import App
    from textual.widgets import Footer
    from .onboarding import OnboardingScreen, ModelSourceScreen
    from .dashboard import DashboardScreen, MainMenuScreen
    _TEXTUAL = True
except ImportError:
    _TEXTUAL = False
    class App: pass

if _TEXTUAL:
    class AutoConduckApp(App):
        BINDINGS = [("ctrl+c", "quit", "quit"), ("ctrl+q", "ignore_quit", "disabled")]
        CSS = "Screen { padding: 1; } #footer { color: $text-muted; } #header { color: $success; } .focused { background: $boost; color: $text; }"
        def __init__(self, configured=False, tune_mode=None): super().__init__(); self.configured = configured; self.tune_mode = tune_mode; self.paused = False
        def on_mount(self):
            if self.tune_mode:
                from .tune import TuneScreen, SimpleTuneScreen, AdvancedTuneScreen
                screen = SimpleTuneScreen(self) if self.tune_mode == "simple" else AdvancedTuneScreen(self) if self.tune_mode == "advanced" else TuneScreen(self)
                self.push_screen(screen)
            else:
                # Configured: go to the main navigation menu.
                # Not configured: run the onboarding flow first.
                self.push_screen(MainMenuScreen() if self.configured else OnboardingScreen(self))
        def action_pause(self): self.paused = not self.paused
        def action_edit(self): self.push_screen(ModelSourceScreen(self))
        def action_ignore_quit(self): pass
        def action_help(self): self.notify("up/down move  enter open  d stats  m models  t tune  s settings  a launch agent  ctrl+c quit")
else:
    class AutoConduckApp(App):
        def __init__(self, *args, **kwargs): raise RuntimeError("Textual is required to use the AutoConduck TUI")