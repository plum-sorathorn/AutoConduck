"""Textual application shell with optional-dependency guard."""
from __future__ import annotations
try:
    from textual.app import App
    from textual.widgets import Footer
    from .onboarding import OnboardingScreen, ModelSourceScreen
    from .dashboard import DashboardScreen
    _TEXTUAL = True
except ImportError:
    _TEXTUAL = False
    class App: pass

if _TEXTUAL:
    class AutoConduckApp(App):
        BINDINGS = [("ctrl+c", "quit", "quit"), ("ctrl+q", "ignore_quit", "disabled")]
        CSS = "Screen { padding: 1; } #footer { color: $text-muted; } #header { color: $success; } .focused { background: $boost; color: $text; }"
        def __init__(self, configured=False): super().__init__(); self.configured = configured; self.paused = False
        def on_mount(self): self.push_screen(DashboardScreen() if self.configured else OnboardingScreen(self))
        def action_pause(self): self.paused = not self.paused
        def action_edit(self): self.push_screen(ModelSourceScreen(self))
        def action_ignore_quit(self): pass
        def action_help(self): self.notify("↑/↓ move · enter toggle · esc back · / filter · ctrl+c quit · p pause · e edit")
else:
    class AutoConduckApp(App):
        def __init__(self, *args, **kwargs): raise RuntimeError("Textual is required to use the AutoConduck TUI")
