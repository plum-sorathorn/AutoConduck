"""Compact live routing dashboard."""
from __future__ import annotations
from .keymap import FOOTER_HINT
def move_cursor(cursor: int, delta: int, length: int) -> int:
    return 0 if length <= 0 else max(0, min(length - 1, cursor + delta))

def render_log_rows(records: list[dict], cursor: int) -> str:
    if not records: return "(no routing decisions yet)"
    lines = []
    for index, record in enumerate(records):
        stamp = record.get("time", record.get("timestamp", "--"))
        route = record.get("route", "fast")
        model = record.get("model", record.get("model_used", "--"))
        prompt = str(record.get("prompt", "")).replace("\n", " ")[:40]
        confidence = record.get("confidence", "--")
        line = f"› {stamp} {route} {model} {prompt} ({confidence})" if index == cursor else f"  {stamp} {route} {model} {prompt} ({confidence})"
        lines.append(f"[reverse]{line}[/reverse]" if index == cursor else line)
    return "\n".join(lines)
try:
    from textual.containers import Vertical
    from textual.screen import Screen
    from textual.widgets import Static, Input
    _TEXTUAL = True
except ImportError:
    _TEXTUAL = False
    class Screen: pass
    def _require(): raise RuntimeError("Textual is required to use the AutoConduck TUI")

if _TEXTUAL:
    class DashboardScreen(Screen):
        BINDINGS = [("d", "drill", "drill"), ("/", "filter", "filter"), ("p", "pause", "pause")]
        def __init__(self): super().__init__(); self.records: list[dict] = []; self.cursor = 0; self.paused = False
        def compose(self):
            yield Vertical(Static(self._header(), id="header"), Static("recent routing decisions\n" + render_log_rows(self.records, self.cursor), id="log", markup=True), Static("active agents: none", id="agents"), Static("[↑/↓] move  [d] details  [/] filter  [p] pause  [ctrl+c] quit", id="footer"))
        def _header(self): return "┌─ AutoConduck ─ proxy: ● running" + (" ─ PAUSED" if self.paused else "") + " ─ saved: $0.00 today ┐"
        def on_key(self, event):
            if event.key == "down": self.cursor = move_cursor(self.cursor, 1, len(self.records)); self.query_one("#log").update("recent routing decisions\n" + render_log_rows(self.records, self.cursor))
            elif event.key == "up": self.cursor = move_cursor(self.cursor, -1, len(self.records)); self.query_one("#log").update("recent routing decisions\n" + render_log_rows(self.records, self.cursor))
        def action_pause(self): self.paused = not self.paused; self.query_one("#header").update(self._header())
        def action_filter(self): self.mount(Input(placeholder="filter by agent/model/path", id="filter"))
        def action_drill(self):
            record = self.records[self.cursor] if self.records else {}
            self.app.push_screen(DrillDownScreen(record))
    class DrillDownScreen(Screen):
        def __init__(self, decision=None): super().__init__(); self.decision = decision or {}
        def compose(self): yield Static("┌─ Routing Decision ─┐\n" + "\n".join(f"{k}: {v}" for k,v in self.decision.items()) + "\n\n[left] back  [c] copy full plan JSON  [ctrl+c] quit")
        def on_key(self, event):
            if event.key == "left": self.app.pop_screen()
            elif event.key == "c" and self.decision: self.app.copy_to_clipboard(json.dumps(self.decision))
else:
    class DashboardScreen(Screen):
        def __init__(self, *args, **kwargs): _require()
    class DrillDownScreen(Screen):
        def __init__(self, *args, **kwargs): _require()
