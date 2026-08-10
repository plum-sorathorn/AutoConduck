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
        prompt = str(record.get("prompt", "")).replace("\n", " ")[:40].replace("[", "\\[")
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
        DUCK_FRAMES = [
            '  __(o<   [bold cyan]AutoConduck[/bold cyan] · [green]proxy active[/green]',
            '  __(•<   [bold cyan]AutoConduck[/bold cyan] · [green]proxy active[/green]',
            '  __(^<   [bold cyan]AutoConduck[/bold cyan] · [green]proxy active[/green]',
        ]
        def __init__(self):
            super().__init__()
            self.records: list[dict] = []
            self.cursor = 0
            self.paused = False
            self.duck_frame = 0
            self.totals = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0}
        def compose(self):
            yield Vertical(
                Static(self._mascot_header(), id="header", markup=True),
                Static(self._stats_summary(), id="stats", markup=True),
                Static("recent routing decisions\n" + render_log_rows(self.records, self.cursor), id="log", markup=True),
                Static("active agents: none", id="agents"),
                Static("[↑/↓] move  [d] details  [/] filter  [p] pause  [ctrl+c] quit", id="footer")
            )
        def on_mount(self):
            self._update_stats()
            self.set_interval(1.2, self._tick)
        def _tick(self):
            self.duck_frame = (self.duck_frame + 1) % len(self.DUCK_FRAMES)
            self._update_stats()
            try:
                self.query_one("#header", Static).update(self._mascot_header())
                self.query_one("#stats", Static).update(self._stats_summary())
            except Exception:
                pass
        def _update_stats(self):
            try:
                from autoconduck import stats
                records = stats.load_records(limit=50)
                if records:
                    agg = stats.aggregate(records)
                    self.totals = agg.get("totals", self.totals)
            except Exception:
                pass
        def _mascot_header(self):
            status = "[yellow]PAUSED[/yellow]" if self.paused else "[green]● RUNNING[/green]"
            duck = self.DUCK_FRAMES[self.duck_frame]
            return f"{duck} \\[{status}]\n  \\___)"
        def _stats_summary(self):
            t = self.totals
            return (f"┌─ [bold]Usage & Cost Tracker[/bold] ─────────────────────────────────────────┐\n"
                    f"│ Calls: [bold]{t['calls']}[/bold]  ·  Tokens: [bold]{t['total_tokens']:,}[/bold] ({t['prompt_tokens']:,} in / {t['completion_tokens']:,} out) │\n"
                    f"│ Credits / Spend: [bold green]${t['cost']:.4f}[/bold green] USD                                      │\n"
                    f"└──────────────────────────────────────────────────────────────────┘")
        def on_key(self, event):
            if event.key == "down":
                self.cursor = move_cursor(self.cursor, 1, len(self.records))
                self.query_one("#log").update("recent routing decisions\n" + render_log_rows(self.records, self.cursor))
            elif event.key == "up":
                self.cursor = move_cursor(self.cursor, -1, len(self.records))
                self.query_one("#log").update("recent routing decisions\n" + render_log_rows(self.records, self.cursor))
        def action_pause(self):
            self.paused = not self.paused
            self.query_one("#header").update(self._mascot_header())
        def action_filter(self):
            self.mount(Input(placeholder="filter by agent/model/path", id="filter"))
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
