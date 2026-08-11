"""Compact live routing dashboard / main navigation hub."""
from __future__ import annotations
import json
from .keymap import FOOTER_HINT


def move_cursor(cursor: int, delta: int, length: int) -> int:
    return 0 if length <= 0 else max(0, min(length - 1, cursor + delta))


def render_log_rows(records: list[dict], cursor: int) -> str:
    if not records:
        return "(no routing decisions yet)"
    lines = []
    for index, record in enumerate(records):
        stamp = record.get("time", record.get("timestamp", "--"))
        route = record.get("route", "fast")
        model = record.get("model", record.get("model_used", "--"))
        prompt = str(record.get("prompt", "")).replace("\n", " ")[:40].replace("[", "\\[")
        confidence = record.get("confidence", "--")
        line = (
            f"› {stamp} {route} {model} {prompt} ({confidence})"
            if index == cursor
            else f"  {stamp} {route} {model} {prompt} ({confidence})"
        )
        lines.append(f"[reverse]{line}[/reverse]" if index == cursor else line)
    return "\n".join(lines)


try:
    from textual.containers import Vertical
    from textual.screen import Screen
    from textual.widgets import Static, Input
    _TEXTUAL = True
except ImportError:
    _TEXTUAL = False
    class Screen:
        pass
    def _require():
        raise RuntimeError("Textual is required to use the AutoConduck TUI")


if _TEXTUAL:
    class MainMenuScreen(Screen):
        """Main navigation hub.

        Replaces the old passive dashboard as the app entry point.
        From here you can navigate to every other TUI screen.
        """

        DUCK_FRAMES = [
            '  __(o<   [bold cyan]AutoConduck[/bold cyan]',
            '  __(.<   [bold cyan]AutoConduck[/bold cyan]',
            '  __(^<   [bold cyan]AutoConduck[/bold cyan]',
        ]

        MENU_ITEMS = [
            ("d", "Live Routing Stats",     "Real-time routing decisions & cost tracker"),
            ("m", "Configure Models",       "Add providers, select models, set API keys"),
            ("t", "Tune Budget",            "Budget limits, cost targets, ambiguity bands"),
            ("s", "Settings",               "Launch behaviour, thresholds, log level"),
            ("a", "Launch Agent",           "Start a configured coding agent"),
        ]

        def __init__(self):
            super().__init__()
            self.cursor = 0
            self.duck_frame = 0
            self.totals = {
                "calls": 0, "prompt_tokens": 0,
                "completion_tokens": 0, "total_tokens": 0, "cost": 0.0,
            }

        def compose(self):
            yield Vertical(
                Static(self._header(), id="header", markup=True),
                Static(self._stats_line(), id="stats", markup=True),
                Static(self._menu(), id="menu", markup=True),
                Static(
                    "[up/down] navigate  [enter] open  [key] shortcut  [ctrl+c] quit",
                    id="footer",
                ),
            )

        def on_mount(self):
            self._update_stats()
            self.set_interval(1.5, self._tick)

        def _tick(self):
            self.duck_frame = (self.duck_frame + 1) % len(self.DUCK_FRAMES)
            self._update_stats()
            try:
                self.query_one("#header", Static).update(self._header())
                self.query_one("#stats", Static).update(self._stats_line())
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

        def _header(self):
            duck = self.DUCK_FRAMES[self.duck_frame]
            return f"{duck} [green]RUNNING[/green]\n  \\___)"

        def _stats_line(self):
            t = self.totals
            return (
                f"Calls: [bold]{t['calls']}[/bold]  "
                f"Tokens: [bold]{t['total_tokens']:,}[/bold]  "
                f"Spend: [bold green]${t['cost']:.4f}[/bold green] USD"
            )

        def _menu(self):
            lines = ["", "  Navigate to:", ""]
            for i, (key, title, desc) in enumerate(self.MENU_ITEMS):
                mark = ">> " if i == self.cursor else "   "
                row = f"{mark}[{key}] [bold]{title}[/bold]  [dim]{desc}[/dim]"
                lines.append(f"[reverse]{row}[/reverse]" if i == self.cursor else row)
            return "\n".join(lines)

        def _navigate(self, idx: int):
            _, title, _ = self.MENU_ITEMS[idx]
            app = self.app
            if idx == 0:  # Live Stats
                app.push_screen(DashboardScreen())
            elif idx == 1:  # Configure Models
                from .onboarding import ModelSourceScreen
                app.push_screen(ModelSourceScreen(app))
            elif idx == 2:  # Tune Budget
                from .tune import TuneScreen
                app.push_screen(TuneScreen(app))
            elif idx == 3:  # Settings
                from .settings import SettingsScreen
                app.push_screen(SettingsScreen(app))
            elif idx == 4:  # Launch Agent
                app.push_screen(LaunchAgentScreen(app))

        def on_key(self, event):
            if event.key == "down":
                self.cursor = move_cursor(self.cursor, 1, len(self.MENU_ITEMS))
                self.query_one("#menu").update(self._menu())
            elif event.key == "up":
                self.cursor = move_cursor(self.cursor, -1, len(self.MENU_ITEMS))
                self.query_one("#menu").update(self._menu())
            elif event.key == "enter":
                self._navigate(self.cursor)
            else:
                for i, (key, _, _) in enumerate(self.MENU_ITEMS):
                    if event.key == key:
                        self.cursor = i
                        self.query_one("#menu").update(self._menu())
                        self._navigate(i)
                        break

    class LaunchAgentScreen(Screen):
        """Pick and launch a configured coding agent."""

        def __init__(self, controller=None):
            super().__init__()
            self.controller = controller
            self.cursor = 0
            self._agents = []
            self._msg = ""

        def _load_agents(self):
            from autoconduck.tui.onboarding import AGENTS, detect_agents, is_agent_configured
            detected = detect_agents()
            configured = {a for a in AGENTS if is_agent_configured(a)}
            self._agents = [
                {
                    "id": a,
                    "detected": detected.get(a) or "not found",
                    "configured": a in configured,
                }
                for a in AGENTS
            ]

        def _render_rows(self):
            lines = ["  Select a coding agent to launch:\n"]
            for i, ag in enumerate(self._agents):
                status = "[green]ready[/green]" if ag["configured"] else "[red]not configured[/red]"
                mark = ">> " if i == self.cursor else "   "
                row = f"{mark}[bold]{ag['id']}[/bold]  {status}  [dim]{ag['detected']}[/dim]"
                lines.append(f"[reverse]{row}[/reverse]" if i == self.cursor else row)
            if self._msg:
                lines.append(f"\n{self._msg}")
            return "\n".join(lines)

        def compose(self):
            self._load_agents()
            yield Vertical(
                Static("AutoConduck - Launch Agent"),
                Static(self._render_rows(), id="rows", markup=True),
                Static(
                    "[up/down] navigate  [enter] launch  [left] back  [ctrl+c] quit",
                    id="footer",
                ),
            )

        def _launch(self):
            if not self._agents:
                return
            ag = self._agents[self.cursor]
            if not ag["configured"]:
                self._msg = f"Agent '{ag['id']}' is not configured — run: autoconduck install {ag['id']}"
                self.query_one("#rows").update(self._render_rows())
                return
            # Launch in a background thread so the TUI stays responsive
            import threading
            from autoconduck.main import cmd_launch_agent

            self._msg = f"Launching {ag['id']}…"
            self.query_one("#rows").update(self._render_rows())

            def _run():
                cmd_launch_agent(ag["id"])

            threading.Thread(target=_run, daemon=True).start()

        def on_key(self, event):
            if event.key == "down":
                self.cursor = move_cursor(self.cursor, 1, len(self._agents))
                self.query_one("#rows").update(self._render_rows())
            elif event.key == "up":
                self.cursor = move_cursor(self.cursor, -1, len(self._agents))
                self.query_one("#rows").update(self._render_rows())
            elif event.key == "enter":
                self._launch()
            elif event.key == "left":
                self.app.pop_screen()

    class DashboardScreen(Screen):
        """Live routing decisions log — accessible from the main menu."""

        BINDINGS = [("d", "drill", "drill"), ("/", "filter", "filter"), ("p", "pause", "pause")]
        DUCK_FRAMES = [
            '  __(o<   [bold cyan]AutoConduck[/bold cyan] . [green]proxy active[/green]',
            '  __(.<   [bold cyan]AutoConduck[/bold cyan] . [green]proxy active[/green]',
            '  __(^<   [bold cyan]AutoConduck[/bold cyan] . [green]proxy active[/green]',
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
                Static("[up/down] move  [d] details  [/] filter  [p] pause  [left] back  [ctrl+c] quit", id="footer"),
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
            status = "[yellow]PAUSED[/yellow]" if self.paused else "[green]RUNNING[/green]"
            duck = self.DUCK_FRAMES[self.duck_frame]
            return f"{duck} [{status}]\n  \\___)"

        def _stats_summary(self):
            t = self.totals
            return (
                f"+----- Usage & Cost Tracker --------------------------------+\n"
                f"| Calls: [bold]{t['calls']}[/bold]  Tokens: [bold]{t['total_tokens']:,}[/bold] ({t['prompt_tokens']:,} in / {t['completion_tokens']:,} out) |\n"
                f"| Spend: [bold green]${t['cost']:.4f}[/bold green] USD                                       |\n"
                f"+-----------------------------------------------------------+"
            )

        def on_key(self, event):
            if event.key == "down":
                self.cursor = move_cursor(self.cursor, 1, len(self.records))
                self.query_one("#log").update("recent routing decisions\n" + render_log_rows(self.records, self.cursor))
            elif event.key == "up":
                self.cursor = move_cursor(self.cursor, -1, len(self.records))
                self.query_one("#log").update("recent routing decisions\n" + render_log_rows(self.records, self.cursor))
            elif event.key == "left":
                self.app.pop_screen()

        def action_pause(self):
            self.paused = not self.paused
            self.query_one("#header").update(self._mascot_header())

        def action_filter(self):
            self.mount(Input(placeholder="filter by agent/model/path", id="filter"))

        def action_drill(self):
            record = self.records[self.cursor] if self.records else {}
            self.app.push_screen(DrillDownScreen(record))

    class DrillDownScreen(Screen):
        def __init__(self, decision=None):
            super().__init__()
            self.decision = decision or {}

        def compose(self):
            content = (
                "+----- Routing Decision -----+\n"
                + "\n".join(f"{k}: {v}" for k, v in self.decision.items())
                + "\n\n[left] back  [c] copy JSON  [ctrl+c] quit"
            )
            yield Static(content)

        def on_key(self, event):
            if event.key == "left":
                self.app.pop_screen()
            elif event.key == "c" and self.decision:
                try:
                    self.app.copy_to_clipboard(json.dumps(self.decision))
                except Exception:
                    pass

else:
    class MainMenuScreen(Screen):
        def __init__(self, *args, **kwargs):
            _require()

    class DashboardScreen(Screen):
        def __init__(self, *args, **kwargs):
            _require()

    class DrillDownScreen(Screen):
        def __init__(self, *args, **kwargs):
            _require()

    class LaunchAgentScreen(Screen):
        def __init__(self, *args, **kwargs):
            _require()