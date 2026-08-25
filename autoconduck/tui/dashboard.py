"""Compact live routing dashboard / main navigation hub."""

from __future__ import annotations

from typing import Any

from autoconduck.tui.dashboard_screens import (
    DrillDownScreen,
    LaunchAgentScreen,
    UpdateCatalogScreen,
    UpdateScreen,
)
from autoconduck.tui.dashboard_widgets import (
    _cell_len,
    decision_path,
    _format_box_lines,
    move_cursor,
    record_value,
    render_log_rows,
)
from autoconduck.tui.keymap import FOOTER_HINT

try:
    from textual.containers import Vertical
    from textual.screen import Screen
    from textual.widgets import DataTable, Static
    _TEXTUAL = True
except ImportError:
    _TEXTUAL = False

    class Screen:  # type: ignore
        pass

    def _require():
        raise RuntimeError("Textual is required to use the AutoConduck TUI")


if _TEXTUAL:
    class MainMenuScreen(Screen):
        """Main navigation hub."""

        DUCK_FRAMES = [
            '  __(o<   [bold cyan]AutoConduck[/bold cyan]',
            '  __(.<   [bold cyan]AutoConduck[/bold cyan]',
            '  __(^<   [bold cyan]AutoConduck[/bold cyan]',
        ]

        MENU_ITEMS = [
            ("d", "Live Routing Stats", "Real-time routing decisions & cost tracker"),
            ("m", "Model Catalog", "Providers, presets, and API key vault"),
            ("u", "Check for Updates", "Check latest version & upgrade AutoConduck"),
            ("s", "Settings", "Launch behaviour, thresholds, log level"),
            ("a", "Launch Agent", "Start a configured coding agent"),
        ]

        def __init__(self):
            super().__init__()
            self.cursor = 0
            self.duck_frame = 0
            self.totals = {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost": 0.0,
            }

        def compose(self):
            yield Vertical(
                Static(self._header(), id="header", markup=True),
                Static(self._stats_line(), id="stats", markup=True),
                Static(self._menu(), id="menu", markup=True),
                Static(
                    "[up/down] navigate  [enter] open  [key] shortcut  [ctrl+c] quit",
                    id="footer",
                    markup=False,
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
                if i == self.cursor:
                    row = f"{mark}\\[{key}] [bold cyan]{title}[/bold cyan]  [dim]{desc}[/dim]"
                else:
                    row = f"{mark}\\[{key}] [bold]{title}[/bold]  [dim]{desc}[/dim]"
                lines.append(row)
            return "\n".join(lines)

        def _navigate(self, idx: int):
            _, title, _ = self.MENU_ITEMS[idx]
            app = self.app
            if idx == 0:
                app.push_screen(DashboardScreen())
            elif idx == 1:
                from .onboarding.screens_models import ModelCatalogScreen
                app.push_screen(ModelCatalogScreen(app))
            elif idx == 2:
                app.push_screen(UpdateScreen(app))
            elif idx == 3:
                from .settings import SettingsScreen

                app.push_screen(SettingsScreen(app))
            elif idx == 4:
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

    class DashboardScreen(Screen):
        """Live routing decisions log — accessible from the main menu."""

        BINDINGS = [("d", "drill", "drill selected decision"), ("p", "pause", "pause"), ("enter", "drill", "open decision")]
        DUCK_FRAMES = [
            '  __(o<   [bold cyan]AutoConduck[/bold cyan] . [green]proxy active[/green]',
            '  __(.<   [bold cyan]AutoConduck[/bold cyan] . [green]proxy active[/green]',
            '  __(^<   [bold cyan]AutoConduck[/bold cyan] . [green]proxy active[/green]',
        ]

        def __init__(self):
            super().__init__()
            self.records: list[dict[str, Any]] = []
            self.cursor = 0
            self.paused = False
            self.duck_frame = 0
            self.totals = {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost": 0.0,
            }
            self.models_breakdown: dict[str, dict[str, Any]] = {}
            self.paths: dict[str, int] = {}
            self.pseudos: dict[str, int] = {}
            self.metrics: dict[str, Any] = {}

        def compose(self):
            yield Vertical(
                Static(self._mascot_header(), id="header", markup=True),
                Static(self._graph_view(), id="graph", markup=True),
                Static(self._telemetry_cards(), id="telemetry", markup=True),
                Static(self._stats_summary(), id="stats", markup=True),
                DataTable(id="decisions", cursor_type="row"),
                Static(
                    "[up/down] move  [enter/d] details  [p] pause  [esc/left] back  [ctrl+c] quit",
                    id="footer",
                    markup=False,
                ),
            )

        def on_mount(self):
            try:
                table = self.query_one("#decisions", DataTable)
                table.add_columns("Timestamp", "Turn #", "Path", "Model Selected", "In/Out Tokens", "Cost ($)", "Turn Latency (ms)")
            except Exception:
                pass
            self._update_stats()
            self.set_interval(0.8, self._tick)

        def _tick(self):
            self.duck_frame = (self.duck_frame + 1) % len(self.DUCK_FRAMES)
            self._update_stats()
            try:
                self.query_one("#header", Static).update(self._mascot_header())
                self.query_one("#graph", Static).update(self._graph_view())
                self.query_one("#telemetry", Static).update(self._telemetry_cards())
                self.query_one("#stats", Static).update(self._stats_summary())
                self._refresh_table()
            except Exception:
                pass

        def _update_stats(self):
            try:
                from autoconduck import stats

                records = stats.load_records(limit=100)
                if records:
                    self.records = list(reversed(records))
                    agg = stats.aggregate(records)
                    self.totals = agg.get("totals", self.totals)
                    self.models_breakdown = agg.get("models", {})
                    self.paths = agg.get("paths", {})
                    self.pseudos = agg.get("pseudos", {})
                    self.metrics = agg.get("totals", {})
                else:
                    self.metrics = {}
            except Exception:
                pass

        def _refresh_table(self):
            try:
                table = self.query_one("#decisions", DataTable)
                table.clear(columns=False)
                for index, record in enumerate(self.records):
                    stamp = str(record_value(record, "time", "timestamp", "ts"))[:19]
                    path = decision_path(record)
                    model = str(record_value(record, "model", "model_used"))[:24]
                    prompt = int(record.get("prompt_tokens", 0) or 0)
                    completion = int(record.get("completion_tokens", 0) or 0)
                    latency = record_value(record, "latency_ms", "turn_latency_ms")
                    table.add_row(stamp, str(record_value(record, "turn", "turn_number", default=index + 1)), path, model, f"{prompt:,}/{completion:,}", f"${float(record.get('cost', 0) or 0):.4f}", str(latency))
            except Exception:
                pass

        def _telemetry_cards(self) -> str:
            t = self.metrics or self.totals
            cost = float(t.get("cost", 0) or 0)
            savings = float(t.get("savings_percentage", 0) or 0)
            prompt = int(t.get("prompt_tokens", 0) or 0)
            completion = int(t.get("completion_tokens", 0) or 0)
            cache = float(t.get("prompt_cache_hit_ratio", t.get("cache_hit_ratio", 0)) or 0)
            return "\n".join(_format_box_lines("Live Telemetry", [
                f"Cost: [bold green]${cost:.4f}[/bold green] session/monthly  Savings: [bold cyan]{savings:.1f}%[/bold cyan]",
                f"Requests: [bold]{t.get('calls', 0)}[/bold]  Active streams: [bold]{t.get('active_streams', t.get('concurrent_streams', 0))}[/bold]  Token velocity: {prompt:,} prompt / {completion:,} completion tok/s",
                f"Prefix cache: [bold]{cache:.1f}%[/bold] hit  Latency saved: [bold]{float(t.get('estimated_latency_saved_ms', 0) or 0):.1f} ms[/bold]",
            ], width=76))

        def _graph_view(self) -> str:
            try:
                from autoconduck import stats

                active = stats.get_active_routing()
            except Exception:
                active = {"active": False, "path": "FAST", "node": "idle"}

            is_active = active.get("active", False)
            path = active.get("path", "FAST")
            node = active.get("node", "idle")
            detail = active.get("step_detail", "Ready")
            model = active.get("selected_model", "autoconduck")
            val = active.get("task_value", 0.0)
            completed = active.get("subtasks_completed", 0)
            total = active.get("subtasks_total", 0)

            if is_active and decision_path({"path": path}) == "SLOW":
                tg_n = "[green][OK] GUARD[/green]"
                slm_n = (
                    "[bold yellow]● SLM PLAN[/bold yellow]"
                    if node in ("init", "rag", "slm")
                    else "[green][OK] SLM PLAN[/green]"
                )
                dag_label = (
                    f"DAG NODES ({completed}/{total})"
                    if total
                    else "DYNAMIC DAG"
                )
                dag_n = (
                    f"[bold yellow]● {dag_label}[/bold yellow]"
                    if node not in ("init", "rag", "slm", "synthesizer", "idle")
                    else (
                        "[green][OK] DYNAMIC DAG[/green]"
                        if node == "synthesizer"
                        else "[dim]○ DYNAMIC DAG[/dim]"
                    )
                )
                syn_n = (
                    "[bold yellow]● SYNTHESIZER[/bold yellow]"
                    if node == "synthesizer"
                    else "[dim]○ SYNTHESIZER[/dim]"
                )

                lines = [
                    f"Target: [bold]{model}[/bold] | Active Node: [bold yellow]{node}[/bold yellow]",
                    "",
                    f"[START] ──► {tg_n} ──► {slm_n} ──► {dag_n} ──► {syn_n} ──► [END]",
                    "",
                    f"Status: [bold cyan]{detail}[/bold cyan]",
                ]
                return "\n".join(
                    _format_box_lines(
                        "[bold cyan]LangGraph Dynamic DAG Execution[/bold cyan]",
                        lines,
                        width=76,
                    )
                )
            elif is_active and decision_path({"path": path}) == "FAST":
                lines = [
                    f"Selected Model: [bold green]{model}[/bold green] | Task Value V: [bold]{val:.2f}[/bold]",
                    "",
                    "[START] ──► [Turn Guard (0ms)] ──► [FAST DIRECT DISPATCH] ──► [END]",
                    "",
                    f"Status: [bold green]{detail}[/bold green]",
                ]
                return "\n".join(
                    _format_box_lines(
                        "[bold green]Direct FAST Path Execution[/bold green]",
                        lines,
                        width=76,
                    )
                )
            elif is_active and decision_path({"path": path}) == "OMP":
                lines = [f"Delegated Model: [bold cyan]{model}[/bold cyan]", "", "[START] ──► [Turn Guard] ──► [OMP Agent Delegation] ──► [OMP Subagent Hook] ──► [END]", "", f"Status: [bold cyan]{detail}[/bold cyan]"]
                return "\n".join(_format_box_lines("OMP Agent Delegation", lines, width=76))
            elif is_active:
                lines = [f"Model: [bold yellow]{model}[/bold yellow]", "", "[START] ──► [Turn Guard] ──► [FALLBACK PATH] ──► [END]", "", f"Status: [bold yellow]{detail}[/bold yellow]"]
                return "\n".join(_format_box_lines("Fallback Path", lines, width=76))
            else:
                from autoconduck import __version__
                lines = [
                    f"Engine: [bold]AutoConduck {__version__} Engine[/bold] | Standby | SLM & RAG Active",
                    "",
                    "[START] ──► [Turn Guard] ──► (Direct Fast Path / Dynamic DAG Engine)",
                ]
                return "\n".join(
                    _format_box_lines(
                        "[bold dim]SLM Orchestration Engine Standby[/bold dim]",
                        lines,
                        width=76,
                    )
                )

        def _mascot_header(self):
            status_label = "PAUSED" if self.paused else "RUNNING"
            status_color = "yellow" if self.paused else "green"
            duck = self.DUCK_FRAMES[self.duck_frame]
            return f"{duck} [{status_color}][{status_label}][/{status_color}]\n   \\___)"

        def _stats_summary(self):
            t = self.totals
            lines = [
                f"Calls: [bold]{t['calls']}[/bold]    Spend: [bold green]${t['cost']:.4f}[/bold green] USD",
                f"Tokens: [bold]{t['total_tokens']:,}[/bold] ({t['prompt_tokens']:,} in / {t['completion_tokens']:,} out)",
            ]
            if self.models_breakdown:
                lines.append("-" * 72)
                lines.append(
                    f"{'Model':<28} | {'Calls':<6} | {'Tokens':<12} | {'Spend ($)':<10}"
                )
                lines.append("-" * 72)
                for m, row in list(self.models_breakdown.items())[:5]:
                    m_display = m[:28]
                    lines.append(
                        f"{m_display:<28} | {row['calls']:<6} | {row['total_tokens']:<12,} | ${row['cost']:<9.4f}"
                    )
            if self.paths or self.pseudos:
                lines.append("-" * 72)
                path_str = ", ".join(
                    f"{k}={v}" for k, v in sorted(self.paths.items())
                )
                lines.append(f"Paths: {path_str}")
            return "\n".join(
                _format_box_lines("Usage & Cost Accounting", lines, width=76)
            )

        def on_key(self, event):
            if event.key == "down":
                self.cursor = move_cursor(self.cursor, 1, len(self.records))
                self._refresh_table()
            elif event.key == "up":
                self.cursor = move_cursor(self.cursor, -1, len(self.records))
                self._refresh_table()
            elif event.key in ("left", "escape", "b"):
                self.app.pop_screen()

        def action_pause(self):
            self.paused = not self.paused
            self.query_one("#header").update(self._mascot_header())

        def action_drill(self):
            record = self.records[self.cursor] if self.records else {}
            self.app.push_screen(DrillDownScreen(record))

        def on_data_table_row_selected(self, event):
            try:
                self.cursor = event.cursor_row
                self.action_drill()
            except Exception:
                pass

else:
    class MainMenuScreen(Screen):  # type: ignore
        def __init__(self, *args, **kwargs):
            _require()

    class DashboardScreen(Screen):  # type: ignore
        def __init__(self, *args, **kwargs):
            _require()


__all__ = [
    "MainMenuScreen",
    "DashboardScreen",
    "DrillDownScreen",
    "LaunchAgentScreen",
    "UpdateScreen",
    "UpdateCatalogScreen",
    "move_cursor",
    "render_log_rows",
    "_format_box_lines",
]
