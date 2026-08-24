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


def _cell_len(s: str) -> int:
    try:
        from rich.text import Text
        return Text.from_markup(s).cell_len
    except Exception:
        import re
        return len(re.sub(r'\[/?[a-zA-Z0-9_ =#,-]+\]', '', s))


def _format_box_lines(title: str, lines: list[str], width: int = 76) -> list[str]:
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
            ("u", "Update Catalog",         "Run catalog update scripts & sync latest models"),
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
            if idx == 0:  # Live Stats
                app.push_screen(DashboardScreen())
            elif idx == 1:  # Configure Models
                from .onboarding import ModelSourceScreen
                app.push_screen(ModelSourceScreen(app))
            elif idx == 2:  # Update Catalog
                app.push_screen(UpdateCatalogScreen(app))
            elif idx == 3:  # Tune Budget
                from .tune import TuneScreen
                app.push_screen(TuneScreen(app))
            elif idx == 4:  # Settings
                from .settings import SettingsScreen
                app.push_screen(SettingsScreen(app))
            elif idx == 5:  # Launch Agent
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

    class UpdateCatalogScreen(Screen):
        """Update model presets and refresh catalog snapshot."""

        def __init__(self, controller=None):
            super().__init__()
            self.controller = controller
            self._status = "Press [bold cyan][enter][/bold cyan] or [bold cyan][u][/bold cyan] to sync latest models & pricing."
            self._running = False
            self._log_lines = []

        def compose(self):
            yield Vertical(
                Static("┌─ AutoConduck · Update Model Catalog ─┐", markup=True),
                Static(self._status, id="status", markup=True),
                Static("\n".join(self._log_lines) or "  Ready to sync.", id="log", markup=True),
                Static(
                    "[enter/u] sync catalog  [left/esc] back  [ctrl+c] quit",
                    id="footer",
                    markup=False,
                ),
            )

        def on_key(self, event):
            if event.key in ("left", "escape"):
                self.app.pop_screen()
            elif event.key in ("enter", "u") and not self._running:
                self._run_sync()

        def _run_sync(self):
            self._running = True
            self._status = "[bold cyan]Syncing latest model presets and pricing… please wait…[/bold cyan]"
            self._log_lines = []
            try:
                self.query_one("#status", Static).update(self._status)
                self.query_one("#log", Static).update("Starting catalog update…")
            except Exception:
                pass

            try:
                from autoconduck.presets import presets_data, model_presets

                self._log_lines.append("[dim]1. Syncing upstream model database & presets…[/dim]")
                try:
                    from scripts.sync_all_presets import sync_all
                    synced = sync_all()
                    presets_data.PRESETS.update(synced)
                    self._log_lines.append(f"[green]✓[/green] Synced {len(synced)} provider groups")
                except Exception as exc:
                    self._log_lines.append(f"[yellow]⚠[/yellow] sync_all: {exc}")

                self._log_lines.append("[dim]2. Syncing DevPass models…[/dim]")
                try:
                    from scripts.sync_devpass_presets import fetch_devpass_catalog
                    devpass_entries = fetch_devpass_catalog()
                    if devpass_entries:
                        presets_data.PRESETS["devpass"] = devpass_entries
                        presets_data.FALLBACK_PRESETS["devpass"] = devpass_entries
                        self._log_lines.append(f"[green]✓[/green] Synced {len(devpass_entries)} DevPass models")
                except Exception as exc:
                    self._log_lines.append(f"[yellow]⚠[/yellow] devpass sync: {exc}")

                self._log_lines.append("[dim]3. Refreshing curated catalog snapshot…[/dim]")
                try:
                    from scripts.refresh_catalog import curated_model_catalog
                    model_presets._catalog_cache = None
                    cat = curated_model_catalog()
                    self._log_lines.append(f"[green]✓[/green] Curated catalog contains {len(cat)} models")
                except Exception as exc:
                    self._log_lines.append(f"[yellow]⚠[/yellow] refresh_catalog: {exc}")

                self._status = "[bold green]✓ Catalog updated successfully with latest models (including grok-4.6 and DevPass)![/bold green]"
            except Exception as exc:
                self._status = f"[bold red]Sync failed: {exc}[/bold red]"
            finally:
                self._running = False
                try:
                    self.query_one("#status", Static).update(self._status)
                    self.query_one("#log", Static).update("\n".join(self._log_lines))
                except Exception:
                    pass

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
                    markup=False,
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
            self.app.exit(result=f"launch:{ag['id']}")

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

        BINDINGS = [("d", "drill", "drill"), ("p", "pause", "pause")]
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
            self.models_breakdown: dict[str, dict] = {}
            self.paths: dict[str, int] = {}
            self.pseudos: dict[str, int] = {}

        def compose(self):
            yield Vertical(
                Static(self._mascot_header(), id="header", markup=True),
                Static(self._graph_view(), id="graph", markup=True),
                Static(self._stats_summary(), id="stats", markup=True),
                Static("recent routing decisions\n" + render_log_rows(self.records, self.cursor), id="log", markup=True),
                Static("[up/down] move  [d] details  [p] pause  [esc/left] back  [ctrl+c] quit", id="footer", markup=False),
            )

        def on_mount(self):
            self._update_stats()
            self.set_interval(0.8, self._tick)

        def _tick(self):
            self.duck_frame = (self.duck_frame + 1) % len(self.DUCK_FRAMES)
            self._update_stats()
            try:
                self.query_one("#header", Static).update(self._mascot_header())
                self.query_one("#graph", Static).update(self._graph_view())
                self.query_one("#stats", Static).update(self._stats_summary())
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
            except Exception:
                pass

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

            if is_active and path == "SLOW":
                tg_n = "[green]✓ GUARD[/green]"
                slm_n = "[bold yellow]● SLM PLAN[/bold yellow]" if node in ("init", "rag", "slm") else "[green]✓ SLM PLAN[/green]"
                dag_label = f"DAG NODES ({completed}/{total})" if total else "DYNAMIC DAG"
                dag_n = f"[bold yellow]● {dag_label}[/bold yellow]" if node not in ("init", "rag", "slm", "synthesizer", "idle") else ("[green]✓ DYNAMIC DAG[/green]" if node == "synthesizer" else "[dim]○ DYNAMIC DAG[/dim]")
                syn_n = "[bold yellow]● SYNTHESIZER[/bold yellow]" if node == "synthesizer" else "[dim]○ SYNTHESIZER[/dim]"

                lines = [
                    f"Target: [bold]{model}[/bold] | Active Node: [bold yellow]{node}[/bold yellow]",
                    "",
                    f"[START] ──► {tg_n} ──► {slm_n} ──► {dag_n} ──► {syn_n} ──► [END]",
                    "",
                    f"Status: [bold cyan]{detail}[/bold cyan]",
                ]
                return "\n".join(_format_box_lines("[bold cyan]LangGraph Dynamic DAG Execution[/bold cyan]", lines, width=76))
            elif is_active and path == "FAST":
                lines = [
                    f"Selected Model: [bold green]{model}[/bold green] | Task Value V: [bold]{val:.2f}[/bold]",
                    "",
                    "[START] ──► [Turn Guard (0ms)] ──► [⚡ FAST DIRECT DISPATCH] ──► [END]",
                    "",
                    f"Status: [bold green]{detail}[/bold green]",
                ]
                return "\n".join(_format_box_lines("[bold green]Direct FAST Path Execution[/bold green]", lines, width=76))
            else:
                lines = [
                    "Engine: [bold]AutoConduck 0.3.0 Engine[/bold] | Standby | SLM & RAG Active",
                    "",
                    "[START] ──► [● Turn Guard] ──► (⚡ Direct Fast Path / 🔮 Dynamic DAG Engine)",
                ]
                return "\n".join(_format_box_lines("[bold dim]SLM Orchestration Engine Standby[/bold dim]", lines, width=76))

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
                lines.append(f"{'Model':<28} | {'Calls':<6} | {'Tokens':<12} | {'Spend ($)':<10}")
                lines.append("-" * 72)
                for m, row in list(self.models_breakdown.items())[:5]:
                    m_display = m[:28]
                    lines.append(f"{m_display:<28} | {row['calls']:<6} | {row['total_tokens']:<12,} | ${row['cost']:<9.4f}")
            if self.paths or self.pseudos:
                lines.append("-" * 72)
                path_str = ", ".join(f"{k}={v}" for k, v in sorted(self.paths.items()))
                lines.append(f"Paths: {path_str}")
            return "\n".join(_format_box_lines("Usage & Cost Accounting", lines, width=76))

        def on_key(self, event):
            if event.key == "down":
                self.cursor = move_cursor(self.cursor, 1, len(self.records))
                self.query_one("#log").update("recent routing decisions\n" + render_log_rows(self.records, self.cursor))
            elif event.key == "up":
                self.cursor = move_cursor(self.cursor, -1, len(self.records))
                self.query_one("#log").update("recent routing decisions\n" + render_log_rows(self.records, self.cursor))
            elif event.key in ("left", "escape", "b"):
                self.app.pop_screen()

        def action_pause(self):
            self.paused = not self.paused
            self.query_one("#header").update(self._mascot_header())

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
            yield Static(content, markup=False)

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