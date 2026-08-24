"""Secondary TUI screens for catalog updating, agent launching, and log drilldown."""

from __future__ import annotations

import json
from typing import Any

from autoconduck.tui.dashboard_widgets import move_cursor

try:
    from textual.containers import Vertical
    from textual.screen import Screen
    from textual.widgets import Static
    _TEXTUAL = True
except ImportError:
    _TEXTUAL = False

    class Screen:  # type: ignore
        pass

    def _require():
        raise RuntimeError("Textual is required to use the AutoConduck TUI")


if _TEXTUAL:
    class UpdateCatalogScreen(Screen):
        """Update model presets and refresh catalog snapshot."""

        def __init__(self, controller=None):
            super().__init__()
            self.controller = controller
            self._status = (
                "Press [bold cyan][enter][/bold cyan] or [bold cyan][u][/bold cyan] to sync latest models & pricing."
            )
            self._running = False
            self._log_lines: list[str] = []

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

                self._status = "[bold green]✓ Catalog updated successfully with latest models![/bold green]"
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
            self._agents: list[dict[str, Any]] = []
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

    class DrillDownScreen(Screen):
        """View complete details and metadata for a single routing event."""

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
    class UpdateCatalogScreen(Screen):  # type: ignore
        def __init__(self, *args, **kwargs):
            _require()

    class LaunchAgentScreen(Screen):  # type: ignore
        def __init__(self, *args, **kwargs):
            _require()

    class DrillDownScreen(Screen):  # type: ignore
        def __init__(self, *args, **kwargs):
            _require()
