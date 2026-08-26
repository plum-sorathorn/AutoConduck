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
    class UpdateScreen(Screen):
        """Check for updates and run in-app upgrade."""

        BINDINGS = [
            ("enter", "upgrade", "Run upgrade"),
            ("left", "back", "Back"),
            ("escape", "back", "Back"),
        ]

        def __init__(self, controller=None):
            super().__init__()
            self.controller = controller
            self._status = "Press [bold cyan]ENTER[/bold cyan] to run upgrade."
            self._running = False
            self._log_lines: list[str] = []

        def compose(self):
            from pathlib import Path
            from autoconduck import __version__, update
            method = update.detect_install_method()
            cmd = update.upgrade_command(method) or "git pull && pip install -e ."
            yield Vertical(
                Static("┌─ AutoConduck · Software Updates ─┐", markup=True),
                Static(f"Installed version: [bold cyan]{__version__}[/bold cyan] ({method})", markup=True),
                Static(f"Upgrade command:   [dim]{cmd}[/dim]", markup=True),
                Static(self._status, id="status", markup=True),
                Static("\n".join(self._log_lines) or "  Ready to upgrade.", id="log", markup=True),
                Static(
                    "[enter] upgrade  [left/esc] back  [ctrl+c] quit",
                    id="footer",
                    markup=False,
                ),
            )

        def action_upgrade(self):
            if not self._running:
                self._running = True
                self._status = "[bold cyan]Starting upgrade… please wait…[/bold cyan]"
                self._log_lines = ["  Initializing upgrade process…"]
                try:
                    self.query_one("#status", Static).update(self._status)
                    self.query_one("#log", Static).update("\n".join(self._log_lines))
                except Exception:
                    pass
                self.run_worker(self._run_upgrade(), exclusive=True)

        def action_sync(self):
            self.action_upgrade()

        def action_back(self):
            if self.controller:
                try:
                    self.controller.pop_screen()
                    return
                except Exception:
                    pass
            self.app.pop_screen()

        def on_key(self, event):
            if event.key in ("left", "escape"):
                self.action_back()
            elif event.key == "enter" and not self._running:
                self.action_upgrade()

        async def _run_upgrade(self):
            import asyncio
            from pathlib import Path
            from autoconduck import update, launcher
            from autoconduck.config import load_config
            try:
                from autoconduck.server import DEFAULT_PORT
            except ImportError:
                DEFAULT_PORT = 11434

            try:
                method = update.detect_install_method()
                cmd_str = update.upgrade_command(method)

                if not cmd_str:
                    self._status = "[yellow]No managed package manager detected. Please update via git pull and reinstall.[/yellow]"
                    self._running = False
                    try:
                        self.query_one("#status", Static).update(self._status)
                    except Exception:
                        pass
                    return

                self._status = f"[bold cyan]Running upgrade ({cmd_str})… please wait…[/bold cyan]"
                self._log_lines = [f"[dim]Starting: {cmd_str}[/dim]"]
                try:
                    self.query_one("#status", Static).update(self._status)
                    self.query_one("#log", Static).update("\n".join(self._log_lines))
                except Exception:
                    pass

                # Stop server before upgrading to release any locks
                try:
                    cfg = load_config()
                    port = getattr(cfg, "port", None) or DEFAULT_PORT
                    launcher.stop_server(port)
                    launcher.kill_existing_on_port(port)
                except Exception:
                    pass

                cwd = None
                if "editable" in method:
                    source_dir = update._module_path().parent.parent
                    if (source_dir / "pyproject.toml").exists():
                        cwd = str(source_dir)
                    elif (Path.cwd() / "pyproject.toml").exists():
                        cwd = str(Path.cwd())

                proc = await asyncio.create_subprocess_shell(
                    cmd_str,
                    cwd=cwd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )

                while True:
                    line = await proc.stdout.readline()
                    if not line:
                        break
                    text = line.decode("utf-8", errors="replace").rstrip()
                    if text:
                        self._log_lines.append(f"  {text}")
                        if len(self._log_lines) > 20:
                            self._log_lines = self._log_lines[-20:]
                        try:
                            self.query_one("#log", Static).update("\n".join(self._log_lines))
                        except Exception:
                            pass

                await proc.wait()

                if proc.returncode == 0:
                    self._status = "[bold green][OK] AutoConduck upgraded successfully! Please restart AutoConduck.[/bold green]"
                else:
                    self._status = f"[bold red]Upgrade failed with exit code {proc.returncode}.[/bold red]"
            except Exception as exc:
                self._status = f"[bold red]Upgrade error: {exc}[/bold red]"
                self._log_lines.append(f"[bold red]Error: {exc}[/bold red]")
            finally:
                self._running = False
                try:
                    self.query_one("#status", Static).update(self._status)
                    self.query_one("#log", Static).update("\n".join(self._log_lines))
                except Exception:
                    pass

    UpdateCatalogScreen = UpdateScreen

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
    class UpdateScreen(Screen):  # type: ignore
        def __init__(self, *args, **kwargs):
            _require()

    class UpdateCatalogScreen(Screen):  # type: ignore
        def __init__(self, *args, **kwargs):
            _require()

    class LaunchAgentScreen(Screen):  # type: ignore
        def __init__(self, *args, **kwargs):
            _require()

    class DrillDownScreen(Screen):  # type: ignore
        def __init__(self, *args, **kwargs):
            _require()
