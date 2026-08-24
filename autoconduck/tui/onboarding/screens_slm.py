"""SLM Engine selection, download, and integration onboarding screen."""
from __future__ import annotations

import asyncio
from typing import Any

from .helpers import move_cursor, render_slm_rows
from autoconduck.routing.slm_downloader import (
    SLM_MODELS_CATALOG,
    download_slm_model,
    integrate_slm_model,
    is_slm_model_installed,
)

try:
    from textual.app import ComposeResult
    from textual.containers import Vertical
    from textual.screen import Screen
    from textual.widgets import Static
    _TEXTUAL = True
except ImportError:
    _TEXTUAL = False
    class Screen: pass
    ComposeResult = object


if _TEXTUAL:
    class SLMSetupScreen(Screen):
        def __init__(self, controller, target_dir=None):
            super().__init__()
            self.controller = controller
            self.target_dir = target_dir
            self.models = SLM_MODELS_CATALOG
            self.cursor = 0
            self.selected_id = "qwen2.5-coder-0.5b-instruct"
            self._downloading = False

        def compose(self):
            curr = self.models[self.cursor]
            desc = curr.get("description", "")
            yield Vertical(
                Static("┌─ AutoConduck · Local SLM Engine Setup ─┐"),
                Static(
                    "AutoConduck uses an embedded Small Language Model (SLM) for sub-100ms task decomposition and dynamic orchestration.\n"
                    "Choose a local model to download and integrate, or skip to use the built-in heuristic fallback:"
                ),
                Static(
                    render_slm_rows(self.models, self.selected_id, self.cursor, target_dir=self.target_dir),
                    id="models",
                    markup=True,
                ),
                Static(f"[dim]ℹ {desc}[/dim]", id="description", markup=True),
                Static("", id="status", markup=True),
                Static("[↑/↓] move · [space] select · [enter/→] confirm & continue · [←] back · [ctrl+c] quit"),
            )

        def _update_view(self):
            try:
                curr = self.models[self.cursor]
                desc = curr.get("description", "")
                self.query_one("#models").update(
                    render_slm_rows(self.models, self.selected_id, self.cursor, target_dir=self.target_dir)
                )
                self.query_one("#description").update(f"[dim]ℹ {desc}[/dim]")
            except Exception:
                pass

        def on_key(self, e):
            if self._downloading:
                return

            if e.key == "down":
                self.cursor = move_cursor(self.cursor, 1, len(self.models))
                self._update_view()
            elif e.key == "up":
                self.cursor = move_cursor(self.cursor, -1, len(self.models))
                self._update_view()
            elif e.key == "space":
                self.selected_id = self.models[self.cursor]["id"]
                self._update_view()
            elif e.key in ("enter", "right"):
                # If cursor is on a different model and it's not selected yet, select it
                if self.selected_id != self.models[self.cursor]["id"]:
                    self.selected_id = self.models[self.cursor]["id"]
                    self._update_view()
                self._confirm()
            elif e.key == "left":
                if self.controller:
                    self.controller.pop_screen()

        def _confirm(self):
            selected = next((m for m in self.models if m["id"] == self.selected_id), None)
            if not selected or selected["id"] == "none":
                integrate_slm_model("none", target_dir=self.target_dir)
                self._finish()
                return

            if is_slm_model_installed(self.selected_id, target_dir=self.target_dir):
                integrate_slm_model(self.selected_id, target_dir=self.target_dir)
                try:
                    self.query_one("#status").update(
                        f"[bold green]Integrated {selected['name']} from local cache.[/bold green]"
                    )
                except Exception:
                    pass
                self._finish()
                return

            # Needs download
            self._downloading = True
            try:
                self.query_one("#status").update(
                    f"[bold cyan]Downloading {selected['name']} ({selected['size_mb']} MB)... please wait...[/bold cyan]"
                )
            except Exception:
                pass

            # Run in worker or task
            self.run_worker(self._async_download_and_integrate(selected), exclusive=True)

        async def _async_download_and_integrate(self, selected: dict[str, Any]):
            def progress(downloaded: int, total: int):
                if total > 0:
                    pct = int((downloaded / total) * 100)
                    mb_down = downloaded / (1024 * 1024)
                    mb_tot = total / (1024 * 1024)
                    try:
                        self.query_one("#status").update(
                            f"[bold cyan]Downloading {selected['name']}... {pct}% ({mb_down:.1f}/{mb_tot:.1f} MB)[/bold cyan]"
                        )
                    except Exception:
                        pass

            try:
                await asyncio.to_thread(
                    download_slm_model,
                    self.selected_id,
                    target_dir=self.target_dir,
                    progress_callback=progress,
                )
                integrate_slm_model(self.selected_id, target_dir=self.target_dir)
                try:
                    self.query_one("#status").update(
                        f"[bold green]{selected['name']} installed and integrated successfully![/bold green]"
                    )
                except Exception:
                    pass
                await asyncio.sleep(0.3)
                self._finish()
            except Exception as exc:
                self._downloading = False
                try:
                    self.query_one("#status").update(
                        f"[bold red]Download failed: {exc}. Using heuristic fallback.[/bold red]"
                    )
                except Exception:
                    pass

        def _finish(self):
            from ..dashboard import MainMenuScreen
            if self.controller:
                self.controller.switch_screen(MainMenuScreen())
else:
    class SLMSetupScreen(Screen):
        def __init__(self, *a, **k):
            from . import _require_textual
            _require_textual()
