"""SLM Engine selection, download, and integration onboarding screen."""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from .helpers import move_cursor, render_slm_rows
from .slm_progress import download_progress
from autoconduck.routing.slm_downloader import (
    SLM_MODELS_CATALOG,
    download_slm_model,
    integrate_slm_model,
    is_slm_model_installed,
)

logger = logging.getLogger(__name__)

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
                 Static("[dim]Connectivity checks pending...[/dim]", id="health", markup=True),
                 Static("", id="status", markup=True),
                Static("[↑/↓] move / select · [enter/→] confirm & continue · [←] back · [ctrl+c] quit"),
             )

        def on_mount(self):
            self.run_worker(self._probe_health(), exclusive=False)

        async def _probe_health(self):
            try:
                from .health import probe_connectivity, render_health_matrix
                from autoconduck.config import get_config, provider_for, resolve_api_key

                cfg = get_config()
                providers = []
                for entry in getattr(cfg, "model_list", []) or []:
                    if not isinstance(entry, dict):
                        continue
                    provider = provider_for(entry, cfg)
                    if any(row.get("name") == provider for row in providers):
                        continue
                    providers.append({
                        "name": provider,
                        "api_key": resolve_api_key(entry.get("api_key"), provider=provider),
                        "base_url": entry.get("api_base") or entry.get("base_url", ""),
                    })
                matrix = await probe_connectivity(providers, [getattr(cfg, "port", 11434)])
                self.query_one("#health").update(render_health_matrix(matrix))
            except Exception as exc:
                logger.debug("Onboarding connectivity probe unavailable: %s", exc)
                try:
                    self.query_one("#health").update("[dim]Connectivity checks unavailable; continuing offline.[/dim]")
                except Exception:
                    pass

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
                self.selected_id = self.models[self.cursor]["id"]
                self._update_view()
            elif e.key == "up":
                self.cursor = move_cursor(self.cursor, -1, len(self.models))
                self.selected_id = self.models[self.cursor]["id"]
                self._update_view()
            elif e.key == "space":
                self.selected_id = self.models[self.cursor]["id"]
                self._update_view()
            elif e.key in ("enter", "right"):
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
                state = download_progress(downloaded, total)
                pct = f"{state.percent}%" if state.percent is not None else f"{downloaded / (1024 * 1024):.1f} MB"
                size = f"/{total / (1024 * 1024):.1f} MB" if total > 0 else ""
                try:
                    self.query_one("#status").update(f"[bold cyan]Downloading {selected['name']}... {pct} ({downloaded / (1024 * 1024):.1f}{size})[/bold cyan]")
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
                logger.warning("SLM onboarding download failed; using heuristic fallback: %s", exc)
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
