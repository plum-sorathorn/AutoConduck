from __future__ import annotations

import asyncio
import threading
import time
from typing import Optional

from .config import Config, save_config, load_config, get_config
from .model_presets import discover_models, PRESETS


def _has_textual() -> bool:
    try:
        import textual  # noqa: F401

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Textual TUI (gracefully degrades if not installed)
# ---------------------------------------------------------------------------

if _has_textual():
    from textual.app import App, ComposeResult
    from textual.containers import Vertical, Horizontal
    from textual.widgets import Header, Footer, Static, Button, Checkbox, Input, DataTable, Label
    from textual.screen import Screen

    class OnboardingScreen(Screen):
        def __init__(self, cfg: Config, port: int):
            super().__init__()
            self.cfg = cfg
            self.port = port
            self.selected_presets: set[str] = set()

        def compose(self) -> ComposeResult:
            yield Header()
            yield Static("[b]AutoConduck — Onboarding[/b]\n\nSelect model presets:", id="title")
            with Vertical():
                for key in ["openai", "anthropic", "google", "mistral"]:
                    yield Checkbox(f"{key} preset", id=f"preset-{key}")
                yield Static(f"Port: {self.port}", id="port-label")
                yield Input(placeholder="port", value=str(self.port), id="port-input")
            with Horizontal():
                yield Button("Continue", id="continue", variant="primary")
                yield Button("Skip (headless)", id="skip")
            yield Footer()

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "continue":
                # collect presets
                for key in ["openai", "anthropic", "google", "mistral"]:
                    try:
                        cb = self.query_one(f"#preset-{key}", Checkbox)
                        if cb.value:
                            self.selected_presets.add(key)
                    except Exception:
                        pass
                try:
                    port_val = int(self.query_one("#port-input", Input).value or str(self.port))
                except Exception:
                    port_val = self.port
                self.cfg.port = port_val
                if self.selected_presets:
                    models = discover_models(list(self.selected_presets))
                    self.cfg.models = models
                save_config(self.cfg)
                # patch agents
                self._patch_agents()
                self.app.push_screen(DashboardScreen(self.cfg))
            elif event.button.id == "skip":
                self.app.exit()

        def _patch_agents(self):
            try:
                from .agents import all_adapters

                for ad in all_adapters():
                    try:
                        if ad.detect():
                            ad.patch(self.cfg)
                    except Exception:
                        pass
            except Exception:
                pass

    class DashboardScreen(Screen):
        def __init__(self, cfg: Config):
            super().__init__()
            self.cfg = cfg

        def compose(self) -> ComposeResult:
            yield Header()
            yield Static("[b]AutoConduck — Live Dashboard[/b]  (q: quit, e: edit, r: re-patch)", id="dash-title")
            yield DataTable(id="events")
            yield Static("", id="stats")
            yield Footer()

        def on_mount(self) -> None:
            table = self.query_one("#events", DataTable)
            table.add_columns("ts", "path", "pseudo", "real", "T", "cost")
            self.set_interval(1.0, self.refresh_data)
            self.refresh_data()

        def refresh_data(self) -> None:
            try:
                from .telemetry import telemetry

                events = telemetry.recent(30)
                table = self.query_one("#events", DataTable)
                table.clear()
                for e in events[-20:]:
                    table.add_row(
                        time.strftime("%H:%M:%S", time.localtime(e.ts)),
                        e.path,
                        e.pseudo_model or "-",
                        (e.real_model or "")[:18],
                        f"{e.T_i:.2f}" if e.T_i is not None else "-",
                        f"{e.cost_est:.4f}" if e.cost_est else "-",
                    )
                stats_w = self.query_one("#stats", Static)
                s = telemetry.stats()
                stats_w.update(f"total={s.get('total_requests',0)}  fast_ratio={s.get('fast_path_ratio',0)}  avg_overhead={s.get('avg_overhead_ms',0)}ms")
            except Exception:
                pass

        def on_key(self, event) -> None:
            if event.key == "q":
                self.app.exit()
            elif event.key == "e":
                self.app.push_screen(EditScreen(self.cfg))
            elif event.key == "r":
                try:
                    from .agents import all_adapters

                    for ad in all_adapters():
                        try:
                            ad.patch(self.cfg)
                        except Exception:
                            pass
                except Exception:
                    pass

    class EditScreen(Screen):
        def __init__(self, cfg: Config):
            super().__init__()
            self.cfg = cfg

        def compose(self) -> ComposeResult:
            yield Header()
            yield Static("[b]Edit Models[/b]", id="edit-title")
            with Vertical():
                for key in ["openai", "anthropic", "google", "mistral"]:
                    yield Checkbox(f"{key}", id=f"edit-{key}")
                yield Input(placeholder="custom model id (comma separated)", id="custom")
            with Horizontal():
                yield Button("Save", id="save", variant="primary")
                yield Button("Cancel", id="cancel")
            yield Footer()

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "save":
                presets = []
                for key in ["openai", "anthropic", "google", "mistral"]:
                    try:
                        cb = self.query_one(f"#edit-{key}", Checkbox)
                        if cb.value:
                            presets.append(key)
                    except Exception:
                        pass
                custom_raw = ""
                try:
                    custom_raw = self.query_one("#custom", Input).value.strip()
                except Exception:
                    pass
                custom = []
                if custom_raw:
                    for mid in custom_raw.split(","):
                        mid = mid.strip()
                        if mid:
                            custom.append({"id": mid, "provider": "openai", "tier": "balanced"})
                models = discover_models(presets, custom)
                if models:
                    self.cfg.models = models
                    save_config(self.cfg)
                self.app.pop_screen()
            elif event.button.id == "cancel":
                self.app.pop_screen()

    class AutoConduckApp(App):
        CSS = """
        Screen { align: center middle; }
        #title, #dash-title, #edit-title { text-align: center; margin: 1; }
        DataTable { height: 1fr; }
        """

        def __init__(self, cfg: Config, port: int):
            super().__init__()
            self.cfg = cfg
            self.port = port
            self._proxy_thread: threading.Thread | None = None

        def on_mount(self) -> None:
            # start proxy in background thread
            self._proxy_thread = threading.Thread(target=self._run_proxy, daemon=True)
            self._proxy_thread.start()
            # decide initial screen
            if not self.cfg.models:
                self.push_screen(OnboardingScreen(self.cfg, self.port))
            else:
                self.push_screen(DashboardScreen(self.cfg))

        def _run_proxy(self):
            try:
                import uvicorn
                from .proxy import create_app

                app = create_app(self.cfg)
                uvicorn.run(app, host="127.0.0.1", port=self.cfg.port, log_level="warning", access_log=False)
            except Exception:
                pass

    def run_tui(cfg: Config, port: int):
        app = AutoConduckApp(cfg, port)
        app.run()

    def run_edit(cfg: Config):
        # simple edit without proxy
        from textual.app import App as TApp

        class EditApp(TApp):
            def on_mount(self):
                self.push_screen(EditScreen(cfg))

        EditApp().run()

else:
    # fallback no textual

    def run_tui(cfg: Config, port: int):
        print("[autoconduck] textual not installed, running headless proxy")
        import uvicorn
        from .proxy import create_app

        app = create_app(cfg)
        uvicorn.run(app, host="127.0.0.1", port=port, log_level=cfg.log_level)

    def run_edit(cfg: Config):
        print("[autoconduck] textual not installed, edit via config file at ~/.autoconduck/config.yaml")
