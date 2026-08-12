"""Advanced onboarding screens."""
from __future__ import annotations
from .helpers import *
from .helpers import _delete_provider, _persist
def _models_value(widget):
    return getattr(widget, "text", getattr(widget, "value", ""))
from ..onboarding_models import apply_api_key, upsert_custom_models, remove_custom_provider, search_match
from autoconduck.config import get_config, save_config, resolve_api_key
from autoconduck.model_presets import curated_model_catalog, resolve_models
try:
    from autoconduck import launcher
except ImportError:
    launcher=None
try:
    from textual.app import ComposeResult
    from textual.containers import Vertical
    from textual.screen import Screen
    from textual.binding import Binding
    from textual.widgets import Static, Input, Label
    try:
        from textual.widgets import TextArea
    except ImportError:
        TextArea = Input
    _TEXTUAL = True
except ImportError:
    _TEXTUAL = False
    class Screen: pass
    ComposeResult = object
    Binding = None

if _TEXTUAL:
        class ApiKeyScreen(Screen):
            BINDINGS = [("ctrl+s", "save", "Save")]
            DISPLAY_NAMES = {
                "anthropic": "Anthropic",
                "openai": "OpenAI",
                "google": "Google",
                "llmgateway": "LLM Gateway",
            }

            def __init__(self, controller, agents, key):
                super().__init__()
                self.controller = controller
                self.agents = set(agents)
                self.key = key

            def compose(self):
                overrides = get_config().preset_overrides.get(self.key, [])
                models = "\n".join(row.get("id", "") for row in overrides)
                initial = next(
                    (
                        row.get("api_key") or resolve_api_key(row)
                        for row in overrides
                        if row.get("api_key") or row.get("api_key_env")
                    ),
                    "",
                )
                yield Vertical(
                    Static(self.DISPLAY_NAMES.get(self.key, self.key)),
                    Static(models, markup=False),
                    Input(
                        value=initial,
                        placeholder="API key (or env:NAME for an environment variable)",
                        id="api_key",
                    ),
                    Static("", id="error"),
                    Static("ctrl+s: save · left: back · [ctrl+c] quit"),
                )

            def on_key(self, event):
                if event.key == "left":
                    self.controller.pop_screen()

            def on_input_submitted(self, event):
                if event.input.id == "api_key":
                    event.input.blur()

            def action_save(self):
                try:
                    cfg = get_config()
                    value = self.query_one("#api_key").value
                    if value.strip() and not value.strip().startswith("env:"):
                        from ...auth import set_provider_key

                        set_provider_key(self.key, value.strip())
                    cfg.preset_overrides[self.key] = apply_api_key(
                        cfg.preset_overrides.get(self.key, []), value
                    )
                    _persist(cfg)
                    if self.agents & LauncherIntegrationScreen.ELIGIBLE:
                        self.controller.push_screen(
                            LauncherIntegrationScreen(self.controller, self.agents)
                        )
                    else:
                        self._finish()
                except Exception as exc:
                    self.query_one("#error").update(str(exc))

            def _finish(self):
                from ..dashboard import DashboardScreen

                self.controller.switch_screen(DashboardScreen())

        class CustomProvidersScreen(Screen):
            def __init__(self, controller, agents):
                super().__init__()
                self.controller = controller
                self.agents = set(agents)
                self.cursor = 0
                self.confirm_delete = None

            def _providers(self):
                cfg = get_config()
                names = list(
                    dict.fromkeys(
                        x.get("provider", "")
                        for x in cfg.custom_models
                        if x.get("provider")
                    )
                )
                return [
                    {
                        "provider": n,
                        "enabled": any(
                            x.get("provider") == n and x.get("enabled", True)
                            for x in cfg.custom_models
                        ),
                    }
                    for n in names
                ]

            def on_screen_resume(self):
                self.query_one("#custom").update(
                    render_provider_rows(self._providers(), self.cursor)
                )

            def compose(self):
                yield Vertical(
                    Static(
                        render_provider_rows(self._providers(), self.cursor),
                        id="custom",
                        markup=True,
                    ),
                    Static(
                        "n: add · e: edit · d: delete · space: toggle · right: continue · left: back"
                    ),
                )

            def on_key(self, e):
                providers = self._providers()
                if self.confirm_delete:
                    if e.key in ("y", "enter"):
                        _delete_provider(self.confirm_delete)
                        self.confirm_delete = None
                        self.cursor = 0
                    elif e.key in ("n", "esc"):
                        self.confirm_delete = None
                    self.query_one("#custom").update(
                        render_provider_rows(self._providers(), self.cursor)
                    )
                    return
                if e.key == "down":
                    self.cursor = move_cursor(self.cursor, 1, len(providers))
                elif e.key == "up":
                    self.cursor = move_cursor(self.cursor, -1, len(providers))
                elif e.key in ("space", "t") and providers:
                    name = providers[self.cursor]["provider"]
                    cfg = get_config()
                    cur = any(
                        x.get("provider") == name and x.get("enabled", True)
                        for x in cfg.custom_models
                    )
                    for x in cfg.custom_models:
                        if x.get("provider") == name:
                            x["enabled"] = not cur
                    save_config(cfg)
                elif e.key == "n":
                    self.controller.push_screen(
                        ProviderFormScreen(self.controller, self.agents)
                    )
                elif e.key == "e" and providers:
                    self.controller.push_screen(
                        ProviderFormScreen(
                            self.controller, self.agents, providers[self.cursor]["provider"]
                        )
                    )
                elif e.key == "d" and providers:
                    self.confirm_delete = providers[self.cursor]["provider"]
                    self.query_one("#custom").update(f"Delete {self.confirm_delete}? [y/n]")
                elif e.key == "right":
                    if providers:
                        self._continue()
                elif e.key == "left":
                    self.controller.pop_screen()
                self.query_one("#custom").update(
                    render_provider_rows(self._providers(), self.cursor)
                )

            def _continue(self):
                cfg = get_config()
                cfg.selected_presets = list(
                    dict.fromkeys(cfg.selected_presets + ["custom"])
                )
                _persist(cfg)
                if self.agents & LauncherIntegrationScreen.ELIGIBLE:
                    self.controller.push_screen(
                        LauncherIntegrationScreen(self.controller, self.agents)
                    )
                else:
                    from ..dashboard import DashboardScreen

                    self.controller.switch_screen(DashboardScreen())


from .screens_extra import ProviderFormScreen, LauncherIntegrationScreen

if not _TEXTUAL:
    ApiKeyScreen=CustomProvidersScreen=ProviderFormScreen=LauncherIntegrationScreen=Screen
