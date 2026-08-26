"""Core onboarding screens."""
from __future__ import annotations
from .helpers import *
from .helpers import _persist
from ..onboarding_models import models_for_provider, overrides_for_toggle, default_enabled_ids, search_match, apply_api_key
from autoconduck.config import get_config, resolve_api_key
from autoconduck.presets.model_presets import PRESETS
from .screens_custom import ApiKeyScreen, CustomProvidersScreen
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
    class OnboardingScreen(Screen):
        def __init__(self, app_controller=None):
            super().__init__()
            self.controller = app_controller
            self.detected = detect_agents()
            configured = {agent for agent in AGENTS if is_agent_configured(agent)}
            # Detection is informational only: agent integration must be an explicit
            # user choice in the onboarding flow.
            self.selected = set(configured) if configured else set()
            self.cursor = 0

        def compose(self):
            yield Vertical(
                Static("┌─ AutoConduck · Detected agents ─┐"),
                Static(
                    render_agent_rows(
                        list(AGENTS), self.detected, self.selected, self.cursor
                    ),
                    id="agents",
                    markup=True,
                ),
                Static("[↑/↓] move  [enter] toggle  [→] continue  [ctrl+c] quit"),
            )

        def on_key(self, e):
            if e.key == "down":
                self.cursor = move_cursor(self.cursor, 1, len(AGENTS))
            elif e.key == "up":
                self.cursor = move_cursor(self.cursor, -1, len(AGENTS))
            elif e.key == "enter":
                self.selected.symmetric_difference_update({AGENTS[self.cursor]})
            elif e.key == "right" and self.controller:
                self.controller.push_screen(
                    ModelSourceScreen(self.controller, self.selected)
                )
            self.query_one("#agents").update(
                render_agent_rows(
                    list(AGENTS), self.detected, self.selected, self.cursor
                )
            )

    class ModelSourceScreen(Screen):
        SOURCES = ["Anthropic", "OpenAI", "Google", "LLM Gateway", "Custom endpoint…"]

        def __init__(self, app_controller=None, selected=None):
            super().__init__()
            self.controller = app_controller
            if selected is None:
                self.agent_selected = {
                    agent
                    for agent in AGENTS
                    if is_agent_configured(agent) or agent in detect_agents()
                }
            else:
                self.agent_selected = set(selected)
            self.cursor = 0
            self.selected = set()

        def compose(self):
            yield Vertical(
                Static(
                    render_source_rows(self.SOURCES, self.selected, self.cursor),
                    id="sources",
                    markup=True,
                ),
                Static(
                    "[↑/↓] move · [enter/right] select · [left] back · [ctrl+c] quit"
                ),
            )

        def _choose(self):
            key = (
                "custom"
                if self.cursor == 5
                else (
                    "llmgateway"
                    if self.cursor == 3
                    else (
                        "custom"
                        if self.cursor == 4
                        else self.SOURCES[self.cursor].lower()
                    )
                )
            )
            if key == "custom":
                self.controller.push_screen(
                    CustomProvidersScreen(self.controller, self.agent_selected)
                )
            else:
                self.controller.push_screen(
                    ModelSelectionScreen(self.controller, self.agent_selected, key)
                )

        def on_key(self, e):
            if e.key == "down":
                self.cursor = move_cursor(self.cursor, 1, len(self.SOURCES))
            elif e.key == "up":
                self.cursor = move_cursor(self.cursor, -1, len(self.SOURCES))
            elif e.key in ("enter", "right"):
                self._choose()
            elif e.key == "left":
                self.controller.pop_screen()
            self.query_one("#sources").update(
                render_source_rows(self.SOURCES, self.selected, self.cursor)
            )

    class ModelSelectionScreen(Screen):
        BINDINGS = [Binding("/", "focus_search", "filter")]

        def __init__(self, controller, agents, key):
            super().__init__()
            self.controller = controller
            self.agents = set(agents)
            self.key = key
            self.models = models_for_provider(key, PRESETS)
            self.cursor = 0
            self.filtered_models = self.models
            self._error = None
            cfg = get_config()
            self.enabled = default_enabled_ids(
                self.models,
                cfg.preset_overrides.get(self.key),
                model_list=getattr(cfg, "model_list", None),
            )

        def compose(self):
            yield Vertical(
                Input(placeholder="type to filter models…", id="search"),
                Static(
                    render_model_rows(self.models, self.enabled, self.cursor),
                    id="models",
                    markup=True,
                ),
                Static(self._footer(), id="footer"),
            )

        def on_mount(self):
            try:
                self.query_one("#search", Input).focus()
            except Exception:
                pass

        def _footer(self):
            if self._error:
                return self._error
            return f"{len(self.enabled)}/{len(self.models)} selected · [enter] toggle · [right] confirm · [left] back · [ctrl+c] quit"

        def on_input_changed(self, event):
            if event.input.id == "search":
                term = event.value
                self.filtered_models = [
                    m for m in self.models if search_match(term, m["id"])
                ]
                self.cursor = 0
                self.query_one("#models").update(
                    render_model_rows(self.filtered_models, self.enabled, self.cursor)
                )

        def on_input_submitted(self, event):
            if event.input.id == "search":
                if self.filtered_models:
                    self.enabled.symmetric_difference_update(
                        {self.filtered_models[self.cursor]["id"]}
                    )
                    if self.enabled:
                        self._error = None
                    self.query_one("#models").update(
                        render_model_rows(
                            self.filtered_models, self.enabled, self.cursor
                        )
                    )
                    self.query_one("#footer").update(self._footer())
                try:
                    event.input.focus()
                except Exception:
                    pass

        def _confirm(self):
            if len(self.models) > 6 and not self.enabled:
                self._error = "Select at least one model (enter to toggle)"
                self.query_one("#footer").update(self._footer())
                return
            cfg = get_config()
            cfg.selected_presets = list(
                dict.fromkeys(cfg.selected_presets + [self.key])
            )
            cfg.preset_overrides[self.key] = overrides_for_toggle(
                self.key, self.models, self.enabled, cfg.preset_overrides.get(self.key)
            )
            _persist(cfg)
            self.controller.push_screen(
                ApiKeyScreen(self.controller, self.agents, self.key)
            )

        def _finish(self):
            from ..dashboard import MainMenuScreen

            self.controller.switch_screen(MainMenuScreen())

        def action_focus_search(self):
            self.query_one("#search", Input).focus()

        def on_key(self, e):
            if e.key == "down":
                self.cursor = move_cursor(self.cursor, 1, len(self.filtered_models))
                self.query_one("#models").update(
                    render_model_rows(self.filtered_models, self.enabled, self.cursor)
                )
                try:
                    self.query_one("#search", Input).focus()
                except Exception:
                    pass
                e.stop()
            elif e.key == "up":
                self.cursor = move_cursor(self.cursor, -1, len(self.filtered_models))
                self.query_one("#models").update(
                    render_model_rows(self.filtered_models, self.enabled, self.cursor)
                )
                try:
                    self.query_one("#search", Input).focus()
                except Exception:
                    pass
                e.stop()
            elif e.key == "enter" and self.filtered_models:
                self.enabled.symmetric_difference_update(
                    {self.filtered_models[self.cursor]["id"]}
                )
                if self.enabled:
                    self._error = None
                self.query_one("#models").update(
                    render_model_rows(self.filtered_models, self.enabled, self.cursor)
                )
                self.query_one("#footer").update(self._footer())
                try:
                    self.query_one("#search", Input).focus()
                except Exception:
                    pass
                e.stop()
            elif e.key == "right":
                self._confirm()
            elif e.key == "left":
                self.controller.pop_screen()
            self.query_one("#models").update(
                render_model_rows(self.filtered_models, self.enabled, self.cursor)
            )
            self.query_one("#footer").update(self._footer())

else:
    class OnboardingScreen(Screen):
        def __init__(self,*a,**k): _require_textual()
    ModelSourceScreen=OnboardingScreen
    ModelSelectionScreen=OnboardingScreen
