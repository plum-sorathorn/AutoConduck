"""Additional advanced onboarding screens."""

from __future__ import annotations

from autoconduck.config import get_config, resolve_api_key, save_config
from autoconduck.presets.model_presets import curated_model_catalog, resolve_models

from ..onboarding_models import search_match, upsert_custom_models
from .helpers import *


def _models_value(widget):
    return getattr(widget, "text", getattr(widget, "value", ""))


try:
    from autoconduck import launcher
except ImportError:
    launcher = None
try:
    from textual.app import ComposeResult
    from textual.binding import Binding
    from textual.containers import Vertical
    from textual.screen import Screen
    from textual.widgets import Input, Label, Static

    try:
        from textual.widgets import TextArea
    except ImportError:
        TextArea = Input
    _TEXTUAL = True
except ImportError:
    _TEXTUAL = False

    class Screen:
        pass

    ComposeResult = object
    Binding = None


class ProviderFormScreen(Screen):
    BINDINGS = [("ctrl+s", "save", "Save"), ("/", "focus_search", "filter")]

    def __init__(self, controller, agents, provider=None):
        super().__init__()
        self.controller = controller
        self.agents = agents
        self.provider = provider
        self._search_rows = []
        self._search_cursor = 0

    def compose(self):
        old = next(
            (
                x
                for x in get_config().custom_models
                if x.get("provider") == self.provider
            ),
            {},
        )
        models = "\n".join(
            x["id"]
            for x in get_config().custom_models
            if x.get("provider") == self.provider
        )
        api_key = old.get("api_key") or resolve_api_key(old)
        widgets = [
            Static("Custom provider"),
            Input(
                value=old.get("provider", self.provider or ""),
                placeholder="provider name",
                id="provider",
            ),
            Input(
                value=old.get("base_url", ""),
                placeholder="OpenAI / Default base_url (e.g. https://api.openai.com/v1 or https://gateway.ai/v1)",
                id="base_url",
            ),
            Input(
                value=old.get("anthropic_base_url", ""),
                placeholder="Anthropic base_url (optional)",
                id="anthropic_base_url",
            ),
            Input(
                value=api_key,
                placeholder="API key (or env:NAME for an environment variable)",
                id="api_key",
            ),
            Input(placeholder="type to filter models…", id="model_search"),
            Static("", id="model_results", markup=True),
        ]
        widgets.extend(
            [
                Label("One model ID per line (newline-separated)", classes="label"),
                TextArea(models, id="models"),
                *(
                    [
                        Static(
                            render_models_placeholder(False),
                            id="placeholder",
                            markup=True,
                        )
                    ]
                    if not models
                    else []
                ),
                Static("ctrl+s: save · left: cancel", id="error"),
            ]
        )
        yield Vertical(*widgets)

    def on_mount(self):
        self._render_results()

    def _render_results(self):
        term = self.query_one("#model_search", Input).value.lower()
        rows = [
            r
            for r in curated_model_catalog()
            if search_match(term, r["id"], r["provider"])
        ][:12]
        self._search_rows = rows
        if self._search_cursor >= len(rows):
            self._search_cursor = max(0, len(rows) - 1)
        lines = []
        for i, r in enumerate(rows):
            mark = i == self._search_cursor
            lines.append(
                (
                    (["[reverse]› " if mark else "  "][0])
                    + model_option_label(r)
                    + f" · {r['provider']}"
                )
                + (["[/reverse]" if mark else ""][0])
            )
        self.query_one("#model_results").update("\n".join(lines) or "No matches")

    def on_input_changed(self, event):
        if event.input.id == "model_search":
            self._search_cursor = 0
            self._render_results()

    def on_input_submitted(self, event):
        if event.input.id == "model_search":
            if getattr(self, "_search_rows", []):
                area = self.query_one("#models")
                current = _models_value(area)
                value = self._search_rows[self._search_cursor]["id"]
                ids = [line.strip() for line in current.splitlines() if line.strip()]
                if value not in ids:
                    area.text = (
                        current.rstrip() + "\n" if current.strip() else ""
                    ) + value
            event.input.focus()
            return

    def on_key(self, event):
        if event.key == "left":
            self.action_cancel()
            event.stop()
            return
        if (
            event.key in ("up", "down")
            and self.query_one("#model_search", Input).has_focus
        ):
            self._search_cursor = move_cursor(
                self._search_cursor,
                1 if event.key == "down" else -1,
                len(getattr(self, "_search_rows", [])),
            )
            self._render_results()
            event.stop()

    def action_focus_search(self):
        self.query_one("#model_search", Input).focus()

    def action_cancel(self):
        self.controller.pop_screen()

    def action_save(self):
        try:
            try:
                self.query_one("#error", Static).update(
                    "[yellow]Saving custom provider settings...[/yellow]"
                )
            except Exception:
                pass
            name = self.query_one("#provider").value.strip()
            base_url = self.query_one("#base_url").value.strip()
            try:
                anthropic_base_url = self.query_one("#anthropic_base_url").value.strip()
            except Exception:
                anthropic_base_url = ""
            api_key = self.query_one("#api_key").value.strip()
            model_ids = [
                line.strip()
                for line in _models_value(self.query_one("#models")).splitlines()
                if line.strip()
            ]
            if not name:
                raise ValueError("add a provider name")
            if not model_ids:
                raise ValueError("add at least one model id")
            cfg = get_config()
            old = self.provider
            cfg.custom_models = upsert_custom_models(
                [x for x in cfg.custom_models if x.get("provider") != old or not old],
                name,
                base_url,
                api_key,
                model_ids,
                anthropic_base_url=anthropic_base_url,
            )
            resolve_models(cfg)
            save_config(cfg)
            self.controller.pop_screen()
        except Exception as exc:
            try:
                self.query_one("#error").update(str(exc))
            except Exception:
                pass


class LauncherIntegrationScreen(Screen):
    ELIGIBLE = {"claude_code", "opencode", "pi"}

    def __init__(self, controller, selected=None):
        super().__init__()
        self.controller = controller
        self.agents = sorted(set(selected or ()) & self.ELIGIBLE)
        self.checked = set(self.agents)
        self.cursor = 0
        self.result = ""

    def compose(self):
        yield Vertical(
            Static("Launcher integration"),
            Static(
                render_check_rows(self.agents, self.checked, self.cursor),
                id="agents",
                markup=True,
            ),
            Static("[enter] toggle · [right] install · [left] back · [ctrl+c] quit"),
        )

    def on_key(self, e):
        if e.key == "down":
            self.cursor = move_cursor(self.cursor, 1, len(self.agents))
        elif e.key == "up":
            self.cursor = move_cursor(self.cursor, -1, len(self.agents))
        elif e.key in ("space", "enter"):
            self.checked.symmetric_difference_update({self.agents[self.cursor]})
        elif e.key == "left":
            self.controller.pop_screen()
        elif e.key == "right":
            self._install()
        self.query_one("#agents").update(
            render_check_rows(self.agents, self.checked, self.cursor)
        )

    def _install(self):
        try:
            self.query_one("#agents").update(
                "[bold cyan]Installing AutoConduck agent integrations… please wait…[/bold cyan]"
            )
        except Exception:
            pass
        configure_selected_agents(self.checked)
        from .screens_slm import SLMSetupScreen
        self.controller.push_screen(SLMSetupScreen(self.controller))

    def _finish(self):
        from ..dashboard import MainMenuScreen

        self.controller.switch_screen(MainMenuScreen())
