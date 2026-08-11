"""Textual onboarding screens and small rendering helpers."""

from __future__ import annotations
import shutil
from pathlib import Path
from .onboarding_models import (
    models_for_provider,
    overrides_for_toggle,
    upsert_custom_models,
    remove_custom_provider,
    default_enabled_ids,
    search_match,
    apply_api_key,
)
from autoconduck.config import get_config, save_config, resolve_api_key
from autoconduck.model_presets import PRESETS, curated_model_catalog
from autoconduck.model_presets import resolve_models

try:
    from autoconduck import launcher
except ImportError:
    launcher = None


def move_cursor(cursor: int, delta: int, length: int) -> int:
    return 0 if length <= 0 else max(0, min(length - 1, cursor + delta))


def render_agent_rows(names, detected, selected, cursor):
    return "\n".join(
        ("[reverse]" if i == cursor else "")
        + f"{'›' if i == cursor else ' '} {'✓' if n in selected else ' '} {n:16} {detected.get(n) or 'not found'}"
        + ("[/reverse]" if i == cursor else "")
        for i, n in enumerate(names)
    )


def render_source_rows(sources, selected, cursor, models=None):
    rows = ["┌─ Model Sources ─┐"] + [
        (
            f"[reverse]› {'✓' if i in selected else ' '} {s}[/reverse]"
            if i == cursor
            else f"  {'✓' if i in selected else ' '} {s}"
        )
        for i, s in enumerate(sources)
    ]
    return "\n".join(
        rows
        + ["", "Selected models for routing pool:"]
        + [f"  {m}" for m in (models or [sources[i] for i in sorted(selected)])]
        or ["  none"]
    )


def render_model_rows(models, enabled, cursor):
    def price(m):
        return f" · in ${format_price(m.get('price_in', 0))} · out ${format_price(m.get('price_out', 0))} /1M"

    return (
        "\n".join(
            (
                f"[reverse]› {'✓' if m['id'] in enabled else ' '} {m['id']} ({m.get('tier', 'balanced')}){price(m)}[/reverse]"
                if i == cursor
                else f"  {'✓' if m['id'] in enabled else ' '} {m['id']} ({m.get('tier', 'balanced')}){price(m)}"
            )
            for i, m in enumerate(models)
        )
        or "No models available for this provider."
    )


def format_price(value) -> str:
    value = float(value or 0)
    return f"{value:.3f}" if 0 < value < 1 else f"{value:.2f}"


def model_option_label(row) -> str:
    return f"{row['id']} (${format_price(row['price_in'])} / ${format_price(row['price_out'])} per 1M)"


def render_provider_rows(providers, cursor):
    def _name(p):
        return p.get("provider", "") if isinstance(p, dict) else str(p)

    def _on(p):
        return bool(p.get("enabled", True)) if isinstance(p, dict) else True

    return (
        "\n".join(
            (
                f"[reverse]› {'✓' if _on(p) else ' '} {_name(p)}[/reverse]"
                if i == cursor
                else f"  {'✓' if _on(p) else ' '} {_name(p)}"
            )
            for i, p in enumerate(providers)
        )
        or "No custom providers. Press n to add one."
    )


def render_check_rows(agents, checked, cursor):
    return (
        "\n".join(
            (
                f"[reverse]› {'✓' if a in checked else ' '} {a}[/reverse]"
                if i == cursor
                else f"  {'✓' if a in checked else ' '} {a}"
            )
            for i, a in enumerate(agents)
        )
        or "No eligible agents selected."
    )


def _models_value(widget):
    return getattr(widget, "text", None) or getattr(widget, "value", "")


MODELS_PLACEHOLDER = (
    "One model ID per line (newline-separated)\nExample:\n  gpt-4o\n  gpt-4o-mini"
)


def render_models_placeholder(has_models):
    if has_models:
        return ""
    return f"[dim]┌─ {MODELS_PLACEHOLDER}[/dim]"


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

    class Screen:
        pass

    ComposeResult = object
    Binding = None


def _require_textual():
    if not _TEXTUAL:
        raise RuntimeError("Textual is required to use the AutoConduck TUI")


AGENTS = ("claude_code", "opencode", "pi")


def detect_agents():
    home = Path.home()
    locations = {
        "claude_code": [home / ".claude" / "settings.json"],
        "opencode": [home / ".config" / "opencode" / "config.json"],
        "pi": [home / ".pi" / "agent" / "settings.json"],
    }
    commands = {"claude_code": "claude", "opencode": "opencode", "pi": "pi"}
    return {
        n: next(
            (str(p) for p in locations[n] if p.exists()),
            shutil.which(commands[n]) if n in commands else None,
        )
        for n in AGENTS
    }


def is_agent_configured(agent_id: str) -> bool:
    from ..agents import all_adapters

    adapter = next(
        (
            a
            for a in all_adapters()
            if a.id == agent_id or getattr(a, "binary_name", None) == agent_id
        ),
        None,
    )
    if not adapter:
        return False
    for p in adapter.config_paths():
        if p.exists():
            try:
                txt = p.read_text(encoding="utf-8").lower()
                if "autoconduck" in txt or "# begin autoconduck" in txt:
                    return True
            except Exception:
                pass
    return False


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
        SOURCES = ["Anthropic", "OpenAI", "Google", "LLM Gateway", "OpenCode Go", "Custom endpoint…"]

        def __init__(self, app_controller=None, selected=None):
            super().__init__()
            self.controller = app_controller
            self.agent_selected = set(selected or ())
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
                        "opencodego"
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
                self.models, cfg.preset_overrides.get(self.key)
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
            if len(self.models) > 5 and not self.enabled:
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
            from .dashboard import DashboardScreen

            self.controller.switch_screen(DashboardScreen())

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
                value = self.query_one("#api_key", Input).value
                if value.strip() and not value.strip().startswith("env:"):
                    from ..auth import set_provider_key

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
            from .dashboard import DashboardScreen

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
                from .dashboard import DashboardScreen

                self.controller.switch_screen(DashboardScreen())

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
                    placeholder="OpenAI / Default base_url (e.g. http://localhost:8000/v1)",
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
                        (
                            (["[reverse]› " if mark else "  "][0])
                            + model_option_label(r)
                            + f" · {r['provider']}"
                        )
                        + (["[/reverse]" if mark else ""][0])
                    )
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
                    ids = [
                        line.strip() for line in current.splitlines() if line.strip()
                    ]
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
                        "[yellow]⏳ Saving custom provider settings...[/yellow]"
                    )
                except Exception:
                    pass
                name = self.query_one("#provider").value.strip()
                base_url = self.query_one("#base_url").value.strip()
                try:
                    anthropic_base_url = self.query_one(
                        "#anthropic_base_url"
                    ).value.strip()
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
                    [
                        x
                        for x in cfg.custom_models
                        if x.get("provider") != old or not old
                    ],
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
                Static(
                    "[enter] toggle · [right] install · [left] back · [ctrl+c] quit"
                ),
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
            if launcher is not None:
                launcher.install_shims(sorted(self.checked))
            self._finish()

        def _finish(self):
            from .dashboard import MainMenuScreen

            self.controller.switch_screen(MainMenuScreen())
else:

    class OnboardingScreen(Screen):
        def __init__(self, *a, **k):
            _require_textual()

    ModelSourceScreen = OnboardingScreen
    ModelSelectionScreen = OnboardingScreen
    ApiKeyScreen = OnboardingScreen
    CustomProvidersScreen = OnboardingScreen
    ProviderFormScreen = OnboardingScreen
    LauncherIntegrationScreen = OnboardingScreen


def _persist(cfg):
    # Onboarding should not import LiteLLM's large registry just to persist a
    # user's selection. Runtime startup can enrich pricing lazily when needed.
    from autoconduck.model_presets import resolve_models

    models = resolve_models(cfg, use_litellm=False)
    cfg.model_list = [m.model_dump() for m in models]
    save_config(cfg)


def _delete_provider(provider):
    cfg = get_config()
    cfg.custom_models = remove_custom_provider(cfg.custom_models, provider)
    save_config(cfg)
