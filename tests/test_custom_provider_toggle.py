"""Regression: custom provider add → list refresh on resume → toggle persists enabled."""
from __future__ import annotations

from types import SimpleNamespace

import yaml
import pytest
from textual.app import App

import autoconduck.config as config_mod
from autoconduck.config import get_config, load_config
from autoconduck.model_presets import resolve_models
from autoconduck.tui.onboarding import (
    CustomProvidersScreen,
    ProviderFormScreen,
    render_provider_rows,
)
from autoconduck.tui.onboarding_models import upsert_custom_models


class _FakeStatic:
    def __init__(self):
        self.content = ""

    def update(self, content):
        self.content = content


class _FakeInput:
    def __init__(self, value=""):
        self.value = value
        self.text = value


class _Controller:
    def __init__(self):
        self.stack = []
        self.popped = 0

    def push_screen(self, screen):
        self.stack.append(screen)

    def pop_screen(self):
        self.popped += 1
        if self.stack:
            self.stack.pop()

    def switch_screen(self, screen):
        self.stack = [screen]


def _reset_config_cache():
    config_mod._config = None
    config_mod._config_digest = None
    config_mod._config_path = None


def test_custom_provider_add_resume_toggle_persists(tmp_path, monkeypatch):
    """
    Approach: direct-method invocation (not Textual Pilot).
    Why: Pilot is impractical here without a full terminal App lifecycle for
    Input/TextArea focus; we call the exact handler methods the keys invoke
    with minimal fake events and widgets, isolating writes via AUTOCONDUCK_HOME.
    """
    monkeypatch.setenv("AUTOCONDUCK_HOME", str(tmp_path))
    _reset_config_cache()

    controller = _Controller()
    list_screen = CustomProvidersScreen(controller, set())
    custom_static = _FakeStatic()
    list_screen.query_one = lambda selector, *a, **k: custom_static  # type: ignore[method-assign]

    # --- add via ProviderFormScreen Enter path (real on_key body) ---
    form = ProviderFormScreen(controller, set())
    widgets = {
        "#provider": _FakeInput("my-local"),
        "#base_url": _FakeInput("http://127.0.0.1:9999"),
        "#api_key": _FakeInput("MY_KEY"),
        "#models": _FakeInput("model-one\nmodel-two"),
    }
    form.query_one = lambda selector, *a, **k: widgets[selector]  # type: ignore[method-assign]
    form.action_save()
    assert controller.popped == 1

    # upsert stores enabled=True; resolve_models ran before save
    cfg = get_config()
    assert all(row.get("enabled") is True for row in cfg.custom_models)
    assert {row["id"] for row in cfg.custom_models} == {"model-one", "model-two"}
    assert any(m.get("id") == "model-one" for m in cfg.model_list)

    # --- return → list visible WITHOUT keypress (on_screen_resume) ---
    list_screen.on_screen_resume()
    assert "my-local" in custom_static.content
    assert "\u2713" in custom_static.content  # enabled defaults True after upsert

    # toggle OFF (space) then ON (t) so checkmark reappears after toggle
    list_screen.on_key(SimpleNamespace(key="space"))
    assert "my-local" in custom_static.content
    off_row = render_provider_rows(list_screen._providers(), 0)
    assert "\u2713" not in off_row
    cfg_off = get_config()
    assert all(x.get("enabled") is False for x in cfg_off.custom_models if x.get("provider") == "my-local")

    list_screen.on_key(SimpleNamespace(key="t"))
    assert "\u2713" in custom_static.content
    assert "my-local" in custom_static.content

    # persisted on disk under tmp AUTOCONDUCK_HOME
    raw = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert all(
        row.get("enabled") is True
        for row in raw.get("custom_models", [])
        if row.get("provider") == "my-local"
    )

    # resolve_models pool reflects enabled after toggle
    cfg_on = load_config()
    pool = resolve_models(cfg_on)
    by_id = {m.id: m for m in pool}
    assert by_id["model-one"].enabled is True
    assert by_id["model-two"].enabled is True


def test_render_provider_rows_checkmark_for_enabled():
    on = render_provider_rows([{"provider": "p-on", "enabled": True}], 0)
    off = render_provider_rows([{"provider": "p-off", "enabled": False}], 0)
    assert "\u2713" in on and "p-on" in on
    assert "p-off" in off
    # disabled row uses a blank where the checkmark would be
    assert "\u2713" not in off


@pytest.mark.asyncio
async def test_provider_form_ctrl_enter_does_not_save(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOCONDUCK_HOME", str(tmp_path))
    _reset_config_cache()

    class _TestApp(App):
        def __init__(self, form):
            super().__init__()
            self.form = form

    app = _TestApp(None)
    form = ProviderFormScreen(app, set())
    app.form = form
    async with app.run_test() as pilot:
        app.push_screen(form)
        await pilot.pause()
        form.query_one("#provider").value = "pilot-provider"
        form.query_one("#base_url").value = "http://127.0.0.1:9999"
        form.query_one("#models").text = "pilot-model"
        await pilot.press("ctrl+enter")
        await pilot.pause()

    cfg = load_config()
    assert not any(row["provider"] == "pilot-provider" for row in cfg.custom_models)


@pytest.mark.asyncio
async def test_provider_form_enter_in_input_does_not_save(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOCONDUCK_HOME", str(tmp_path))
    _reset_config_cache()

    class _TestApp(App):
        def __init__(self, form):
            super().__init__()
            self.form = form

    app = _TestApp(None)
    form = ProviderFormScreen(app, set())
    app.form = form
    async with app.run_test() as pilot:
        app.push_screen(form)
        await pilot.pause()
        form.query_one("#provider").value = "enter-provider"
        form.query_one("#base_url").value = "http://127.0.0.1:9999"
        form.query_one("#models").text = "enter-model"
        form.query_one("#provider").focus()
        await pilot.press("enter")
        await pilot.pause()

    cfg = load_config()
    assert not any(row["provider"] == "enter-provider" for row in cfg.custom_models)


@pytest.mark.asyncio
async def test_provider_form_ctrl_s_in_models_uses_textual_dispatch(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOCONDUCK_HOME", str(tmp_path))
    _reset_config_cache()

    class _TestApp(App):
        def __init__(self, form):
            super().__init__()
            self.form = form

    app = _TestApp(None)
    form = ProviderFormScreen(app, set())
    app.form = form
    async with app.run_test() as pilot:
        app.push_screen(form)
        await pilot.pause()
        form.query_one("#provider").value = "ctrl-s-provider"
        form.query_one("#base_url").value = "http://127.0.0.1:9999"
        form.query_one("#models").text = "ctrl-s-model"
        form.query_one("#models").focus()
        await pilot.press("ctrl+s")
        await pilot.pause()

    cfg = load_config()
    assert any(row["provider"] == "ctrl-s-provider" for row in cfg.custom_models)
    assert "ctrl-s-provider" in {
        row["provider"] for row in CustomProvidersScreen(app, set())._providers()
    }
