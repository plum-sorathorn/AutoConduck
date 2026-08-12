"""Regression tests for the split onboarding screen import surface."""
from __future__ import annotations

from autoconduck.tui.onboarding import (
    AGENTS,
    ApiKeyScreen,
    CustomProvidersScreen,
    LauncherIntegrationScreen,
    MODELS_PLACEHOLDER,
    ModelSelectionScreen,
    ModelSourceScreen,
    OnboardingScreen,
    ProviderFormScreen,
    detect_agents,
    format_price,
    is_agent_configured,
    move_cursor,
    render_agent_rows,
    render_provider_rows,
    render_source_rows,
)
from autoconduck.tui.onboarding.helpers import _delete_provider, _persist


class _Controller:
    def __init__(self):
        self.pushed = []
        self.popped = 0
        self.switched = []

    def push_screen(self, screen):
        self.pushed.append(screen)

    def pop_screen(self):
        self.popped += 1

    def switch_screen(self, screen):
        self.switched.append(screen)


def test_import_surface_complete():
    # The imports above are the assertion: every public onboarding symbol resolves.
    assert all(
        symbol is not None
        for symbol in (
            AGENTS,
            format_price,
            render_agent_rows,
            render_source_rows,
            render_provider_rows,
            move_cursor,
            detect_agents,
            is_agent_configured,
            MODELS_PLACEHOLDER,
            OnboardingScreen,
            ModelSourceScreen,
            ModelSelectionScreen,
            ApiKeyScreen,
            CustomProvidersScreen,
            ProviderFormScreen,
            LauncherIntegrationScreen,
        )
    )


def test_custom_provider_navigation_from_source_screen():
    controller = _Controller()
    screen = ModelSourceScreen(controller, {"opencode"})
    screen.cursor = 5  # The final row is "Custom endpoint…".

    screen._choose()

    assert len(controller.pushed) == 1
    assert isinstance(controller.pushed[0], CustomProvidersScreen)


def test_custom_providers_screen_importable():
    from autoconduck.tui.onboarding.screens_custom import CustomProvidersScreen as Imported

    screen = Imported(_Controller(), {"opencode"})
    assert isinstance(screen, CustomProvidersScreen)


def test_api_key_save_resolves_auth_import_and_finishes(monkeypatch):
    import autoconduck.tui.onboarding.screens_custom as screens_custom
    import autoconduck.tui.onboarding.helpers as helpers

    from autoconduck.tui.onboarding.screens_custom import apply_api_key

    assert apply_api_key is not None

    class Config:
        preset_overrides = {"openai": [{"id": "gpt-test", "provider": "openai"}]}

    cfg = Config()
    monkeypatch.setattr(screens_custom, "get_config", lambda: cfg)
    monkeypatch.setattr(screens_custom, "save_config", lambda value: None)
    monkeypatch.setattr(screens_custom, "_persist", lambda value: None)
    monkeypatch.setattr(helpers, "save_config", lambda value: None)
    monkeypatch.setattr("autoconduck.auth.set_provider_key", lambda key, value: None)

    screen = ApiKeyScreen(_Controller(), set(), "openai")

    class _Input:
        value = "sk-test"

        def blur(self):
            pass

    class _Error:
        def update(self, value):
            raise AssertionError(value)

    screen.query_one = lambda selector: _Input() if selector == "#api_key" else _Error()
    screen.action_save()

    assert len(cfg.preset_overrides["openai"]) == 1
    assert cfg.preset_overrides["openai"][0]["id"] == "gpt-test"
    assert len(screen.controller.switched) == 1
    assert screen.controller.switched[0].__class__.__name__ == "DashboardScreen"


def test_private_helpers_resolve_through_split_star_imports():
    from autoconduck.tui.onboarding.screens import _persist as screen_persist
    from autoconduck.tui.onboarding.screens_custom import _delete_provider as custom_delete

    assert screen_persist is _persist
    assert custom_delete is _delete_provider


def test_model_selection_confirm_persists_and_pushes_api_key(monkeypatch):
    import autoconduck.tui.onboarding.helpers as helpers
    import autoconduck.tui.onboarding.screens as screens

    class Config:
        selected_presets = []
        preset_overrides = {}
        model_list = []

    cfg = Config()
    monkeypatch.setattr(screens, "get_config", lambda: cfg)
    monkeypatch.setattr(helpers, "get_config", lambda: cfg)
    monkeypatch.setattr(helpers, "save_config", lambda value: None)
    monkeypatch.setattr(
        "autoconduck.model_presets.resolve_models",
        lambda value, use_litellm=False: [],
    )

    controller = _Controller()
    screen = ModelSelectionScreen(controller, {"opencode"}, "openai")
    screen.enabled = {screen.models[0]["id"]}
    screen._confirm()

    assert len(controller.pushed) == 1
    assert isinstance(controller.pushed[0], ApiKeyScreen)
