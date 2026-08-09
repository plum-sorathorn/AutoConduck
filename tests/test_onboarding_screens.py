from autoconduck.tui.onboarding import (
    CustomProvidersScreen,
    LauncherIntegrationScreen,
    ModelSelectionScreen,
    MODELS_PLACEHOLDER,
    ProviderFormScreen,
    render_check_rows,
    render_model_rows,
    render_models_placeholder,
    render_provider_rows,
)


def test_onboarding_screens_do_not_override_textual_render_hook():
    for cls in (
        ModelSelectionScreen,
        CustomProvidersScreen,
        LauncherIntegrationScreen,
        ProviderFormScreen,
    ):
        assert "_render" not in cls.__dict__


def test_onboarding_render_helpers_return_expected_markup():
    model_rows = render_model_rows([{"id": "model-a", "tier": "fast"}], {"model-a"}, 0)
    provider_rows = render_provider_rows([{"provider": "custom-provider", "enabled": True}], 0)
    check_rows = render_check_rows(["opencode", "aider"], {"opencode"}, 0)

    assert isinstance(model_rows, str)
    assert "model-a" in model_rows
    assert "custom-provider" in provider_rows
    assert "\u2713" in provider_rows  # enabled checkmark
    assert "opencode" in check_rows
    assert "  aider" in check_rows
    assert "[reverse]" in check_rows


def test_render_models_placeholder_shows_format_and_hides_when_models_exist():
    assert render_models_placeholder(True) == ""
    shown = render_models_placeholder(False)
    assert "newline-separated" in shown
    assert "gpt-4o" in shown
