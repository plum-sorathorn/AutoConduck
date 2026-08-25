from autoconduck.tui.keymap import KEYMAP, QUIT_KEY
from autoconduck.tui.onboarding import (
    format_price,
    model_option_label,
    render_model_rows,
)
from autoconduck.tui.onboarding_models import search_match, upsert_custom_models


def test_keymap_ctrl_c_single_quit():
    assert QUIT_KEY == "ctrl+c"
    assert "ctrl+c" in KEYMAP


def test_search_match_helper():
    assert search_match("deepseek", "deepseek-v4-flash") is True
    assert search_match("sonnet", "claude-3-5-sonnet") is True
    assert search_match("xyz", "gpt-4o") is False


def test_format_price():
    assert format_price(0.15) == "0.150"
    assert format_price(0.0) == "0.00"


def test_upsert_custom_models():
    initial = [{"id": "existing-model", "provider": "openai"}]
    updated = upsert_custom_models(
        initial,
        provider="custom-llm",
        base_url="https://api.custom.com/v1",
        api_key_env="CUSTOM_KEY",
        model_ids=["custom-1", "custom-2"],
    )
    assert len(updated) == 3
    ids = {m["id"] for m in updated}
    assert "custom-1" in ids
    assert "custom-2" in ids


def test_update_screen_instantiation():
    from autoconduck.tui.dashboard_screens import UpdateScreen, UpdateCatalogScreen
    screen = UpdateScreen()
    assert screen._running is False
    assert UpdateCatalogScreen is UpdateScreen
