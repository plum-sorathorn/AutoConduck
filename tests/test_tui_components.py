from autoconduck.tui.dashboard_widgets import _format_box_lines, render_log_rows
from autoconduck.tui.keymap import KEYMAP, QUIT_KEY
from autoconduck.tui.onboarding_models import filter_catalog


def test_keymap_has_quit_and_navigation():
    assert QUIT_KEY == "ctrl+c"
    assert "down" in KEYMAP and "up" in KEYMAP


def test_dashboard_helpers_render_bounded_rows():
    rendered = render_log_rows([{"time": "now", "route": "fast", "model": "m", "prompt": "hello"}], 0)
    assert "now fast m hello" in rendered
    lines = _format_box_lines("Title", ["one", "two"], width=20)
    assert len(lines) == 4
    assert all(len(line) == 20 for line in lines)


def test_catalog_filter_matches_provider_capability_context_and_fuzzy_term():
    models = [
        {"id": "Acme/Reasoner-Pro", "provider": "acme", "context_window": 128000, "is_reasoning": True, "supports_tools": True},
        {"id": "other/basic", "provider": "other", "context_window": 8000},
    ]
    result = filter_catalog(models, term="reasoner pro", provider="acme", capabilities={"thinking", "tool use"}, min_context=100000)
    assert [row["id"] for row in result] == ["Acme/Reasoner-Pro"]


def test_model_catalog_screen_can_be_constructed():
    from autoconduck.tui.onboarding.screens_models import ModelCatalogScreen
    assert ModelCatalogScreen() is not None
