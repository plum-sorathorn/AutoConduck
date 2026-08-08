from autoconduck.tui.onboarding_models import (
    models_for_provider, overrides_for_toggle, remove_custom_provider,
    upsert_custom_models,
)

def test_toggle_overrides_and_custom_crud():
    models = [{"id": "a", "tier": "budget"}, {"id": "b", "tier": "balanced"}]
    assert [x["id"] for x in overrides_for_toggle("x", models, {"b"})] == ["b"]
    rows = upsert_custom_models([], "local", "http://localhost", "KEY", ["one", "one", "two"])
    assert [x["id"] for x in rows] == ["one", "two"]
    assert all(x.get("enabled") is True for x in rows)
    assert remove_custom_provider(rows, "local") == []


