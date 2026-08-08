from autoconduck.tui.onboarding_models import (
    models_for_provider, overrides_for_toggle, remove_custom_provider,
    upsert_custom_models,
)

def test_toggle_overrides_and_custom_crud():
    models = [{"id": "a", "tier": "budget"}, {"id": "b", "tier": "balanced"}]
    assert [x["id"] for x in overrides_for_toggle("x", models, {"b"})] == ["b"]
    rows = upsert_custom_models([], "local", "http://localhost", "KEY", ["one", "one", "two"])
    assert [x["id"] for x in rows] == ["one", "two"]
    assert remove_custom_provider(rows, "local") == []

def test_devpass_models_are_normalized():
    rows = models_for_provider("devpass", {}, {"api_key_env": "KEY", "models": ["gateway-model"]})
    assert rows[0]["id"] == "gateway-model"
    assert rows[0]["api_key_env"] == "KEY"
