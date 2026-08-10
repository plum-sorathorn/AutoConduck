from autoconduck.tui.onboarding_models import (
    models_for_provider, overrides_for_toggle, remove_custom_provider,
    upsert_custom_models,
    default_enabled_ids,
    apply_api_key,
)

def test_toggle_overrides_and_custom_crud():
    models = [{"id": "a", "tier": "budget"}, {"id": "b", "tier": "balanced"}]
    assert [x["id"] for x in overrides_for_toggle("x", models, {"b"})] == ["b"]
    rows = upsert_custom_models([], "local", "http://localhost", "KEY", ["one", "one", "two"])
    assert [x["id"] for x in rows] == ["one", "two"]
    assert all(x.get("enabled") is True for x in rows)
    assert remove_custom_provider(rows, "local") == []

def test_custom_models_use_registry_prices(monkeypatch):
    monkeypatch.setattr(
        "autoconduck.tui.onboarding_models._ingest_litellm_costs",
        lambda: {"openai/registry-model": {"price_in": 1.25, "price_out": 4.5}},
    )
    rows = upsert_custom_models([], "local", "http://localhost", "KEY", ["registry-model", "unknown"])
    assert rows[0]["price_in"] == 1.25 and rows[0]["price_out"] == 4.5
    assert rows[1]["price_in"] == 0.001 and rows[1]["price_out"] == 0.002

def test_custom_models_distinguish_environment_names_from_literal_keys():
    env_rows = upsert_custom_models([], "local", "http://localhost", "LLMGATEWAY_API_KEY", ["env-model"])
    assert env_rows[0]["api_key_env"] == "LLMGATEWAY_API_KEY"
    assert "api_key" not in env_rows[0]

    literal_rows = upsert_custom_models([], "local", "http://localhost", "sk-abc123", ["literal-model"])
    assert literal_rows[0]["api_key"] == "sk-abc123"
    assert "api_key_env" not in literal_rows[0]

def test_large_presets_start_empty_but_preserve_overrides():
    models = [{"id": str(i)} for i in range(6)]
    assert default_enabled_ids(models, None) == set()
    assert default_enabled_ids(models, [{"id": "2"}, {"id": "missing"}]) == {"2"}

def test_apply_api_key_environment_name_is_immutable():
    entries = [{"id": "a", "api_key": "old", "provider": "openai"}]
    result = apply_api_key(entries, "MY_API_KEY")
    assert result == [{"id": "a", "provider": "openai", "api_key_env": "MY_API_KEY"}]
    assert entries == [{"id": "a", "api_key": "old", "provider": "openai"}]

def test_apply_api_key_literal_removes_environment_name():
    entries = [{"id": "a", "api_key_env": "OLD_KEY"}]
    assert apply_api_key(entries, "sk-lit") == [{"id": "a", "api_key": "sk-lit"}]

def test_apply_api_key_blank_value_returns_deep_copies():
    entries = [{"id": "a", "nested": {"value": 1}}]
    result = apply_api_key(entries, "  ")
    assert result == entries
    assert result is not entries
    assert result[0] is not entries[0]

def test_toggle_preserves_saved_authentication():
    models = [{"id": "a", "provider": "llmgateway"}]
    existing = [{"id": "a", "api_key": "sk-saved"}]
    assert overrides_for_toggle("llmgateway", models, {"a"}, existing)[0]["api_key"] == "sk-saved"


