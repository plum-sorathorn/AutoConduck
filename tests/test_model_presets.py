import builtins

import autoconduck.model_presets as model_presets


def test_ingest_litellm_costs_scans_once(monkeypatch):
    calls = 0
    fake_litellm = type(
        "FakeLiteLLM",
        (),
        {"model_cost": {"test-model": {"input_cost_per_token": 0.001, "output_cost_per_token": 0.002}}},
    )
    original_import = builtins.__import__

    def importing(name, *args, **kwargs):
        nonlocal calls
        if name == "litellm":
            calls += 1
            return fake_litellm
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(model_presets, "_litellm_costs_cache", None)
    monkeypatch.setattr(builtins, "__import__", importing)

    assert model_presets._ingest_litellm_costs() == model_presets._ingest_litellm_costs()
    assert calls == 1


def test_discover_models_keeps_custom_base_url():
    custom = [
        {
            "id": "my-custom-model",
            "provider": "openai",
            "base_url": "https://example.com/v1",
            "api_key_env": "MY_KEY",
            "enabled": True,
        }
    ]
    entries = model_presets.discover_models(preset_keys=["custom"], custom_models=custom, use_litellm=False)
    assert len(entries) == 1
    assert entries[0].id == "my-custom-model"
    assert entries[0].base_url == "https://example.com/v1"

def test_discover_models_preserves_literal_api_key():
    custom = [{"id": "literal-model", "api_key": "sk-lit", "provider": "openai"}]
    entries = model_presets.discover_models(preset_keys=["custom"], custom_models=custom, use_litellm=False)
    assert entries[0].api_key == "sk-lit"
    assert entries[0].model_dump()["api_key"] == "sk-lit"
