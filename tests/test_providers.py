from autoconduck.providers import CustomEndpoint, discover_models, generate_litellm_config


def test_generate_config_resolves_environment_key(monkeypatch):
    monkeypatch.setenv("TEST_GATEWAY_KEY", "secret")
    endpoint = CustomEndpoint(display_name="Test", base_url="https://example.test", api_key_env="TEST_GATEWAY_KEY")
    entries = generate_litellm_config(endpoint, ["model-a"])
    params = entries[0]["litellm_params"]
    assert params["model"] == "openai/model-a"
    assert params["api_base"] == "https://example.test/v1"
    assert params["api_key"] == "secret"


def test_discover_models_failure_returns_empty(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("unreachable")
    monkeypatch.setattr("autoconduck.providers.httpx.get", fail)
    endpoint = CustomEndpoint(display_name="Test", base_url="https://unreachable.test")
    assert discover_models(endpoint) == []
