from autoconduck.config import resolve_api_key


def test_resolve_api_key_unset_environment_is_empty(monkeypatch):
    monkeypatch.delenv("llmgtwy_test", raising=False)
    assert resolve_api_key({"api_key_env": "llmgtwy_test"}) == ""


def test_resolve_api_key_literal_wins(monkeypatch):
    monkeypatch.delenv("llmgtwy_test", raising=False)
    assert resolve_api_key({"api_key": "literal", "api_key_env": "llmgtwy_test"}) == "literal"


def test_resolve_api_key_environment(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "secret")
    assert resolve_api_key({"api_key_env": "TEST_API_KEY"}) == "secret"


def test_litellm_params_for_resolves_auth_key_without_config_entry(monkeypatch, tmp_path):
    from autoconduck import auth
    from autoconduck.config import Config
    from autoconduck.messages_api import litellm_params_for

    monkeypatch.setenv("AUTOCONDUCK_HOME", str(tmp_path))
    auth.save_auth({"openai": "sk-auth-key-123", "anthropic": "sk-ant-key-456"})
    cfg = Config()

    params_oai = litellm_params_for("gpt-4o-mini", cfg)
    assert params_oai == {"model": "openai/gpt-4o-mini", "api_key": "sk-auth-key-123"}

    params_ant = litellm_params_for("anthropic/claude-3-5-sonnet", cfg)
    assert params_ant == {"model": "anthropic/claude-3-5-sonnet", "api_key": "sk-ant-key-456"}
