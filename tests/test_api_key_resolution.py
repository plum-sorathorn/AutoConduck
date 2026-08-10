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
