from types import SimpleNamespace

import pytest

from autoconduck import auth
from autoconduck.config import resolve_api_key, provider_for


def test_auth_round_trip_and_env(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOCONDUCK_HOME", str(tmp_path))
    auth.save_auth({"openai": "literal", "gateway": "env:GATEWAY_KEY"})
    monkeypatch.setenv("GATEWAY_KEY", "resolved")
    assert auth.load_auth()["openai"] == "literal"
    assert auth.get_provider_key("gateway") == "resolved"


def test_resolution_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOCONDUCK_HOME", str(tmp_path))
    monkeypatch.setenv("OLD_KEY", "environment")
    entry = {"id": "openai/gpt", "api_key": "literal", "api_key_env": "OLD_KEY"}
    assert resolve_api_key(entry) == "literal"
    auth.set_provider_key("openai", "auth")
    assert resolve_api_key(entry) == "auth"
    auth.save_auth({})
    assert resolve_api_key({"id": "gpt", "api_key_env": "OLD_KEY"}) == "environment"


def test_provider_derivation(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOCONDUCK_HOME", str(tmp_path))
    cfg = SimpleNamespace(custom_models=[{"provider": "gateway", "base_url": "http://gateway"}])
    assert provider_for({"provider": "explicit", "id": "x"}, cfg) == "explicit"
    assert provider_for({"id": "anthropic/claude"}, cfg) == "anthropic"
    assert provider_for({"id": "model", "base_url": "http://gateway"}, cfg) == "gateway"


@pytest.mark.skipif(__import__("os").name == "nt", reason="POSIX permissions")
def test_auth_file_mode(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOCONDUCK_HOME", str(tmp_path))
    auth.save_auth({"x": "y"})
    assert auth.auth_path().stat().st_mode & 0o777 == 0o600


def test_malformed_auth_is_empty(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOCONDUCK_HOME", str(tmp_path))
    auth.auth_path().parent.mkdir(parents=True, exist_ok=True)
    auth.auth_path().write_text("[broken", encoding="utf-8")
    assert auth.load_auth() == {}
