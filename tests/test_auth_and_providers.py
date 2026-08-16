"""Authentication, provider key resolution, and custom endpoints unit tests."""
import os
import yaml
import pytest

from autoconduck import auth
from autoconduck.config import Config, resolve_api_key
from autoconduck.providers import CustomEndpoint, generate_litellm_config


def test_auth_save_and_load(tmp_path, monkeypatch):
    auth_file = tmp_path / "auth.yaml"
    monkeypatch.setattr(auth, "auth_path", lambda: auth_file)

    auth.set_provider_key("openai", "sk-test-key")
    assert auth.get_provider_key("openai") == "sk-test-key"

    content = yaml.safe_load(auth_file.read_text(encoding="utf-8"))
    assert content["providers"]["openai"] == "sk-test-key"


def test_resolve_api_key_precedence(tmp_path, monkeypatch):
    auth_file = tmp_path / "auth.yaml"
    auth_file.write_text(yaml.dump({"providers": {"openai": "sk-auth-key"}}), encoding="utf-8")
    monkeypatch.setattr(auth, "auth_path", lambda: auth_file)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-key")

    entry = {"provider": "openai", "api_key": "sk-literal", "api_key_env": "OPENAI_API_KEY"}

    # 1. auth.yaml takes highest precedence
    assert resolve_api_key(entry) == "sk-auth-key"

    # 2. When not in auth.yaml, literal api_key is second
    auth_file.write_text(yaml.dump({"providers": {}}), encoding="utf-8")
    assert resolve_api_key(entry) == "sk-literal"

    # 3. When no literal, env var is used
    entry.pop("api_key")
    assert resolve_api_key(entry) == "sk-env-key"


def test_auth_migration_from_config(tmp_path, monkeypatch):
    auth_file = tmp_path / "auth.yaml"
    monkeypatch.setattr(auth, "auth_path", lambda: auth_file)

    cfg = Config(
        model_list=[{"id": "m1", "provider": "deepseek", "api_key": "sk-secret-literal"}]
    )

    migrated = auth.migrate_from_config(cfg)
    assert migrated > 0
    assert auth.get_provider_key("deepseek") == "sk-secret-literal"


def test_custom_endpoint_and_litellm_config():
    endpoint = CustomEndpoint(
        display_name="My Custom Model",
        base_url="https://api.example.com/v1",
        api_key="sk-test",
    )
    llm_cfg = generate_litellm_config(endpoint, ["my-custom-model"])
    assert len(llm_cfg) == 1
    assert llm_cfg[0]["model_name"] == "my-custom-model"
