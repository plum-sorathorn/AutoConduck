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


def test_in_memory_config_preservation_on_missing_file(tmp_path, monkeypatch):
    from autoconduck.config.manager import get_config, _has_configured_models
    import autoconduck.config.manager as manager_mod

    cfg_file = tmp_path / "config.yaml"
    cfg_data = {
        "model_list": [
            {"id": "qwen3.7-flash", "provider": "openai", "cost_input": 0.1, "cost_output": 0.2, "enabled": True}
        ]
    }
    cfg_file.write_text(yaml.dump(cfg_data), encoding="utf-8")
    monkeypatch.setattr(manager_mod, "config_path", lambda path=None: cfg_file)

    loaded = get_config()
    assert _has_configured_models(loaded)
    assert loaded.model_list[0]["id"] == "qwen3.7-flash"

    # Simulate config.yaml disappearing or truncated to 0 bytes
    cfg_file.unlink()

    retained = get_config()
    assert _has_configured_models(retained)
    assert retained.model_list[0]["id"] == "qwen3.7-flash"


def test_backup_recovery_on_cold_boot(tmp_path, monkeypatch):
    from autoconduck.config.manager import load_config, _has_configured_models
    import autoconduck.config.manager as manager_mod

    cfg_file = tmp_path / "config.yaml"
    backups_dir = tmp_path / "backups" / "config"
    backups_dir.mkdir(parents=True, exist_ok=True)

    bak_data = {
        "model_list": [
            {"id": "qwen3.7-flash", "provider": "openai", "cost_input": 0.1, "cost_output": 0.2, "enabled": True}
        ]
    }
    bak_file = backups_dir / "20260825-000000-000000.bak"
    bak_file.write_text(yaml.dump(bak_data), encoding="utf-8")

    monkeypatch.setattr(manager_mod, "config_path", lambda path=None: cfg_file)
    monkeypatch.setattr(manager_mod, "backups_dir", lambda agent=None: backups_dir)

    manager_mod._config = None
    manager_mod._config_digest = None
    manager_mod._config_path = None

    loaded = load_config(cfg_file)
    assert _has_configured_models(loaded)
    assert loaded.model_list[0]["id"] == "qwen3.7-flash"
    assert cfg_file.exists()


def test_smart_fallback_provider_env_vars(monkeypatch):
    import autoconduck.auth.auth as auth_mod
    from autoconduck.config.resolver import resolve_orchestrator_model

    empty_cfg = Config()
    monkeypatch.setattr(auth_mod, "load_auth", lambda: {})
    monkeypatch.setenv("LLMGATEWAY_API_KEY", "mock-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    resolved = resolve_orchestrator_model(empty_cfg)
    assert resolved == "qwen3.7-flash"

