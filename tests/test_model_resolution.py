import os

from autoconduck.config import Config, orchestrator_litellm_params, qualify_model
from autoconduck.messages_api import litellm_params_for


def test_qualify_model_is_idempotent():
    assert qualify_model("deepseek-v4-flash") == "openai/deepseek-v4-flash"
    assert qualify_model("openai/deepseek-v4-flash") == "openai/deepseek-v4-flash"


def test_messages_params_qualify_and_normalize(monkeypatch):
    monkeypatch.setenv("LLM_KEY", "secret")
    cfg = Config(custom_models=[{"id": "deepseek-v4-flash", "base_url": "https://api.example", "api_key_env": "LLM_KEY"}])
    assert litellm_params_for("deepseek-v4-flash", cfg) == {
        "model": "openai/deepseek-v4-flash",
        "api_base": "https://api.example/v1",
        "api_key": "secret",
    }


def test_orchestrator_params_use_configured_model(monkeypatch):
    monkeypatch.setenv("LLM_KEY", "secret")
    cfg = Config(model_list=[{"id": "deepseek-v4-flash", "base_url": "https://api.example", "api_key_env": "LLM_KEY"}])
    assert orchestrator_litellm_params(cfg) == {
        "model": "openai/deepseek-v4-flash",
        "api_base": "https://api.example/v1",
        "api_key": "secret",
    }

def test_messages_params_use_literal_api_key_without_environment_name():
    cfg = Config(model_list=[{"id": "literal-model", "api_key": "sk-lit"}])
    assert litellm_params_for("literal-model", cfg) == {
        "model": "openai/literal-model",
        "api_key": "sk-lit",
    }

def test_orchestrator_params_use_literal_api_key_without_environment_name():
    cfg = Config(model_list=[{"id": "literal-model", "api_key": "sk-lit"}])
    assert orchestrator_litellm_params(cfg) == {
        "model": "openai/literal-model",
        "api_key": "sk-lit",
    }
