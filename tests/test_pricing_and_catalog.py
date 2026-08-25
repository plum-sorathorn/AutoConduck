"""Pricing, SLA selection, and catalog presets unit tests."""
import builtins
import pytest

from autoconduck import model_presets
from autoconduck.config import (
    Config,
    SelectionConfig,
    normalize_api_base,
    orchestrator_litellm_params,
    qualify_model,
)
from autoconduck.messages_api import litellm_params_for
from autoconduck.model_presets import (
    CATALOG_SHORTLIST,
    catalog_for_provider,
    clean_model_id,
    curated_model_catalog,
    discover_models,
)
from autoconduck.routing import pricing
from autoconduck.routing.model_pool import CapabilitySLA


# ---------------------------------------------------------------------------
# Pricing & Capability SLA Selection
# ---------------------------------------------------------------------------


def test_select_for_sla_basic():
    cfg = Config(
        model_list=[
            {"id": "cheap", "price_in": 0.1, "price_out": 0.2, "enabled": True},
            {"id": "mid", "price_in": 1.0, "price_out": 3.0, "enabled": True},
            {"id": "expensive", "price_in": 10.0, "price_out": 30.0, "enabled": True},
        ]
    )
    # Without constraints, picks cheapest
    assert pricing.select_for_sla(CapabilitySLA(), cfg) == "cheap"
    # With max_cost cap
    assert pricing.select_for_sla(CapabilitySLA(max_cost=0.5), cfg) == "cheap"
    # If expensive pseudo_model is used
    assert pricing.select_for_sla(CapabilitySLA(), cfg, pseudo_model="autoconduck-expensive") == "expensive"


def test_select_for_sla_context_window_filtering():
    cfg = Config(
        model_list=[
            {"id": "cheap-small", "price_in": 0.1, "price_out": 0.2, "context_window": 8000, "enabled": True},
            {"id": "expensive-large", "price_in": 10.0, "price_out": 30.0, "context_window": 128000, "enabled": True},
        ]
    )
    # Demanding 32k context filters out cheap-small
    assert pricing.select_for_sla(CapabilitySLA(min_context=32000), cfg) == "expensive-large"


def test_select_for_sla_tools_filtering():
    cfg = Config(
        model_list=[
            {"id": "cheap-no-tools", "price_in": 0.1, "price_out": 0.2, "supports_tools": False, "enabled": True},
            {"id": "mid-tools", "price_in": 1.0, "price_out": 2.0, "supports_tools": True, "enabled": True},
        ]
    )
    assert pricing.select_for_sla(CapabilitySLA(requires_tools=True), cfg) == "mid-tools"


# ---------------------------------------------------------------------------
# Model Catalog & Presets Ingestion
# ---------------------------------------------------------------------------


def test_clean_model_id():
    assert clean_model_id("us/meta-llama/llama-3-3-70b-instruct") == "llama-3-3-70b-instruct"
    assert clean_model_id("us.anthropic.claude-3-5-sonnet-20241022-v2:0") == "claude-3-5-sonnet-20241022-v2:0"
    assert clean_model_id("openai/gpt-4o") == "gpt-4o"
    assert clean_model_id("gpt-4o") == "gpt-4o"


def test_qualify_model_is_idempotent():
    assert qualify_model("deepseek-v4-flash") == "openai/deepseek-v4-flash"
    assert qualify_model("openai/deepseek-v4-flash") == "openai/deepseek-v4-flash"


def test_normalize_api_base_repairs_scheme():
    assert normalize_api_base("ttps://opencode.ai/zen/go/v1") == "https://opencode.ai/zen/go/v1"
    assert normalize_api_base("ttps://opencode.ai/zen/go") == "https://opencode.ai/zen/go/v1"
    assert normalize_api_base("opencode.ai/zen/go") == "https://opencode.ai/zen/go/v1"
    assert normalize_api_base("https://api.example/v1") == "https://api.example/v1"
    assert normalize_api_base("http://localhost:8000/v1") == "http://localhost:8000/v1"


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
    entries = discover_models(preset_keys=["custom"], custom_models=custom, use_litellm=False)
    assert len(entries) == 1
    assert entries[0].id == "my-custom-model"
    assert entries[0].base_url == "https://example.com/v1"


def test_discover_models_preserves_literal_api_key():
    custom = [{"id": "literal-model", "api_key": "sk-lit", "provider": "openai"}]
    entries = discover_models(preset_keys=["custom"], custom_models=custom, use_litellm=False)
    assert entries[0].api_key == "sk-lit"
    assert entries[0].model_dump()["api_key"] == "sk-lit"


def test_select_for_sla_integration():
    cfg = Config(
        model_list=[
            {"id": "flash-cheap", "price_in": 0.1, "price_out": 0.2, "context_window": 128000, "supports_tools": True, "enabled": True},
            {"id": "balanced-mid", "price_in": 1.0, "price_out": 2.5, "context_window": 128000, "supports_tools": True, "enabled": True},
            {"id": "frontier-expert", "price_in": 5.0, "price_out": 15.0, "context_window": 200000, "supports_tools": True, "is_reasoning": True, "enabled": True},
        ]
    )
    # Default SLA picks flash-cheap (cheapest capable)
    assert pricing.select_for_sla(CapabilitySLA(), cfg) == "flash-cheap"
    
    # Demanding reasoning picks frontier-expert
    assert pricing.select_for_sla(CapabilitySLA(requires_reasoning=True), cfg) == "frontier-expert"

    # Context window filtering > 150k
    assert pricing.select_for_sla(CapabilitySLA(min_context=150000), cfg) == "frontier-expert"
