"""Pricing, closest-cost selection, phase bands, and catalog presets unit tests."""
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


# ---------------------------------------------------------------------------
# Pricing & Closest-Cost Selection
# ---------------------------------------------------------------------------


def test_select_closest_basic():
    cfg = Config(
        model_list=[
            {"id": "cheap", "price_in": 0.1, "price_out": 0.2, "enabled": True},
            {"id": "mid", "price_in": 1.0, "price_out": 3.0, "enabled": True},
            {"id": "expensive", "price_in": 10.0, "price_out": 30.0, "enabled": True},
        ]
    )
    pool = pricing.pool_ids(cfg)
    assert pricing.select_closest(pool, 0.1, cfg) == "cheap"
    assert pricing.select_closest(pool, 0.5, cfg) == "mid"
    assert pricing.select_closest(pool, 0.9, cfg) == "expensive"


def test_select_closest_respects_max_scaled_cost_cap():
    cfg = Config(
        model_list=[
            {"id": "cheap", "price_in": 0.1, "price_out": 0.2, "enabled": True},
            {"id": "expensive", "price_in": 10.0, "price_out": 30.0, "enabled": True},
        ],
        selection=SelectionConfig(max_file_read_scaled_cost=0.55),
    )
    pool = pricing.pool_ids(cfg)
    assert pricing.select_closest(pool, 0.90, cfg, max_scaled_cost=0.55) == "cheap"


def test_expensive_model_ceiling():
    cfg = Config(
        model_list=[
            {"id": "cheap-fast-model", "price_in": 0.1, "price_out": 0.2, "enabled": True},
            {"id": "expensive-mega-model", "price_in": 100.0, "price_out": 300.0, "enabled": True},
        ],
        selection=SelectionConfig(max_file_read_scaled_cost=0.55),
    )
    assert pricing.is_expensive_model("expensive-mega-model", cfg) is True
    assert pricing.is_expensive_model("cheap-fast-model", cfg) is False


def test_ema_realized_cost_blend():
    cfg = Config(
        model_list=[
            {"id": "adaptive-model", "price_in": 1.0, "price_out": 2.0, "enabled": True}
        ]
    )
    pricing.record_usage("adaptive-model", 100, 200, cost=0.05)
    pricing.record_usage("adaptive-model", 100, 200, cost=0.06)
    pricing.record_usage("adaptive-model", 100, 200, cost=0.07)
    effective = pricing._entry_effective_value("adaptive-model", cfg)
    assert effective > 0.0


def test_degraded_model_exclusion():
    cfg = Config(
        model_list=[
            {"id": "failing-model", "price_in": 0.5, "price_out": 0.5, "enabled": True},
            {"id": "backup-model", "price_in": 0.6, "price_out": 0.6, "enabled": True},
        ]
    )
    # Simulate high error rate
    for _ in range(5):
        pricing.record_error("failing-model")
    assert pricing.is_degraded("failing-model", window_s=300, error_rate=0.2) is True
    pool = pricing.pool_ids(cfg)
    degraded = {m for m in pool if pricing.is_degraded(m, window_s=300, error_rate=0.2)}
    selected = pricing.select_closest(pool, 0.5, cfg, degraded=degraded)
    assert selected == "backup-model"


# ---------------------------------------------------------------------------
# Phase Bands
# ---------------------------------------------------------------------------


def test_phase_band_filters_pool_planner():
    cfg = Config(
        model_list=[
            {"id": "cheap", "price_in": 0.01, "price_out": 0.01, "enabled": True},
            {"id": "flagship", "price_in": 5.0, "price_out": 15.0, "enabled": True},
        ],
        selection=SelectionConfig(phase_bands={"planner": [0.55, 0.85]}),
    )
    pool = pricing.pool_ids(cfg)
    selected = pricing.select_closest(pool, 0.5, cfg, band=(0.55, 0.85))
    assert selected in pool


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
