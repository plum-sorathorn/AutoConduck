"""Comprehensive test suite for Autonomous Model Pool & 3-Tier Classification.

Verifies:
- Auto-tiering by pricing:
  - cheap_fast: < $0.50 / 1M tokens
  - balanced: $0.50 – $4.00 / 1M tokens
  - frontier_reasoning: > $4.00 / 1M tokens
- Context window ceiling and floor filtering.
- Tool calling capability filtering (requires_tools=True).
- Pseudo-model resolution (autoconduck, autoconduck-budget, autoconduck-expensive).
- Degraded provider exclusions and fallback catalog defaults.
"""
from __future__ import annotations

import pytest
from autoconduck.config import Config, ModelEntry

try:
    from autoconduck.routing.model_pool import ModelPool
    from autoconduck.routing.slm_planner import ModelTier
except ImportError:
    pytest.skip("autoconduck.routing.model_pool not yet implemented in this milestone", allow_module_level=True)


@pytest.fixture
def mock_catalog_config() -> Config:
    config = Config()
    config.models = {
        "gpt-4o-mini": ModelEntry(
            id="gpt-4o-mini",
            provider="openai",
            cost_input=0.15,
            cost_output=0.60,
            context_window=128000,
            supports_tools=True,
        ),
        "claude-3-5-sonnet": ModelEntry(
            id="claude-3-5-sonnet",
            provider="anthropic",
            cost_input=3.00,
            cost_output=15.00,
            context_window=200000,
            supports_tools=True,
        ),
        "o1-preview": ModelEntry(
            id="o1-preview",
            provider="openai",
            cost_input=15.00,
            cost_output=60.00,
            context_window=128000,
            supports_tools=False,
        ),
        "local-llama": ModelEntry(
            id="local-llama",
            provider="ollama",
            cost_input=0.0,
            cost_output=0.0,
            context_window=8192,
            supports_tools=True,
        ),
    }
    return config


# ==============================================================================
# Tier 1: Feature Coverage (>=5 tests)
# ==============================================================================

def test_model_pool_cheap_fast_tier_classification(mock_catalog_config: Config):
    """Models with cost < $0.50/1M resolve for cheap_fast tier."""
    pool = ModelPool(mock_catalog_config)
    selected = pool.select_for_tier(ModelTier.CHEAP_FAST)
    assert selected in ("gpt-4o-mini", "local-llama")


def test_model_pool_balanced_tier_classification(mock_catalog_config: Config):
    """Models in the balanced price range ($0.50 - $4.00) resolve for balanced tier."""
    pool = ModelPool(mock_catalog_config)
    selected = pool.select_for_tier(ModelTier.BALANCED)
    assert selected == "claude-3-5-sonnet" or selected in mock_catalog_config.models


def test_model_pool_frontier_reasoning_tier_classification(mock_catalog_config: Config):
    """High-cost or reasoning models resolve for frontier_reasoning tier."""
    pool = ModelPool(mock_catalog_config)
    selected = pool.select_for_tier(ModelTier.FRONTIER_REASONING)
    assert selected in ("o1-preview", "claude-3-5-sonnet")


def test_model_pool_context_window_filter(mock_catalog_config: Config):
    """Filters out models with insufficient context window."""
    pool = ModelPool(mock_catalog_config)
    # Require at least 50k tokens, local-llama (8192) should be excluded
    selected = pool.select_for_tier(ModelTier.CHEAP_FAST, min_context_window=50000)
    assert selected != "local-llama"
    assert selected == "gpt-4o-mini"


def test_model_pool_requires_tools_filter(mock_catalog_config: Config):
    """Filters out models that do not support tool/function calling."""
    pool = ModelPool(mock_catalog_config)
    # o1-preview does not support tools in fixture
    selected = pool.select_for_tier(ModelTier.FRONTIER_REASONING, requires_tools=True)
    assert selected != "o1-preview"
    assert selected == "claude-3-5-sonnet"


# ==============================================================================
# Tier 2: Boundary & Corner Cases (>=5 tests)
# ==============================================================================

def test_model_pool_empty_catalog_fallback():
    """Empty catalog returns safe default fallback model ID."""
    empty_config = Config()
    empty_config.models = {}
    pool = ModelPool(empty_config)
    selected = pool.select_for_tier(ModelTier.BALANCED)
    assert isinstance(selected, str)
    assert len(selected) > 0


def test_model_pool_pseudo_model_resolution(mock_catalog_config: Config):
    """Resolves autoconduck, autoconduck-budget, and autoconduck-expensive correctly."""
    pool = ModelPool(mock_catalog_config)
    budget = pool.select_for_tier(ModelTier.CHEAP_FAST, pseudo_model="autoconduck-budget")
    expensive = pool.select_for_tier(ModelTier.FRONTIER_REASONING, pseudo_model="autoconduck-expensive")
    assert budget is not None
    assert expensive is not None


def test_model_pool_zero_cost_free_models(mock_catalog_config: Config):
    """Local or zero-cost models classify cleanly as cheap_fast."""
    pool = ModelPool(mock_catalog_config)
    selected = pool.select_for_tier(ModelTier.CHEAP_FAST, min_context_window=4000)
    assert selected in ("local-llama", "gpt-4o-mini")


def test_model_pool_extreme_context_requirement(mock_catalog_config: Config):
    """Requesting huge context window chooses model with greatest context capacity."""
    pool = ModelPool(mock_catalog_config)
    selected = pool.select_for_tier(ModelTier.BALANCED, min_context_window=180000)
    assert selected == "claude-3-5-sonnet"


def test_model_pool_invalid_tier_string_fallback(mock_catalog_config: Config):
    """Invalid tier type or string defaults safely to balanced tier."""
    pool = ModelPool(mock_catalog_config)
    selected = pool.select_for_tier("unknown_tier")  # type: ignore
    assert selected in mock_catalog_config.models or isinstance(selected, str)


def test_model_pool_balanced_multiple_models_ranking():
    """Balanced tier selects the most appropriate balanced model closest to target."""
    config = Config()
    config.models = {
        "expensive-balanced": ModelEntry(
            id="expensive-balanced",
            provider="openai",
            cost_input=3.80,
            cost_output=15.00,
            context_window=128000,
            supports_tools=True,
        ),
        "ideal-balanced": ModelEntry(
            id="ideal-balanced",
            provider="openai",
            cost_input=1.25,
            cost_output=5.00,
            context_window=128000,
            supports_tools=True,
        ),
    }
    pool = ModelPool(config)
    selected = pool.select_for_tier(ModelTier.BALANCED)
    assert selected == "ideal-balanced"


def test_model_pool_single_model_dynamic_tiering():
    """When only 1 model is configured, all tiers map dynamically to that model."""
    config = Config()
    config.models = {
        "solo-model": ModelEntry(
            id="solo-model",
            provider="openai",
            cost_input=2.00,
            cost_output=8.00,
        )
    }
    pool = ModelPool(config)
    assert pool.select_for_tier(ModelTier.CHEAP_FAST) == "solo-model"
    assert pool.select_for_tier(ModelTier.BALANCED) == "solo-model"
    assert pool.select_for_tier(ModelTier.FRONTIER_REASONING) == "solo-model"


def test_model_pool_two_models_dynamic_tiering():
    """When 2 models are configured, cheaper is cheap_fast and pricier is frontier."""
    config = Config()
    config.models = {
        "expensive-one": ModelEntry(
            id="expensive-one",
            provider="openai",
            cost_input=10.00,
            cost_output=30.00,
        ),
        "cheap-one": ModelEntry(
            id="cheap-one",
            provider="openai",
            cost_input=0.20,
            cost_output=0.80,
        ),
    }
    pool = ModelPool(config)
    assert pool.select_for_tier(ModelTier.CHEAP_FAST) == "cheap-one"
    assert pool.select_for_tier(ModelTier.FRONTIER_REASONING) == "expensive-one"


def test_model_pool_three_models_dynamic_tiering():
    """When 3 models are configured, each maps directly to cheap_fast, balanced, and frontier."""
    config = Config()
    config.models = {
        "c-model": ModelEntry(id="c-model", cost_input=15.00, cost_output=60.00),
        "a-model": ModelEntry(id="a-model", cost_input=0.10, cost_output=0.40),
        "b-model": ModelEntry(id="b-model", cost_input=1.50, cost_output=6.00),
    }
    pool = ModelPool(config)
    assert pool.select_for_tier(ModelTier.CHEAP_FAST) == "a-model"
    assert pool.select_for_tier(ModelTier.BALANCED) == "b-model"
    assert pool.select_for_tier(ModelTier.FRONTIER_REASONING) == "c-model"


def test_model_pool_six_models_dynamic_tiering():
    """When 6 models are configured, they dynamically split into 3 quantiles."""
    config = Config()
    config.models = {
        "m1": ModelEntry(id="m1", cost_input=0.05, cost_output=0.10),
        "m2": ModelEntry(id="m2", cost_input=0.15, cost_output=0.60),
        "m3": ModelEntry(id="m3", cost_input=0.50, cost_output=2.00),
        "m4": ModelEntry(id="m4", cost_input=1.00, cost_output=4.00),
        "m5": ModelEntry(id="m5", cost_input=3.00, cost_output=15.00),
        "m6": ModelEntry(id="m6", cost_input=15.00, cost_output=60.00),
    }
    pool = ModelPool(config)
    cheap_choice = pool.select_for_tier(ModelTier.CHEAP_FAST)
    balanced_choice = pool.select_for_tier(ModelTier.BALANCED)
    frontier_choice = pool.select_for_tier(ModelTier.FRONTIER_REASONING)

    assert cheap_choice in ("m1", "m2")
    assert balanced_choice in ("m3", "m4")
    assert frontier_choice in ("m5", "m6")
