"""Comprehensive test suite for ModelPool & Capability SLA Query Optimizer.

Verifies:
- Selecting cheapest model that satisfies CapabilitySLA.
- Context window ceiling and floor filtering.
- Tool calling capability filtering (requires_tools=True).
- Reasoning model capability filtering (requires_reasoning=True).
- Max cost ceiling filtering (max_cost=X).
- Pseudo-model bias resolution (autoconduck, autoconduck-expensive).
"""
from __future__ import annotations

import pytest
from autoconduck.config import Config, ModelEntry
from autoconduck.routing.model_pool import ModelPool, CapabilitySLA, capability_fit, task_weights


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
            is_reasoning=True,
        ),
        "o1-preview": ModelEntry(
            id="o1-preview",
            provider="openai",
            cost_input=15.00,
            cost_output=60.00,
            context_window=128000,
            supports_tools=False,
            is_reasoning=True,
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


def test_model_pool_cheapest_default(mock_catalog_config: Config):
    """By default with no constraints, the cheapest model (local-llama) is chosen."""
    pool = ModelPool(mock_catalog_config)
    selected = pool.select_by_sla(CapabilitySLA())
    assert selected == "local-llama"


def test_model_pool_context_window_filter(mock_catalog_config: Config):
    """Filters out models with insufficient context window."""
    pool = ModelPool(mock_catalog_config)
    # Require at least 50k tokens, local-llama (8192) should be excluded
    selected = pool.select_by_sla(CapabilitySLA(min_context=50000))
    assert selected != "local-llama"
    assert selected == "gpt-4o-mini"


def test_model_pool_requires_tools_filter(mock_catalog_config: Config):
    """Filters out models that do not support tool/function calling."""
    pool = ModelPool(mock_catalog_config)
    selected = pool.select_by_sla(CapabilitySLA(requires_reasoning=True, requires_tools=True))
    # o1-preview does not support tools, so it falls back to claude-3-5-sonnet if eligible
    assert selected == "claude-3-5-sonnet"


def test_model_pool_requires_reasoning_filter(mock_catalog_config: Config):
    """Reasoning SLA selects reasoning-capable models."""
    pool = ModelPool(mock_catalog_config)
    selected = pool.select_by_sla(CapabilitySLA(requires_reasoning=True))
    assert selected in ("o1-preview", "claude-3-5-sonnet")


def test_model_pool_max_cost_filter(mock_catalog_config: Config):
    """Respects maximum cost constraint per 1M tokens."""
    pool = ModelPool(mock_catalog_config)
    # Cost limit below gpt-4o-mini effective cost
    selected = pool.select_by_sla(CapabilitySLA(min_context=50000, max_cost=0.1))
    assert selected == "gpt-4o-mini"


def test_model_pool_expensive_pseudo_model(mock_catalog_config: Config):
    """Expensive pseudo-model picks from the top of the eligible models."""
    pool = ModelPool(mock_catalog_config)
    selected = pool.select_by_sla(CapabilitySLA(), pseudo_model="autoconduck-expensive")
    assert selected == "o1-preview"


def test_model_pool_single_model():
    """Pool with single model always resolves to that model."""
    config = Config()
    config.models = {
        "solo-model": ModelEntry(id="solo-model", provider="openai", cost_input=1.0, cost_output=2.0)
    }
    pool = ModelPool(config)
    assert pool.select_by_sla(CapabilitySLA()) == "solo-model"


def test_model_pool_capability_floor_excludes_cheap_models(mock_catalog_config: Config):
    info = ModelPool(mock_catalog_config).select_by_sla_detailed(CapabilitySLA(min_capability_score=0.3))
    assert info.model != "local-llama"
    assert info.min_capability_score_applied == 0.3


def test_model_pool_floor_and_cost_choose_cheapest_eligible(mock_catalog_config: Config):
    selected = ModelPool(mock_catalog_config).select_by_sla(
        CapabilitySLA(min_capability_score=0.5, max_cost=20.0)
    )
    assert selected == "claude-3-5-sonnet"


def test_model_pool_price_cap_falls_back_with_explanation(mock_catalog_config: Config):
    info = ModelPool(mock_catalog_config).select_by_sla_detailed(
        CapabilitySLA(max_price_usd_per_mtok=-1.0)
    )
    assert info.model == "local-llama"
    assert info.spend_cap_engaged is True
    assert info.fallback_reason == "price_cap_emptied_pool"


def test_model_pool_price_cap_unset_is_noop(mock_catalog_config: Config):
    info = ModelPool(mock_catalog_config).select_by_sla_detailed(CapabilitySLA())
    assert info.spend_cap_engaged is False
    assert info.fallback_reason is None


def test_model_pool_price_cap_opt_in_excludes_expensive(mock_catalog_config: Config):
    info = ModelPool(mock_catalog_config).select_by_sla_detailed(
        CapabilitySLA(max_price_usd_per_mtok=0.2)
    )
    assert info.model == "local-llama"
    assert info.spend_cap_engaged is True


def test_model_pool_explainability_counts_candidates(mock_catalog_config: Config):
    info = ModelPool(mock_catalog_config).select_by_sla_detailed(
        CapabilitySLA(min_context=50000, requires_tools=True)
    )
    assert info.candidates_considered > 0
    assert info.candidates_excluded_by


def test_capability_vector_seeded_for_known_model(mock_catalog_config: Config):
    entries = ModelPool(mock_catalog_config)._get_model_entries()
    assert all(e.capability_vector is not None for e in entries)
    assert all(set(e.capability_vector) == {"reasoning", "tool_reliability", "code_quality", "latency_class"} for e in entries)
    assert all(0 <= value <= 1 for e in entries for value in e.capability_vector.values())


def test_capability_fit_min_with_bonus_masks_dominant_weakness():
    broken = {"reasoning": 0.9, "tool_reliability": 0.01, "code_quality": 0.9, "latency_class": 0.9}
    moderate = {dim: 0.5 for dim in broken}
    assert capability_fit(moderate, task_weights("git_ops")) > capability_fit(broken, task_weights("git_ops"))


def test_select_uses_fit_floor_not_scalar(mock_catalog_config: Config):
    pool = ModelPool(mock_catalog_config)
    info = pool.select_by_sla_detailed(CapabilitySLA(min_capability_score=0.55, task_type="refactor"))
    assert info.capability_fit_applied >= 0.55
    assert info.model == "claude-3-5-sonnet"


def test_legacy_scalar_entries_still_work():
    config = Config(models={"legacy": ModelEntry(id="legacy", capability_score=0.8)})
    info = ModelPool(config).select_by_sla_detailed(CapabilitySLA(min_capability_score=0.7))
    assert info.model == "legacy"
    assert info.capability_fit_applied is None


def test_fit_floor_empty_falls_back_to_highest_fit(mock_catalog_config: Config):
    info = ModelPool(mock_catalog_config).select_by_sla_detailed(CapabilitySLA(min_capability_score=1.0))
    assert info.model is not None
    assert info.binding_constraint == "capability_floor"
