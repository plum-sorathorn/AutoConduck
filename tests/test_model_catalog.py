"""Tests for curated_model_catalog / catalog_for_provider and the small
onboarding rendering helpers that consume them."""
import builtins

import pytest

from autoconduck import model_presets
from autoconduck.model_presets import (
    CATALOG_SHORTLIST,
    curated_model_catalog,
    catalog_for_provider,
)
from autoconduck.tui.onboarding import (
    format_price,
    render_model_rows,
    model_option_label,
)


def _fake_litellm(model_cost):
    return type("FakeLiteLLM", (), {"model_cost": model_cost})


def _patch_litellm_import(monkeypatch, model_cost):
    fake = _fake_litellm(model_cost)
    original_import = builtins.__import__

    def importing(name, *args, **kwargs):
        if name == "litellm":
            return fake
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importing)


def _reset_caches(monkeypatch):
    monkeypatch.setattr(model_presets, "_catalog_cache", None)
    monkeypatch.setattr(model_presets, "_litellm_costs_cache", None)


def test_catalog_normalizes_prices_per_1m(monkeypatch):
    _reset_caches(monkeypatch)
    _patch_litellm_import(monkeypatch, {
        "demo-model": {
            "input_cost_per_token": 0.0000025,
            "output_cost_per_token": 0.00001,
            "litellm_provider": "openai",
        }
    })
    catalog = curated_model_catalog()
    row = next(r for r in catalog if r["id"] == "demo-model")
    assert row["price_in"] == pytest.approx(2.5)
    assert row["price_out"] == pytest.approx(10.0)
    assert row["provider"] == "openai"


def test_catalog_excludes_non_chat_models(monkeypatch):
    _reset_caches(monkeypatch)
    _patch_litellm_import(monkeypatch, {
        "image-model": {
            "mode": "image_generation",
            "output_cost_per_image": 0.01,
            "output_cost_per_token": 0.0,
        },
        "embed-model": {
            "mode": "embedding",
            "input_cost_per_token": 1e-6,
            "output_cost_per_token": 1e-6,
        },
        "chat-model": {
            "mode": "chat",
            "input_cost_per_token": 1e-6,
            "output_cost_per_token": 2e-6,
        },
    })
    catalog = curated_model_catalog()
    ids = {r["id"] for r in catalog}
    assert "image-model" not in ids
    assert "embed-model" not in ids
    assert "chat-model" in ids


def test_catalog_dedup_prefers_bare_id(monkeypatch):
    _reset_caches(monkeypatch)
    _patch_litellm_import(monkeypatch, {
        "openai/gpt-4o": {
            "input_cost_per_token": 0.0000099,
            "output_cost_per_token": 0.0000199,
            "litellm_provider": "openai",
        },
        "gpt-4o": {
            "input_cost_per_token": 0.0000025,
            "output_cost_per_token": 0.00001,
            "litellm_provider": "openai",
        },
    })
    catalog = curated_model_catalog()
    matches = [r for r in catalog if r["id"] == "gpt-4o"]
    assert len(matches) == 1
    assert matches[0]["price_in"] == pytest.approx(2.5)
    assert matches[0]["price_out"] == pytest.approx(10.0)


def test_catalog_dedup_reverse_insertion_order_still_one_row(monkeypatch):
    _reset_caches(monkeypatch)
    _patch_litellm_import(monkeypatch, {
        "gpt-4o": {
            "input_cost_per_token": 0.0000025,
            "output_cost_per_token": 0.00001,
            "litellm_provider": "openai",
        },
        "openai/gpt-4o": {
            "input_cost_per_token": 0.0000099,
            "output_cost_per_token": 0.0000199,
            "litellm_provider": "openai",
        },
    })
    catalog = curated_model_catalog()
    matches = [r for r in catalog if r["id"] == "gpt-4o"]
    assert len(matches) == 1
    assert matches[0]["price_in"] == pytest.approx(2.5)
    assert matches[0]["price_out"] == pytest.approx(10.0)


def test_catalog_merges_presets_and_fallback_without_litellm(monkeypatch):
    _reset_caches(monkeypatch)
    monkeypatch.setattr(model_presets, "_ingest_litellm_costs", lambda *a, **k: {})

    catalog = curated_model_catalog()
    by_id = {r["id"]: r for r in catalog}

    assert by_id["gpt-4o"]["price_in"] == pytest.approx(2.5)
    assert by_id["gpt-4o"]["price_out"] == pytest.approx(10.0)

    assert by_id["claude-3-5-haiku-20241022"]["price_in"] == pytest.approx(0.25)
    assert by_id["claude-3-5-haiku-20241022"]["price_out"] == pytest.approx(1.25)

    assert by_id["gemini-1.5-flash"]["price_in"] == pytest.approx(0.075)
    assert by_id["gemini-1.5-flash"]["price_out"] == pytest.approx(0.30)

    assert by_id["deepseek-chat"]["price_in"] == pytest.approx(0.14)
    assert by_id["deepseek-chat"]["price_out"] == pytest.approx(0.28)


def test_catalog_for_provider_filters(monkeypatch):
    _reset_caches(monkeypatch)
    _patch_litellm_import(monkeypatch, {
        "oai-model": {
            "input_cost_per_token": 1e-6,
            "output_cost_per_token": 2e-6,
            "litellm_provider": "openai",
        },
        "anthro-model": {
            "input_cost_per_token": 1e-6,
            "output_cost_per_token": 2e-6,
            "litellm_provider": "anthropic",
        },
    })
    openai_rows = catalog_for_provider("openai")
    assert all(r["provider"].lower() == "openai" for r in openai_rows)
    assert any(r["id"] == "oai-model" for r in openai_rows)
    assert all(r["id"] != "anthro-model" for r in openai_rows)

    assert catalog_for_provider("does-not-exist") == []


def test_catalog_shortlist_subset(monkeypatch):
    _reset_caches(monkeypatch)
    fake_cost = {
        model_id: {
            "input_cost_per_token": 1e-6,
            "output_cost_per_token": 2e-6,
            "litellm_provider": "openai",
        }
        for model_id in CATALOG_SHORTLIST
    }
    _patch_litellm_import(monkeypatch, fake_cost)
    catalog = curated_model_catalog()
    ids = {r["id"] for r in catalog}
    for model_id in CATALOG_SHORTLIST:
        assert model_id in ids


def test_format_price():
    assert format_price(2.5) == "2.50"
    assert format_price(0.075) == "0.075"
    assert format_price(0) == "0.00"
    assert format_price(1) == "1.00"


def test_render_model_rows_shows_per_1m_prices():
    output = render_model_rows(
        [{"id": "model-a", "tier": "fast", "price_in": 2.5, "price_out": 10.0}],
        {"model-a"},
        0,
    )
    assert "/1M" in output
    assert "in $2.50" in output
    assert "out $10.00" in output
    assert "model-a" in output
    assert "[reverse]" in output


def test_dropdown_label_shape():
    row = {"id": "demo-model", "price_in": 2.5, "price_out": 10.0}
    assert model_option_label(row) == "demo-model ($2.50 / $10.00 per 1M)"
