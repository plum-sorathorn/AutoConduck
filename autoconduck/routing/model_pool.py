"""Autonomous Model Pool & 3-Tier Dynamic Model Selector.

Auto-tiers available models based on real-time pricing, context window capabilities,
and tool support:
- cheap_fast: < $0.50 / 1M tokens
- balanced: $0.50 – $4.00 / 1M tokens
- frontier_reasoning: > $4.00 / 1M tokens
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import Any

from autoconduck.config import Config, ModelEntry, get_config, resolve_orchestrator_model
from autoconduck.routing import pricing

logger = logging.getLogger(__name__)


class ModelTier(str, Enum):
    CHEAP_FAST = "cheap_fast"
    BALANCED = "balanced"
    FRONTIER_REASONING = "frontier_reasoning"


class ModelPool:
    """Dynamic model discovery, tier classification, and capability filter pool."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or get_config()

    def _get_model_entries(self) -> list[ModelEntry]:
        """Collect all configured ModelEntry items."""
        cfg = self.config
        entries: list[ModelEntry] = []

        if hasattr(cfg, "models") and isinstance(cfg.models, dict) and cfg.models:
            for k, v in cfg.models.items():
                if isinstance(v, ModelEntry):
                    entries.append(v)
                elif isinstance(v, dict):
                    entries.append(ModelEntry(**v))
            return entries

        # Otherwise read from model_list / custom_models
        from autoconduck.config import _configured_model_sources
        for item in _configured_model_sources(cfg):
            if isinstance(item, ModelEntry):
                entries.append(item)
            elif isinstance(item, dict):
                model_id = item.get("id") or item.get("model_name") or item.get("model")
                if not model_id:
                    continue
                price_in = float(item.get("cost_input") or item.get("price_in") or 0.0)
                price_out = float(item.get("cost_output") or item.get("price_out") or 0.0)
                ctx = int(item.get("context_window") or 128000)
                tools = bool(item.get("supports_tools", True))
                entries.append(
                    ModelEntry(
                        id=str(model_id),
                        provider=str(item.get("provider", "openai")),
                        price_in=price_in,
                        price_out=price_out,
                        cost_input=price_in,
                        cost_output=price_out,
                        context_window=ctx,
                        supports_tools=tools,
                        enabled=item.get("enabled", True),
                    )
                )
        return entries

    def _entry_cost(self, entry: ModelEntry) -> float:
        """Return the effective price per 1M tokens for a ModelEntry."""
        c_in = entry.cost_input if entry.cost_input > 0 else entry.price_in
        c_out = entry.cost_output if entry.cost_output > 0 else entry.price_out
        if c_in > 0 or c_out > 0:
            return float(c_in + 0.5 * c_out if c_out > 0 else c_in)
        return 0.0

    def get_tier_for_entry(self, entry: ModelEntry) -> ModelTier:
        """Classify a ModelEntry into a tier dynamically based on the active user model pool."""
        entries = self._get_model_entries()
        enabled = [e for e in entries if e.enabled]
        if not enabled:
            c = self._entry_cost(entry)
            if c < 0.50:
                return ModelTier.CHEAP_FAST
            elif c <= 4.00:
                return ModelTier.BALANCED
            else:
                return ModelTier.FRONTIER_REASONING

        if len(enabled) == 1:
            return ModelTier.BALANCED

        sorted_models = sorted(
            enabled,
            key=lambda e: (self._entry_cost(e), -e.context_window, e.id),
        )
        n = len(sorted_models)

        matching_indices = [i for i, e in enumerate(sorted_models) if e.id == entry.id]
        if matching_indices:
            idx = matching_indices[0]
            if n == 2:
                return ModelTier.CHEAP_FAST if idx == 0 else ModelTier.FRONTIER_REASONING
            elif n == 3:
                return (
                    ModelTier.CHEAP_FAST
                    if idx == 0
                    else (ModelTier.BALANCED if idx == 1 else ModelTier.FRONTIER_REASONING)
                )
            else:
                k1 = max(1, round(n / 3))
                k2 = max(k1 + 1, round(2 * n / 3))
                if idx < k1:
                    return ModelTier.CHEAP_FAST
                elif idx >= k2:
                    return ModelTier.FRONTIER_REASONING
                else:
                    return ModelTier.BALANCED

        # If entry is not in the configured pool, compare its cost to the dynamic cutoff boundaries
        cost = self._entry_cost(entry)
        if n == 2:
            mid = (self._entry_cost(sorted_models[0]) + self._entry_cost(sorted_models[1])) / 2.0
            return ModelTier.CHEAP_FAST if cost <= mid else ModelTier.FRONTIER_REASONING
        else:
            k1 = max(1, round(n / 3))
            k2 = max(k1 + 1, round(2 * n / 3))
            low_cutoff = self._entry_cost(sorted_models[k1 - 1])
            high_cutoff = self._entry_cost(sorted_models[k2]) if k2 < n else self._entry_cost(sorted_models[-1])
            if cost <= low_cutoff:
                return ModelTier.CHEAP_FAST
            elif cost >= high_cutoff:
                return ModelTier.FRONTIER_REASONING
            else:
                return ModelTier.BALANCED

    def select_for_tier(
        self,
        tier: ModelTier | str,
        min_context_window: int = 0,
        requires_tools: bool = False,
        pseudo_model: str = "autoconduck",
    ) -> str:
        """Select the best matching model ID dynamically partitioned from the user's selected models."""
        # Normalize tier string
        if isinstance(tier, str):
            try:
                target_tier = ModelTier(tier.lower())
            except (ValueError, KeyError):
                target_tier = ModelTier.BALANCED
        else:
            target_tier = tier

        entries = self._get_model_entries()
        if not entries:
            # Fallback when catalog is empty
            fallback = resolve_orchestrator_model(self.config)
            return fallback or "gpt-4o"

        # Filter enabled & non-degraded
        eligible: list[ModelEntry] = [
            e for e in entries
            if e.enabled and not pricing.is_degraded(e.id)
        ]
        if not eligible:
            eligible = [e for e in entries if e.enabled] or entries

        # Filter by tools if required
        if requires_tools:
            tool_supported = [e for e in eligible if e.supports_tools]
            if tool_supported:
                eligible = tool_supported

        # Filter by min_context_window
        if min_context_window > 0:
            ctx_matches = [e for e in eligible if e.context_window >= min_context_window]
            if ctx_matches:
                eligible = ctx_matches
            else:
                eligible = sorted(eligible, key=lambda e: -e.context_window)

        if not eligible:
            return resolve_orchestrator_model(self.config) or entries[0].id

        if len(eligible) == 1:
            return eligible[0].id

        # Sort eligible models by effective cost ascending
        sorted_models = sorted(
            eligible,
            key=lambda e: (self._entry_cost(e), -e.context_window, e.id),
        )
        n = len(sorted_models)

        # Dynamic Tier Selection across the user's filtered model pool:
        if n == 2:
            if target_tier == ModelTier.CHEAP_FAST:
                return sorted_models[0].id
            elif target_tier == ModelTier.FRONTIER_REASONING:
                return sorted_models[1].id
            else:
                # Balanced
                if "expensive" in str(pseudo_model):
                    return sorted_models[1].id
                return sorted_models[0].id

        if n == 3:
            if target_tier == ModelTier.CHEAP_FAST:
                return sorted_models[0].id
            elif target_tier == ModelTier.FRONTIER_REASONING:
                return sorted_models[2].id
            else:
                return sorted_models[1].id

        # For n >= 4: Dynamic Quantile Partitioning
        k1 = max(1, round(n / 3))
        k2 = max(k1 + 1, round(2 * n / 3))

        if target_tier == ModelTier.CHEAP_FAST:
            candidates = sorted_models[:k1]
            best = min(candidates, key=lambda e: (self._entry_cost(e), -e.context_window))
            return best.id
        elif target_tier == ModelTier.FRONTIER_REASONING:
            candidates = sorted_models[k2:]
            best = max(candidates, key=lambda e: (self._entry_cost(e), e.context_window))
            return best.id
        else:
            # Balanced: middle band
            candidates = sorted_models[k1:k2]
            if not candidates:
                candidates = sorted_models
            # Rank by closest to the median cost of the active eligible pool
            median_cost = (
                self._entry_cost(sorted_models[0]) + self._entry_cost(sorted_models[-1])
            ) / 2.0
            if "budget" in str(pseudo_model):
                return candidates[0].id
            elif "expensive" in str(pseudo_model):
                return candidates[-1].id
            best = min(
                candidates,
                key=lambda e: (
                    abs(self._entry_cost(e) - median_cost),
                    self._entry_cost(e),
                    -e.context_window,
                    e.id,
                ),
            )
            return best.id
