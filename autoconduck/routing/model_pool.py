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

    def get_tier_for_entry(self, entry: ModelEntry) -> ModelTier:
        """Classify a ModelEntry into a tier based on input pricing."""
        cost = entry.cost_input if entry.cost_input > 0 else entry.price_in
        if cost < 0.50:
            return ModelTier.CHEAP_FAST
        elif cost <= 4.00:
            return ModelTier.BALANCED
        else:
            return ModelTier.FRONTIER_REASONING

    def select_for_tier(
        self,
        tier: ModelTier | str,
        min_context_window: int = 0,
        requires_tools: bool = False,
        pseudo_model: str = "autoconduck",
    ) -> str:
        """Select the best matching model ID for the requested tier and constraints."""
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

        # Group by tier
        tier_matches = [e for e in eligible if self.get_tier_for_entry(e) == target_tier]

        # Handle pseudo_model bias
        if not tier_matches:
            if target_tier == ModelTier.CHEAP_FAST:
                # Closest cheaper models
                tier_matches = sorted(eligible, key=lambda e: (e.cost_input or e.price_in))
            elif target_tier == ModelTier.FRONTIER_REASONING:
                # Closest more capable models
                tier_matches = sorted(eligible, key=lambda e: -(e.cost_input or e.price_in))
            else:
                tier_matches = eligible

        # Filter by min_context_window
        if min_context_window > 0:
            ctx_matches = [e for e in tier_matches if e.context_window >= min_context_window]
            if ctx_matches:
                tier_matches = ctx_matches
            else:
                all_ctx_matches = [e for e in eligible if e.context_window >= min_context_window]
                if all_ctx_matches:
                    tier_matches = all_ctx_matches
                else:
                    tier_matches = sorted(eligible, key=lambda e: -e.context_window)

        if not tier_matches:
            return resolve_orchestrator_model(self.config) or entries[0].id

        # If multiple in tier, use select_closest or deterministic priority
        if len(tier_matches) == 1:
            return tier_matches[0].id

        # Best candidate based on tier target
        if target_tier == ModelTier.CHEAP_FAST:
            best = min(tier_matches, key=lambda e: (e.cost_input or e.price_in, -e.context_window))
            return best.id
        elif target_tier == ModelTier.FRONTIER_REASONING:
            best = max(tier_matches, key=lambda e: (e.cost_input or e.price_in, e.context_window))
            return best.id
        else:
            # Balanced: median or select_closest
            return tier_matches[0].id
