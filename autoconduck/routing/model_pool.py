"""Dynamic multidimensional capability matching for model routing."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from autoconduck.config import Config
from autoconduck.config.resolver import resolve_orchestrator_model
from autoconduck.presets.model_presets import PRESETS
from autoconduck.routing import pricing

logger = logging.getLogger(__name__)


@dataclass
class CapabilitySLA:
    """Service Level Agreement for model capabilities."""
    min_context: int = 0
    min_output_tokens: int = 0
    requires_tools: bool = False
    requires_reasoning: bool = False
    min_capability_score: float = 0.0
    max_cost: float = float('inf')
    exclude_models: list[str] = field(default_factory=list)


@dataclass
class ModelEntry:
    id: str
    provider: str
    price_in: float
    price_out: float
    cost_input: float
    cost_output: float
    context_window: int
    supports_tools: bool
    enabled: bool = True
    is_reasoning: bool = False
    capability_score: float = 0.0


class ModelPool:
    """Manages dynamic model routing based on Capability SLAs."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def _get_model_entries(self) -> list[ModelEntry]:
        """Fetch all configured models with their metadata."""
        pool = None
        if hasattr(self.config, "model_list") and self.config.model_list:
            pool = self.config.model_list
        elif hasattr(self.config, "get"):
            pool = self.config.get("models.pool") or self.config.get("model_list")
        if not pool and hasattr(self.config, "models") and self.config.models:
            if isinstance(self.config.models, dict):
                pool = list(self.config.models.values())
            elif getattr(self.config.models, "pool", None):
                pool = self.config.models.pool
            
        if not pool or not isinstance(pool, list):
            pool = [{"id": resolve_orchestrator_model(self.config)}]

        entries: list[ModelEntry] = []
        for item in pool:
            if isinstance(item, ModelEntry):
                entries.append(item)
                continue
            if hasattr(item, "model_dump"):
                item = item.model_dump()
            if isinstance(item, str):
                item = {"id": item}
            if isinstance(item, dict):
                model_id = item.get("id")
                if not model_id:
                    continue
                    
                # Get preset metadata if available
                preset = PRESETS.get(model_id, {})
                
                price_in = float(item.get("cost_input") or item.get("price_in") or preset.get("price_in", 0.0))
                price_out = float(item.get("cost_output") or item.get("price_out") or preset.get("price_out", 0.0))
                ctx = int(item.get("context_window") or preset.get("context_window", 128000))
                tools = bool(item.get("supports_tools", preset.get("supports_tools", True)))
                
                # Check if it's a reasoning model (by name or preset)
                is_reasoning = bool(item.get("is_reasoning", preset.get("is_reasoning", False)))
                if any(x in model_id.lower() for x in ["o1", "o3", "deepseek-r1", "reasoning"]):
                    is_reasoning = True
                    
                cap_score = float(item.get("capability_score") or preset.get("capability_score", 0.0))
                
                # Heuristic fallback for unknown models
                if cap_score == 0.0:
                    import re
                    # 1. Parameter Size Heuristic (Excellent for local/free models)
                    param_match = re.search(r'(\d+(?:\.\d+)?)[bx]', model_id.lower())
                    
                    if param_match:
                        params = float(param_match.group(1))
                        if params >= 65:  # e.g., 70b, 72b, 400b
                            cap_score = 0.5
                        elif params >= 25:  # e.g., 32b, 35b
                            cap_score = 0.4
                        elif params >= 10:  # e.g., 14b, 12b
                            cap_score = 0.3
                        else:  # e.g., 7b, 8b
                            cap_score = 0.2
                    else:
                        # 2. Pricing Heuristic (Excellent for proprietary API models)
                        blended_price = price_in + (price_out * 0.5)
                        if blended_price >= 2.0:  # e.g., Claude 3.5 Sonnet, GPT-4o
                            cap_score = 0.5
                        elif blended_price >= 0.5:  # e.g., DeepSeek Reasoner, O3-Mini
                            cap_score = 0.4
                        else:  # e.g., Flash, Mini, Haiku
                            cap_score = 0.2
                    
                    if is_reasoning:
                        cap_score += 0.1  # Boost reasoning models slightly
                entries.append(
                    ModelEntry(
                        id=str(model_id),
                        provider=str(item.get("provider") or preset.get("provider", "openai")),
                        price_in=price_in,
                        price_out=price_out,
                        cost_input=price_in,
                        cost_output=price_out,
                        context_window=ctx,
                        supports_tools=tools,
                        enabled=item.get("enabled", True),
                        is_reasoning=is_reasoning,
                        capability_score=cap_score,
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

    def select_by_sla(
        self,
        sla: CapabilitySLA,
        pseudo_model: str = "autoconduck",
    ) -> str:
        """Select the absolute cheapest model that satisfies the Capability SLA."""
        entries = self._get_model_entries()
        if not entries:
            fallback = resolve_orchestrator_model(self.config)
            return fallback

        # 1. Filter enabled & non-degraded & excluded
        eligible = [
            e for e in entries
            if e.enabled and not pricing.is_degraded(e.id) and e.id not in sla.exclude_models
        ]
        if not eligible:
            eligible = [e for e in entries if e.enabled and e.id not in sla.exclude_models] or entries

        # 2. Filter by tools
        if sla.requires_tools:
            tool_supported = [e for e in eligible if e.supports_tools]
            if tool_supported:
                eligible = tool_supported

        # 3. Filter by reasoning
        if sla.requires_reasoning:
            reasoning_supported = [e for e in eligible if e.is_reasoning]
            if reasoning_supported:
                eligible = reasoning_supported

        # 4. Filter by min_context_window
        if sla.min_context > 0:
            ctx_matches = [e for e in eligible if e.context_window >= sla.min_context]
            if ctx_matches:
                eligible = ctx_matches
            else:
                eligible = sorted(eligible, key=lambda e: -e.context_window)

        # 4.5 Filter by min_capability_score
        if sla.min_capability_score > 0.0:
            cap_matches = [e for e in eligible if getattr(e, "capability_score", 0.0) >= sla.min_capability_score]
            if cap_matches:
                eligible = cap_matches
            else:
                eligible = sorted(eligible, key=lambda e: -getattr(e, "capability_score", 0.0))

        # 5. Filter by max_cost (only if there are eligible models below the ceiling)
        cost_matches = [e for e in eligible if self._entry_cost(e) <= sla.max_cost]
        if cost_matches:
            eligible = cost_matches

        if not eligible:
            return resolve_orchestrator_model(self.config) or entries[0].id

        if len(eligible) == 1:
            return eligible[0].id

        # Sort remaining models strictly by absolute cost ascending
        sorted_models = sorted(
            eligible,
            key=lambda e: (self._entry_cost(e), -e.context_window, e.id),
        )
        
        # If the user invoked a high-end pseudo-model explicitly, we might bias towards the top of the eligible list
        if "expensive" in str(pseudo_model):
            return sorted_models[-1].id
            
        return sorted_models[0].id
