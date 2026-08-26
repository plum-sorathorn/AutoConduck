"""Dynamic multidimensional capability matching for model routing."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from autoconduck.config import Config
from autoconduck.config.resolver import resolve_orchestrator_model
from autoconduck.presets.model_presets import PRESETS

logger = logging.getLogger(__name__)
FILTER_STAGES = ("tools", "reasoning", "context", "capability_floor", "cost", "price_cap")
CAPABILITY_DIMS = ("reasoning", "tool_reliability", "code_quality", "latency_class")
TASK_TYPE_WEIGHTS: dict[str, dict[str, float]] = {
    "chat": {"reasoning": 0.5, "tool_reliability": 0.1, "code_quality": 0.2, "latency_class": 0.2},
    "explain": {"reasoning": 0.5, "tool_reliability": 0.1, "code_quality": 0.3, "latency_class": 0.1},
    "routine": {"reasoning": 0.1, "tool_reliability": 0.4, "code_quality": 0.2, "latency_class": 0.3},
    "debug": {"reasoning": 0.4, "tool_reliability": 0.3, "code_quality": 0.2, "latency_class": 0.1},
    "refactor": {"reasoning": 0.3, "tool_reliability": 0.2, "code_quality": 0.4, "latency_class": 0.1},
    "full_workflow": {"reasoning": 0.3, "tool_reliability": 0.4, "code_quality": 0.2, "latency_class": 0.1},
    "git_ops": {"reasoning": 0.1, "tool_reliability": 0.6, "code_quality": 0.1, "latency_class": 0.2},
    "single_edit": {"reasoning": 0.3, "tool_reliability": 0.2, "code_quality": 0.4, "latency_class": 0.1},
    "multi_edit": {"reasoning": 0.3, "tool_reliability": 0.2, "code_quality": 0.4, "latency_class": 0.1},
    "recon": {"reasoning": 0.4, "tool_reliability": 0.2, "code_quality": 0.2, "latency_class": 0.2},
}
DEFAULT_TASK_TYPE = "chat"


def capability_fit(vector: dict[str, float], weights: dict[str, float] | None = None) -> float:
    weights = weights or TASK_TYPE_WEIGHTS[DEFAULT_TASK_TYPE]
    dominant = [dim for dim, weight in weights.items() if weight > 0.25]
    dims = dominant or list(weights)
    min_val = min((float(vector.get(dim, 0.0)) for dim in dims), default=0.0)
    weighted_sum = sum(weight * float(vector.get(dim, 0.0)) for dim, weight in weights.items())
    return float(max(0.0, min(1.0, min_val + 0.1 * weighted_sum)))


def task_weights(task_type: str | None) -> dict[str, float]:
    return TASK_TYPE_WEIGHTS.get(task_type or "", TASK_TYPE_WEIGHTS[DEFAULT_TASK_TYPE])


@dataclass
class CapabilitySLA:
    """SLA with an optional per-selection advertised price cap in USD per 1M tokens."""
    min_context: int = 0
    min_output_tokens: int = 0
    requires_tools: bool = False
    requires_reasoning: bool = False
    min_capability_score: float = 0.0
    max_cost: float = float('inf')
    exclude_models: list[str] = field(default_factory=list)
    # Effective per-token price is input_price + 0.5 * output_price.
    max_price_usd_per_mtok: float | None = None
    task_type: str | None = None


@dataclass
class SelectionInfo:
    model: str | None = None
    candidates_considered: int = 0
    candidates_excluded_by: dict[str, int] = field(default_factory=dict)
    binding_constraint: str | None = None
    min_capability_score_applied: float = 0.0
    spend_cap_engaged: bool = False
    fallback_reason: str | None = None
    capability_fit_applied: float | None = None
    binding_capability_dim: str | None = None


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
    max_usd_per_min: float | None = None
    capability_vector: dict[str, float] | None = None


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

        try:
            max_pool_size = max(1, int(getattr(getattr(self.config, "selection", None), "max_pool_size", 200)))
            pool = pool[:max_pool_size]
        except Exception:
            pass

        entries: list[ModelEntry] = []
        for item in pool:
            if isinstance(item, ModelEntry):
                if item.capability_vector is None and item.capability_score == 0.0:
                    self._seed_entry(item)
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
                    
                explicit_score = bool(item.get("capability_score")) or bool(preset.get("capability_score"))
                cap_score = float(item.get("capability_score") or preset.get("capability_score", 0.0))
                vector = item.get("capability_vector") or preset.get("capability_vector")
                
                if vector is None and not explicit_score:
                    vector, cap_score = self._seed_values(model_id, price_in, price_out, is_reasoning, tools)
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
                        max_usd_per_min=item.get("max_usd_per_min"),
                        capability_vector=vector,
                 )
                )
        return entries

    @staticmethod
    def _seed_values(model_id: str, price_in: float, price_out: float, is_reasoning: bool, tools: bool) -> tuple[dict[str, float], float]:
        import re
        match = re.search(r'(\d+(?:\.\d+)?)[bx]', model_id.lower())
        if match:
            params = float(match.group(1))
            bucket = 0.5 if params >= 65 else 0.4 if params >= 25 else 0.3 if params >= 10 else 0.2
            latency = 0.3 if params >= 65 else 0.4 if params >= 25 else 0.6 if params >= 10 else 0.8
        else:
            blended = price_in + price_out * 0.5
            bucket = 0.5 if blended >= 2.0 else 0.4 if blended >= 0.5 else 0.2
            latency = 0.5
        vector = {
            "reasoning": min(1.0, (0.8 if is_reasoning else bucket) if is_reasoning else bucket),
            "tool_reliability": 0.6 if tools else 0.3,
            "code_quality": min(1.0, bucket + (0.1 if any(x in model_id.lower() for x in ("coder", "code", "instruct", "opus", "sonnet", "deepseek", "qwen")) else 0.0)),
            "latency_class": latency,
        }
        return vector, round(sum(vector.values()) / 4, 3)

    @classmethod
    def _seed_entry(cls, entry: ModelEntry) -> None:
        vector, score = cls._seed_values(entry.id, entry.price_in, entry.price_out, entry.is_reasoning, entry.supports_tools)
        entry.capability_vector = vector
        entry.capability_score = score

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
        return self._select(sla, pseudo_model)[0]

    def select_by_sla_detailed(
        self, sla: CapabilitySLA, pseudo_model: str = "autoconduck"
    ) -> SelectionInfo:
        return self._select(sla, pseudo_model)[1]

    def _select(self, sla: CapabilitySLA, pseudo_model: str) -> tuple[str, SelectionInfo]:
        """Select a model and collect explainability without changing the hot path shape."""
        info = SelectionInfo(min_capability_score_applied=sla.min_capability_score)
        entries = self._get_model_entries()
        if not entries:
            fallback = resolve_orchestrator_model(self.config)
            info.model = fallback
            info.fallback_reason = "empty_pool"
            return fallback, info

        # 1. Filter enabled & non-degraded & excluded
        from autoconduck.routing import pricing
        eligible = [
            e for e in entries
            if e.enabled and not pricing.is_degraded(e.id) and e.id not in sla.exclude_models
        ]
        info.candidates_considered = len(eligible)
        binding = None
        if not eligible:
            eligible = [e for e in entries if e.enabled and e.id not in sla.exclude_models] or entries

        # 2. Filter by tools
        if sla.requires_tools:
            tool_supported = [e for e in eligible if e.supports_tools]
            if tool_supported:
                if len(tool_supported) < len(eligible):
                    info.candidates_excluded_by["tools"] = len(eligible) - len(tool_supported)
                    binding = "tools"
                eligible = tool_supported

        # 3. Filter by reasoning
        if sla.requires_reasoning:
            reasoning_supported = [e for e in eligible if e.is_reasoning]
            if reasoning_supported:
                if len(reasoning_supported) < len(eligible):
                    info.candidates_excluded_by["reasoning"] = len(eligible) - len(reasoning_supported)
                    binding = "reasoning"
                eligible = reasoning_supported

        # 4. Filter by min_context_window
        if sla.min_context > 0:
            ctx_matches = [e for e in eligible if e.context_window >= sla.min_context]
            if ctx_matches:
                if len(ctx_matches) < len(eligible):
                    info.candidates_excluded_by["context"] = len(eligible) - len(ctx_matches)
                    binding = "context"
                eligible = ctx_matches
            else:
                eligible = sorted(eligible, key=lambda e: -e.context_window)

        # 4.5 Filter by min_capability_score
        if sla.min_capability_score > 0.0:
            weights = task_weights(sla.task_type)
            def fit(entry: ModelEntry) -> float:
                return capability_fit(entry.capability_vector, weights) if entry.capability_vector is not None else entry.capability_score
            cap_matches = [e for e in eligible if fit(e) >= sla.min_capability_score]
            if cap_matches:
                if len(cap_matches) < len(eligible):
                    info.candidates_excluded_by["capability_floor"] = len(eligible) - len(cap_matches)
                    binding = "capability_floor"
                eligible = cap_matches
            else:
                eligible = sorted(eligible, key=lambda e: -fit(e))
                binding = "capability_floor"

        # 5. Filter by max_cost (only if there are eligible models below the ceiling)
        cost_matches = [e for e in eligible if self._entry_cost(e) <= sla.max_cost]
        if cost_matches:
            if len(cost_matches) < len(eligible):
                info.candidates_excluded_by["cost"] = len(eligible) - len(cost_matches)
                binding = "cost"
            eligible = cost_matches

        if not eligible:
            model = resolve_orchestrator_model(self.config) or entries[0].id
            info.model = model
            info.fallback_reason = "sla_emptied_pool"
            info.binding_constraint = binding
            return model, info

        pre_ceiling = eligible
        ceiling_enabled = sla.max_price_usd_per_mtok is not None or any(
            e.max_usd_per_min is not None for e in eligible
        )
        if ceiling_enabled:
            ceiling_matches = []
            for entry in eligible:
                allowed = sla.max_price_usd_per_mtok if sla.max_price_usd_per_mtok is not None else float("inf")
                if entry.max_usd_per_min is not None:
                    allowed = min(allowed, entry.max_usd_per_min)
                if self._entry_cost(entry) <= allowed:
                    ceiling_matches.append(entry)
            removed = len(pre_ceiling) - len(ceiling_matches)
            if removed:
                info.candidates_excluded_by["price_cap"] = removed
                info.spend_cap_engaged = True
                binding = "price_cap"
            if ceiling_matches:
                eligible = ceiling_matches
            else:
                info.spend_cap_engaged = True
                info.fallback_reason = "price_cap_emptied_pool"
                eligible = pre_ceiling

        if len(eligible) == 1:
            info.model = eligible[0].id
            self._set_capability_info(info, eligible[0], sla)
            info.binding_constraint = binding
            return eligible[0].id, info

        # Sort remaining models strictly by absolute cost ascending
        sorted_models = sorted(
            eligible,
            key=lambda e: (self._entry_cost(e), -e.context_window, e.id),
        )
        
        # If the user invoked a high-end pseudo-model explicitly, we might bias towards the top of the eligible list
        if "expensive" in str(pseudo_model):
            info.model = sorted_models[-1].id
            self._set_capability_info(info, sorted_models[-1], sla)
            info.binding_constraint = binding
            return sorted_models[-1].id, info
            
        info.model = sorted_models[0].id
        self._set_capability_info(info, sorted_models[0], sla)
        info.binding_constraint = binding
        return sorted_models[0].id, info

    @staticmethod
    def _set_capability_info(info: SelectionInfo, entry: ModelEntry, sla: CapabilitySLA) -> None:
        if entry.capability_vector is not None:
            info.capability_fit_applied = capability_fit(entry.capability_vector, task_weights(sla.task_type))
            dominant = [dim for dim, weight in task_weights(sla.task_type).items() if weight > 0.25]
            dims = dominant or list(CAPABILITY_DIMS)
            info.binding_capability_dim = min(dims, key=lambda dim: entry.capability_vector.get(dim, 0.0))
