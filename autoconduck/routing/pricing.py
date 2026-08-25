"""Pricing and SLA-based model selection tools."""
from __future__ import annotations

import logging
from typing import Any

from autoconduck.config.resolver import resolve_orchestrator_model
from autoconduck.routing.model_pool import ModelPool, CapabilitySLA, SelectionInfo

logger = logging.getLogger(__name__)


def pool_ids(config: Any) -> list[str]:
    """Extract IDs from config pool."""
    pool = getattr(config, "pool", [])
    if not pool and hasattr(config, "get"):
        pool = config.get("models.pool", [])
    if not pool:
        return []
    ids = []
    for item in pool:
        if isinstance(item, str):
            ids.append(item)
        elif isinstance(item, dict) and item.get("id"):
            ids.append(item["id"])
    return ids


def is_degraded(model_id: str) -> bool:
    """Placeholder for health checking degraded models."""
    return False

def record_error(model_id: str) -> None:
    """Placeholder for recording a model failure."""
    pass


def select_for_sla(
    sla: CapabilitySLA,
    config: Any = None,
    pseudo_model: str = "autoconduck",
) -> str:
    """Select best model based on CapabilitySLA using ModelPool."""
    if config:
        pool = ModelPool(config)
        return pool.select_by_sla(sla, pseudo_model)
    return resolve_orchestrator_model(config)


def select_for_sla_detailed(
    sla: CapabilitySLA, config: Any = None, pseudo_model: str = "autoconduck"
) -> SelectionInfo:
    if config:
        return ModelPool(config).select_by_sla_detailed(sla, pseudo_model)
    return SelectionInfo(model=resolve_orchestrator_model(config))
