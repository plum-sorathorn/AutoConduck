"""Health checks, OpenAI /v1/models listing, and /stats accounting handlers."""

from __future__ import annotations

import autoconduck.config as config_module
from autoconduck.stats import aggregate, load_records


def handle_healthz() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


async def handle_models(serve_model_ids: Any) -> dict[str, Any]:
    """List available and virtual routing model IDs."""
    return {
        "object": "list",
        "data": [
            {"id": m, "object": "model", "owned_by": "autoconduck"}
            for m in serve_model_ids(config_module.get_config())
        ],
    }


async def handle_stats(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate real-time usage and cost savings statistics."""
    usage = aggregate(load_records())
    return {
        "counts": decisions,
        "cost_saved_metered": 0.0,
        "cost_saved_subscription": 0.0,
        "cache_hit_ratio": 0.0,
        "usage": usage["totals"],
        "models": usage["models"],
        "path_counts": usage["paths"],
        "pseudo_counts": usage["pseudos"],
    }
