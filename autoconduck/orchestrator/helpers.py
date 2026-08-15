from __future__ import annotations

from typing import Any

def _response_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        choices = response.get("choices")
        if not choices:
            raise ValueError("No choices in response")
        msg = choices[0].get("message", {})
        content = msg.get("content")
        if content is None:
            content = msg.get("reasoning_content") or ""
        return str(content)
    if hasattr(response, "choices"):
        if not response.choices:
            raise ValueError("No choices in response")
        msg = response.choices[0].message
        content = getattr(msg, "content", None)
        if content is None:
            content = getattr(msg, "reasoning_content", None) or ""
        return str(content)
    return ""


def _executor_model(
    pseudo_model: str, cfg=None, task_value=0.5, compactor_summary="", subtask_count=0
) -> str:
    try:
        from autoconduck import pricing
        from autoconduck.config import get_config
        from autoconduck.config import resolve_orchestrator_model
        from autoconduck.routing.evaluator import complexity_of

        cfg = cfg or get_config()
        lo, hi = cfg.selection.phase_bands["executor"]
        raw = (
            0.5 * task_value
            + 0.3 * complexity_of(compactor_summary, cfg)
            + 0.2 * min(1, subtask_count / 6)
        )
        return pricing.select_closest(
            pricing.pool_ids(cfg),
            lo + (hi - lo) * max(0, min(1, raw)),
            cfg,
            pseudo_model=pseudo_model,
            band=(lo, hi),
        ) or resolve_orchestrator_model(cfg)
    except Exception:
        pass
    from autoconduck.config import select_model_by_tier

    return select_model_by_tier("expensive", cfg)

