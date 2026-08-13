"""Read-only analyst fan-out primitives."""

import logging
from typing import Any

from .planner import OutputContract, SubTask
import math
from autoconduck.routing.evaluator import complexity_of


def subagent_target(subtask_prompt, role, plan_breadth, budget_hint, config):
    weight = {"read": 0.3, "analysis": 0.6, "write": 0.9}.get(role, 0.9)
    hint = (
        budget_hint
        if isinstance(budget_hint, (int, float)) and 0 <= budget_hint <= 1
        else 0.5
    )
    raw = (
        0.4 * complexity_of(subtask_prompt, config)
        + 0.3 * hint * weight
        + 0.3 / math.sqrt(max(1, plan_breadth))
    )
    lo, hi = config.selection.phase_bands["subagent"]
    return lo + (hi - lo) * max(0, min(1, raw))


def build_subagent_prompt(task: SubTask, upstream_summaries: str = "") -> str:
    role_header = (
        "ROLE: You are a read-only file analyst. You do not propose fixes or write code."
        if task.role != "write"
        else "ROLE: You are a file change drafting analyst."
    )
    parts = [
        role_header,
        f"TASK: {task.goal}",
        f"FILES IN SCOPE (only these): {', '.join(task.scope)}",
        f"REQUIRED OUTPUT FORMAT: {task.output_contract}",
        f"DO NOT: {', '.join(task.constraints)}",
        f"CONTEXT FROM SIBLING TASKS: {upstream_summaries}",
    ]
    if task.role == "write":
        parts.append("STRICT OUTPUT DIRECTIVE: Output only concise line-level diffs and necessary code snippets. Do not write conversational introductions or explanations.")
    if task.verified_context:
        bullets = "\n".join(f"- {item}" for item in task.verified_context)
        parts.append(f"VERIFIED CONTEXT (do not re-investigate):\n{bullets}")
    parts.append(
        f"TOOL BUDGET: You may make at most {task.read_budget} additional file reads/tool calls "
        f"beyond what's given above. Work with what you have first."
    )
    verify = getattr(task.output_contract, "verify", None) or []
    if verify:
        parts.append(f"VERIFY BEFORE RETURNING: {', '.join(verify)}")
    return "\n".join(parts)


def _text(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        return str(response["choices"][0]["message"]["content"])
    return str(response.choices[0].message.content)


async def run_subagent(
    task: SubTask,
    upstream_summaries: str,
    client=None,
    cfg=None,
    *,
    plan_breadth: int = 1,
    budget_hint: float | None = None,
) -> str:
    try:
        import asyncio
        from typing import Any
        from autoconduck.config import get_config
        from autoconduck import pricing
        from autoconduck.messages_api import litellm_params_for

        cfg = cfg or get_config()
        prompt = build_subagent_prompt(task, upstream_summaries)
        target = subagent_target(
            prompt, getattr(task, "role", "read"), plan_breadth, budget_hint, cfg
        )
        max_cost = (
            float(getattr(cfg.selection, "max_file_read_scaled_cost", 0.55))
            if getattr(task, "role", "read") != "write"
            else None
        )
        from autoconduck.config import resolve_orchestrator_model

        target_model = pricing.select_closest(
            pricing.pool_ids(cfg),
            target,
            cfg,
            band=cfg.selection.phase_bands["subagent"],
            max_scaled_cost=max_cost,
        ) or resolve_orchestrator_model(cfg)
        params: Any = litellm_params_for(target_model, cfg)
        params["_path"] = "orchestrator-subagent"
        params["_pseudo"] = "autoconduck"
        params["drop_params"] = True
        params.setdefault("max_tokens", 650)
        params.setdefault("timeout", 12.0)
        logging.getLogger("autoconduck.orchestrator").debug(
            "SUBAGENT PROMPT [%s]:\n%s", task.id, prompt
        )
        messages = [{"role": "user", "content": prompt}]

        async def _execute():
            if client is not None and hasattr(client, "completion"):
                return _text(
                    await asyncio.to_thread(client.completion, messages=messages, **params)
                )
            if client is not None and hasattr(client, "chat"):
                return _text(
                    await asyncio.to_thread(
                        client.chat.completions.create, messages=messages, **params
                    )
                )
            import litellm

            return _text(await litellm.acompletion(messages=messages, **params))

        try:
            return await asyncio.wait_for(_execute(), timeout=12.0)
        except asyncio.TimeoutError:
            return f"Subagent [{task.id}] timed out after 12s; proceeding with available context."
    except Exception as exc:
        return f"Subagent error: {exc}"
