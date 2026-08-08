"""Read-only analyst fan-out primitives."""

from typing import Any

from .planner import SubTask


def build_subagent_prompt(task: SubTask, upstream_summaries: str = "") -> str:
    parts = [
        "ROLE: You are a read-only file analyst. You do not propose fixes or write code.",
        f"TASK: {task.goal}",
        f"FILES IN SCOPE (only these): {', '.join(task.scope)}",
        f"REQUIRED OUTPUT FORMAT: {task.output_contract}",
        f"DO NOT: {', '.join(task.constraints)}",
        f"CONTEXT FROM SIBLING TASKS: {upstream_summaries}",
    ]
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


def run_subagent(task: SubTask, upstream_summaries: str, client=None, cfg=None) -> str:
    try:
        from autoconduck.config import resolve_orchestrator_model
        from autoconduck.config import orchestrator_litellm_params
        model = resolve_orchestrator_model(cfg)
        params = orchestrator_litellm_params(cfg)
        messages = [{"role": "user", "content": build_subagent_prompt(task, upstream_summaries)}]
        if client is not None and hasattr(client, "completion"):
            return _text(client.completion(messages=messages, **params))
        if client is not None and hasattr(client, "chat"):
            return _text(client.chat.completions.create(messages=messages, **params))
        import litellm
        return _text(litellm.completion(messages=messages, **params))
    except Exception as exc:
        return f"Subagent error: {exc}"
