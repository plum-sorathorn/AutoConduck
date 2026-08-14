from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from .helpers import _response_text
from .tools import TOOL_SCHEMAS, execute_tool


async def run_executor_tool_loop(client, model: str, system_prompt: str, user_prompt: str, *, allowed_scope: list[str], workspace_root: Path, cfg, max_rounds: int = 10, time_budget_s: float = 180.0) -> str:
    """Run tools with fail-open provider compatibility.

    The dispatch sets ``params["drop_params"] = True`` (as ``_call`` does), so
    providers without function-calling support silently strip ``tools=`` and
    the loop exits at round 1 with plain text. This is intended, not a bug.
    """
    async def _dispatch(msgs, *, with_tools: bool):
        from autoconduck.messages_api import litellm_params_for
        params = litellm_params_for(model, cfg)
        params["_path"] = "orchestrator-executor"
        params["_pseudo"] = "autoconduck"
        params["drop_params"] = True
        if with_tools:
            params["tools"] = TOOL_SCHEMAS
            params["tool_choice"] = "auto"
        if client is not None and hasattr(client, "completion"):
            return await asyncio.to_thread(client.completion, messages=msgs, **params)
        if client is not None and hasattr(client, "chat"):
            return await asyncio.to_thread(client.chat.completions.create, messages=msgs, **params)
        import litellm
        return await litellm.acompletion(messages=msgs, **params)

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    started = time.monotonic()
    for _ in range(max_rounds):
        if time.monotonic() - started > time_budget_s:
            break
        raw = await _dispatch(messages, with_tools=True)
        choice = raw["choices"][0]["message"] if isinstance(raw, dict) else raw.choices[0].message
        tool_calls = choice.get("tool_calls") if isinstance(choice, dict) else getattr(choice, "tool_calls", None)
        if not tool_calls:
            return _response_text(raw)
        if isinstance(choice, dict):
            assistant = dict(choice)
            assistant.setdefault("role", "assistant")
        else:
            assistant = {"role": "assistant", "content": getattr(choice, "content", None), "tool_calls": tool_calls}
        messages.append(assistant)
        for call in tool_calls:
            name = call.get("function", {}).get("name") if isinstance(call, dict) else call.function.name
            arguments = call.get("function", {}).get("arguments", "{}") if isinstance(call, dict) else call.function.arguments
            tc_id = call.get("id") if isinstance(call, dict) else call.id
            result = execute_tool(name, json.loads(arguments), workspace_root=workspace_root, allowed_scope=allowed_scope, cfg=cfg)
            messages.append({"role": "tool", "tool_call_id": tc_id, "content": result})
    return _response_text(await _dispatch(messages, with_tools=False))
