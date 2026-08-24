from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path

import re
from dataclasses import dataclass, field

from .helpers import _response_text
from .tools import TOOL_SCHEMAS, execute_tool, is_read_only_tool, tool_model

STAGNATION_MARKER = "<loop-stagnation:true>"


@dataclass
class LoopState:
    call_signatures: list[str] = field(default_factory=list)
    error_streak: int = 0
    distinct_files_touched: set[str] = field(default_factory=set)
    total_calls: int = 0
    mitigation_rung: int = 0
    error_calls: int = 0


def calculate_stagnation(state: LoopState) -> float:
    """Return a weighted stagnation score in the inclusive range [0, 1]."""
    repetition = 1.0 - len(set(state.call_signatures)) / len(state.call_signatures) if state.call_signatures else 0.0
    errors = min(1.0, state.error_calls / max(1, state.total_calls))
    errors = max(errors, min(1.0, state.error_streak / 3.0))
    progress = min(1.0, len(state.distinct_files_touched) / max(1, state.total_calls))
    return min(1.0, 0.4 * repetition + 0.35 * errors + 0.25 * (1.0 - progress))


def extract_text_tool_calls(text: str) -> list[dict]:
    """Extract tool calls emitted in plaintext by open-source / non-native models."""
    calls = []
    if not text or not isinstance(text, str):
        return calls

    # 1. Format: <tool_call:opensource>name<tool_sep:opensource><arg_key...>...
    opensource_pattern = re.compile(
        r"<tool_call:opensource>\s*([a-zA-Z0-9_\-\.]+)\s*<tool_sep:opensource>(.*?)(?:</tool_call:opensource>|(?=<tool_call:opensource>)|$)",
        re.DOTALL,
    )
    arg_pattern = re.compile(
        r"<arg_key:opensource>\s*([^<]+)\s*</arg_key:opensource>\s*<arg_value:opensource>\s*([^<]*)\s*(?:</arg_value:opensource>|$)",
        re.DOTALL,
    )

    for idx, match in enumerate(opensource_pattern.finditer(text)):
        func_name = match.group(1).strip()
        raw_args = match.group(2)
        args_dict = {}
        for arg_m in arg_pattern.finditer(raw_args):
            k = arg_m.group(1).strip()
            v = arg_m.group(2).strip()
            try:
                args_dict[k] = json.loads(v)
            except Exception:
                args_dict[k] = v
        calls.append(
            {
                "id": f"call_text_{idx}",
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": json.dumps(args_dict),
                },
            }
        )

    if calls:
        return calls

    # 2. Format: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    generic_tag_pattern = re.compile(
        r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL
    )
    for idx, match in enumerate(generic_tag_pattern.finditer(text)):
        try:
            data = json.loads(match.group(1))
            name = data.get("name") or data.get("function")
            args = data.get("arguments") or data.get("parameters") or {}
            if name:
                calls.append(
                    {
                        "id": f"call_text_{idx}",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": (
                                json.dumps(args) if isinstance(args, dict) else str(args)
                            ),
                        },
                    }
                )
        except Exception:
            pass

    return calls


def strip_tool_call_tags(text: str) -> str:
    """Clean raw tool invocation tags from assistant text before client presentation."""
    if not text or not isinstance(text, str):
        return text or ""
    cleaned = re.sub(r"<tool_calls:opensource>.*", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<tool_call:opensource>.*", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"<tool_call>.*?</tool_call>", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


async def run_executor_tool_loop(
    client,
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    allowed_scope: list[str],
    workspace_root: Path,
    cfg,
    max_rounds: int = 10,
    time_budget_s: float = 180.0,
    tool_retry_cap: int = 3,
) -> str:
    """Run tools with fail-open provider compatibility.

    The dispatch sets ``params["drop_params"] = True`` (as ``_call`` does), so
    providers without function-calling support silently strip ``tools=`` and
    the loop exits at round 1 with plain text. This is intended, not a bug.
    """

    async def _dispatch(msgs, *, with_tools: bool, dispatch_model: str, schemas=TOOL_SCHEMAS):
        from autoconduck.server.messages_api import litellm_params_for

        params = litellm_params_for(dispatch_model, cfg)
        params["_path"] = "orchestrator-executor"
        params["_pseudo"] = "autoconduck"
        params["drop_params"] = True
        if with_tools:
            params["tools"] = schemas
            params["tool_choice"] = "auto"
        if client is not None and hasattr(client, "completion"):
            return await asyncio.to_thread(client.completion, messages=msgs, **params)
        if client is not None and hasattr(client, "chat"):
            return await asyncio.to_thread(
                client.chat.completions.create, messages=msgs, **params
            )
        import litellm

        return await litellm.acompletion(messages=msgs, **params)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    started = time.monotonic()
    failures: dict[tuple[str, str], int] = {}
    state = LoopState()
    dispatch_model = model
    force_fast_round = False
    read_only_round = False
    for _ in range(max_rounds):
        if time.monotonic() - started > time_budget_s:
            break
        schemas = [
            schema for schema in TOOL_SCHEMAS
            if is_read_only_tool(schema.get("function", {}).get("name", ""))
        ] if read_only_round else TOOL_SCHEMAS
        raw = await _dispatch(messages, with_tools=True, dispatch_model=dispatch_model, schemas=schemas)
        dispatch_model = model
        read_only_round = False
        choice = (
            raw["choices"][0]["message"]
            if isinstance(raw, dict)
            else raw.choices[0].message
        )
        tool_calls = (
            choice.get("tool_calls")
            if isinstance(choice, dict)
            else getattr(choice, "tool_calls", None)
        )
        content = (
            choice.get("content")
            if isinstance(choice, dict)
            else getattr(choice, "content", None)
        ) or ""

        if not tool_calls:
            text_calls = extract_text_tool_calls(content)
            if text_calls:
                tool_calls = text_calls

        if not tool_calls:
            return strip_tool_call_tags(_response_text(raw))

        cleaned_content = strip_tool_call_tags(content) or None
        if isinstance(choice, dict):
            assistant = dict(choice)
            assistant.setdefault("role", "assistant")
            assistant["content"] = cleaned_content
            assistant["tool_calls"] = tool_calls
        else:
            assistant = {
                "role": "assistant",
                "content": cleaned_content,
                "tool_calls": tool_calls,
            }
        messages.append(assistant)
        next_models = []
        for call in tool_calls:
            name = (
                call.get("function", {}).get("name")
                if isinstance(call, dict)
                else call.function.name
            )
            arguments = (
                call.get("function", {}).get("arguments", "{}")
                if isinstance(call, dict)
                else call.function.arguments
            )
            tc_id = call.get("id") if isinstance(call, dict) else call.id
            if isinstance(arguments, str):
                try:
                    parsed_args = json.loads(arguments)
                except Exception:
                    parsed_args = {}
            else:
                parsed_args = arguments if isinstance(arguments, dict) else {}
            signature = f"{name}:{json.dumps(parsed_args, sort_keys=True, default=str)}"
            state.call_signatures.append(hashlib.sha256(signature.encode()).hexdigest())
            state.total_calls += 1
            path = parsed_args.get("path") or parsed_args.get("file")
            if isinstance(path, str):
                state.distinct_files_touched.add(path)
            result = execute_tool(
                name,
                parsed_args,
                workspace_root=workspace_root,
                allowed_scope=allowed_scope,
                cfg=cfg,
            )
            if result.startswith("ERROR:"):
                state.error_streak += 1
                state.error_calls += 1
                key = (name, json.dumps(parsed_args, sort_keys=True, default=str))
                failures[key] = failures.get(key, 0) + 1
                stagnation = calculate_stagnation(state)
                if stagnation >= 0.55 and state.mitigation_rung < 1:
                    state.mitigation_rung = 1
                    messages.append({"role": "system", "content": f"{STAGNATION_MARKER} Your last few calls to [{name}] failed with similar errors. Change your approach or verify the schema."})
                if stagnation >= 0.70 and state.mitigation_rung < 2:
                    state.mitigation_rung = 2
                    force_fast_round = True
                if stagnation >= 0.82 and state.mitigation_rung < 3:
                    state.mitigation_rung = 3
                    read_only_round = True
                if stagnation >= 0.95 and state.mitigation_rung < 4:
                    state.mitigation_rung = 4
                    break
                if failures[key] >= max(1, tool_retry_cap):
                    terminal = (
                        f"{result}\nTool retry limit exceeded for {name} "
                        f"with these arguments ({tool_retry_cap} attempts)."
                    )
                    messages.append({"role": "tool", "tool_call_id": tc_id, "content": terminal})
                    return terminal
            else:
                state.error_streak = 0
            next_models.append(tool_model(name, model, cfg))
            messages.append({"role": "tool", "tool_call_id": tc_id, "content": result})
        # A read-only continuation is safe to handle with the cheapest FAST
        # model; any mutation keeps the model selected for the executor.
        if force_fast_round:
            dispatch_model = tool_model("read", model, cfg)
            force_fast_round = False
    return strip_tool_call_tags(
        _response_text(await _dispatch(messages, with_tools=False, dispatch_model=dispatch_model))
    )
