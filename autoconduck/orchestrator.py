from __future__ import annotations

import asyncio
import glob as glob_mod
import json
import os
import re
import time
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

from autoconduck.config import OrchestrationSettings
from autoconduck.playbook import AcceptanceCheck, Playbook

MAX_WORKERS = 4
WORKER_TIMEOUT_S = 30
COMPACTION_TOKEN_LIMIT = 1000
PLAN_RETRIES = 1

class SubTask(BaseModel):
    id: str
    goal: str
    files: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    output_contract: str = ""
    context: str = ""
    constraints: list[str] = Field(default_factory=list)
    acceptance: list[AcceptanceCheck] = Field(default_factory=list)
    max_rounds: int | None = None
    tools: bool | None = None


class TaskPlan(BaseModel):
    tasks: list[SubTask]
    global_context: str = ""
    summary: str = ""

    @field_validator("tasks")
    @classmethod
    def _check_len(cls, v: list[SubTask]) -> list[SubTask]:
        if len(v) < 2 or len(v) > 6:
            raise ValueError("tasks must have 2-6 items")
        return v


class OrchestratorResult(BaseModel):
    compacted_context: str | None = None
    plan: TaskPlan | None = None
    degraded_to_fast: bool = False
    reason: str = ""
    worker_ok: int = 0
    worker_fail: int = 0
    plan_model_id: str | None = None
    worker_model_ids: list[str] = Field(default_factory=list)
    # adaptive fields
    rounds_used: int = 0
    round_history: str = ""
    accepted_tasks: list[str] = Field(default_factory=list)
    partial_tasks: list[str] = Field(default_factory=list)


PLANNER_SYSTEM = """You are a task decomposer. Break the user request into 2-6 parallel subtasks.
Each subtask must have isolated file context and an output_contract (what it must return).
Also emit per-task: context (repo evidence or ""), constraints (list, e.g. "do not modify tests/", "verify file exists before citing"), acceptance (2-4 checks per task using kinds file_exists/contains/not_contains/regex where sensible — use file_exists for expected output files, contains/regex for content assertions, llm only when nothing file-based fits), and TaskPlan.summary (one line).
Return JSON matching TaskPlan schema: {"tasks":[{"id":"t1","goal":"...","files":[],"depends_on":[],"output_contract":"...","context":"","constraints":[],"acceptance":[{"kind":"file_exists","path":"...","desc":"..."}],"max_rounds":null,"tools":null}],"global_context":"","summary":"one-line summary"}
Prefer file-disjoint subtasks. IDs must be unique like t1,t2.
"""

REPAIR_SYSTEM = """Your previous JSON was invalid. Return ONLY valid JSON matching {"tasks":[{"id":"t1","goal":"...","files":[],"depends_on":[],"output_contract":"...","context":"","constraints":[],"acceptance":[{"kind":"file_exists","path":"..."}],"max_rounds":null,"tools":null}],"global_context":"","summary":""} with 2-6 tasks."""


def _compact_texts(texts: list[str], limit_tokens: int = COMPACTION_TOKEN_LIMIT) -> str:
    # deterministic template compaction; approx 4 chars per token
    max_chars = limit_tokens * 4
    header = "[AutoConduck subagent findings]\n"
    bullets = []
    for i, t in enumerate(texts, 1):
        snippet = t.strip().replace("\n", " ")[:800]
        bullets.append(f"- Task {i}: {snippet}")
    body = "\n".join(bullets)
    full = header + body
    if len(full) > max_chars:
        full = full[: max_chars - 12] + "\n[truncated]"
    return full


def _build_compacted_context(
    plan: TaskPlan,
    outputs_by_id: dict[str, str],
    status_by_id: dict[str, str],
    rounds_by_id: dict[str, int],
    limit_tokens: int = COMPACTION_TOKEN_LIMIT,
) -> str:
    max_chars = limit_tokens * 4
    header = "[AutoConduck subagent findings]\n"
    # summary line if present
    lines: list[str] = []
    if plan.summary:
        lines.append(f"Summary: {plan.summary}")
    # accepted first, then partial, in plan order
    accepted_ids = [t.id for t in plan.tasks if status_by_id.get(t.id) == "PASS"]
    partial_ids = [t.id for t in plan.tasks if status_by_id.get(t.id) != "PASS"]
    ordered_ids = accepted_ids + partial_ids
    # map id->task for ordering within groups by original plan order
    order_index = {t.id: i for i, t in enumerate(plan.tasks)}
    ordered_ids_sorted_within_group = sorted(accepted_ids, key=lambda x: order_index.get(x, 999)) + sorted(
        partial_ids, key=lambda x: order_index.get(x, 999)
    )
    # actually use plan order with accepted first
    # Build per-task lines in that order
    for tid in ordered_ids_sorted_within_group:
        out = outputs_by_id.get(tid, "")
        snippet = out.strip().replace("\n", " ")[:600]
        rnd = rounds_by_id.get(tid, 1)
        status = status_by_id.get(tid, "PARTIAL")
        lines.append(f"- [{tid}] (round {rnd}, {status}): {snippet}")
    if partial_ids:
        lines.append(f"NOTE: tasks {', '.join(sorted(partial_ids))} did not fully pass verification — execution model should re-verify.")
    body = "\n".join(lines)
    full = header + body
    if len(full) > max_chars:
        full = full[: max_chars - 12] + "\n[truncated]"
    return full


def _resolve_sandbox(path: str) -> tuple[str | None, str]:
    """Resolve path relative to cwd and check sandbox. Returns (resolved_path or None, error_or_resolved)."""
    try:
        cwd = os.path.realpath(os.getcwd())
        # handle absolute or relative
        if os.path.isabs(path):
            # treat absolute as relative to cwd if it escapes? For sandbox, resolve realpath
            resolved = os.path.realpath(path)
        else:
            resolved = os.path.realpath(os.path.join(cwd, path))
        # sandbox check
        # allow exactly cwd and children
        if resolved == cwd or resolved.startswith(cwd + os.sep):
            return resolved, resolved
        else:
            return None, "access denied"
    except Exception as e:
        return None, f"error: {e}"


async def _execute_tool(name: str, args: dict[str, Any]) -> str:
    try:
        if name == "read_file":
            path = str(args.get("path", ""))
            offset = int(args.get("offset", 0) or 0)
            limit = int(args.get("limit", 2000) or 2000)
            resolved, msg = _resolve_sandbox(path)
            if resolved is None:
                return "access denied"
            if not os.path.exists(resolved):
                return f"file not found: {path}"
            if os.path.isdir(resolved):
                return f"is a directory: {path}"
            try:
                with open(resolved, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
                # offset/limit are lines? spec says offset/limit but stub says limit 2000 chars; we support lines
                # Interpret offset as line offset, limit as number of lines, else char limit
                # Use line-based slicing
                sliced = lines[offset : offset + limit]
                content = "".join(sliced)
                # cap 4000 chars
                if len(content) > 4000:
                    content = content[:4000]
                return content if content else "(empty file)"
            except Exception as e:
                return f"read error: {e}"
        elif name == "grep":
            pattern = str(args.get("pattern", ""))
            search_path = str(args.get("path", ".") or ".")
            include = args.get("include")
            max_matches = int(args.get("max_matches", 20) or 20)
            cwd = os.path.realpath(os.getcwd())
            # resolve search_path sandbox
            if search_path != ".":
                resolved, msg = _resolve_sandbox(search_path)
                if resolved is None:
                    return "access denied"
                base = resolved
            else:
                base = cwd
            if not os.path.exists(base):
                return f"path not found: {search_path}"
            try:
                regex = re.compile(pattern)
            except re.error as e:
                return f"invalid regex: {e}"
            matches: list[str] = []
            # if base is file, search that file
            if os.path.isfile(base):
                candidates = [base]
                walk = False
            else:
                walk = True
                candidates = []
            if walk:
                for root, dirs, files in os.walk(base):
                    # sandbox: ensure root still under cwd
                    rr = os.path.realpath(root)
                    if not (rr == cwd or rr.startswith(cwd + os.sep)):
                        continue
                    for fn in files:
                        fp = os.path.join(root, fn)
                        # include filter
                        if include:
                            import fnmatch

                            if not fnmatch.fnmatch(fn, str(include)):
                                continue
                        candidates.append(fp)
                        if len(candidates) > 5000:
                            break
            else:
                pass
            import fnmatch

            for fp in candidates:
                if len(matches) >= max_matches:
                    break
                # include re-check for file case
                if walk and include:
                    if not fnmatch.fnmatch(os.path.basename(fp), str(include)):
                        continue
                try:
                    with open(fp, "r", encoding="utf-8", errors="replace") as f:
                        text = f.read()
                except Exception:
                    continue
                for i, line in enumerate(text.splitlines(), 1):
                    if regex.search(line):
                        rel = os.path.relpath(fp, cwd) if fp.startswith(cwd) else fp
                        matches.append(f"{rel}:{i}:{line.strip()[:200]}")
                        if len(matches) >= max_matches:
                            break
            if not matches:
                return "no matches"
            result = "\n".join(matches[:max_matches])
            if len(result) > 4000:
                result = result[:4000]
            return result
        elif name == "glob":
            pattern = str(args.get("pattern", ""))
            # Use glob relative to cwd, cap 100
            # glob_mod.glob with recursive
            cwd = os.getcwd()
            # pattern may be absolute; sandbox not strictly needed for glob but filter
            results = glob_mod.glob(pattern, recursive=True)
            # also try with cwd prefix if pattern is relative and no results?
            if not results:
                results = glob_mod.glob(os.path.join(cwd, pattern), recursive=True)
                # relativize
                results = [os.path.relpath(p, cwd) if os.path.isabs(p) else p for p in results]
            # sandbox filter: only keep under cwd
            filtered: list[str] = []
            cwd_real = os.path.realpath(cwd)
            for p in results[:100]:
                rp = os.path.realpath(p) if os.path.isabs(p) else os.path.realpath(os.path.join(cwd, p))
                if rp == cwd_real or rp.startswith(cwd_real + os.sep):
                    filtered.append(p)
                else:
                    # still allow if pattern was explicit outside? but spec says sandbox deny, skip
                    continue
            out = "\n".join(filtered[:100])
            if len(out) > 4000:
                out = out[:4000]
            return out if out else "no matches"
        elif name == "list_dir":
            path = str(args.get("path", ".") or ".")
            resolved, msg = _resolve_sandbox(path)
            if resolved is None:
                return "access denied"
            if not os.path.exists(resolved):
                return f"path not found: {path}"
            if not os.path.isdir(resolved):
                return f"not a directory: {path}"
            try:
                entries = os.listdir(resolved)
                entries = sorted(entries)[:200]
                out = "\n".join(entries)
                if len(out) > 4000:
                    out = out[:4000]
                return out if out else "(empty)"
            except Exception as e:
                return f"list error: {e}"
        else:
            return "unknown tool"
    except Exception as e:
        return f"tool error: {e}"


class Orchestrator:
    def __init__(self, litellm_caller: Any | None = None):
        # litellm_caller: async callable (model, messages, ...) -> response
        self._call = litellm_caller

    async def _call_llm(self, model: str, messages: list[dict], **kwargs) -> str:
        if self._call is not None:
            resp = await self._call(model=model, messages=messages, **kwargs)
            # resp may be string or litellm response object
            if isinstance(resp, str):
                return resp
            try:
                # litellm response
                choice = resp.choices[0]  # type: ignore
                content = getattr(choice.message, "content", None) or choice.get("message", {}).get("content", "")  # type: ignore
                return str(content or "")
            except Exception:
                return str(resp)
        # fallback: try litellm directly
        try:
            import litellm  # type: ignore

            resp = await litellm.acompletion(model=model, messages=messages, **kwargs)  # type: ignore
            choice = resp.choices[0]  # type: ignore
            return str(getattr(choice.message, "content", "") or "")
        except Exception as e:
            raise RuntimeError(f"llm call failed: {e}") from e

    async def _call_llm_with_tools(
        self,
        messages: list[dict],
        model_id: str,
        tools: list[dict],
        max_tool_calls: int = 6,
        timeout: int = WORKER_TIMEOUT_S,
    ) -> str:
        # mirror injectability: delegate to _call_llm so mocks on _call_llm also work
        cur_messages: list[dict] = list(messages)

        async def _one_call(msgs: list[dict]) -> Any:
            try:
                return await self._call_llm(model_id, msgs, tools=tools, tool_choice="auto", temperature=0.3, max_tokens=1200)
            except TypeError as e:
                raise RuntimeError("tool path requires callable accepting tools= kwarg") from e

        # single wait_for around loop body is fine; spec says cumulative timeout
        async def _loop() -> str:
            for _ in range(max_tool_calls + 1):
                resp = await _one_call(cur_messages)
                # string shortcut
                if isinstance(resp, str):
                    # if string response, no tool calls
                    return resp
                # try to extract content and tool_calls
                try:
                    choice = resp.choices[0]  # type: ignore
                    msg = choice.message  # type: ignore
                    # content may be in message.content
                    content = getattr(msg, "content", None)
                    tool_calls = getattr(msg, "tool_calls", None)
                    # alternative dict access
                    if tool_calls is None and isinstance(msg, dict):
                        tool_calls = msg.get("tool_calls")
                        content = msg.get("content", content)
                    if not tool_calls:
                        # no tools -> return content
                        if content is None:
                            # try choice.get
                            try:
                                content = choice.get("message", {}).get("content", "")  # type: ignore
                            except Exception:
                                content = ""
                        return str(content or "")
                    # has tool_calls
                    # append assistant message with tool_calls to history
                    # need to preserve tool_calls structure for next turn
                    # Build assistant message dict
                    assistant_msg: dict[str, Any] = {"role": "assistant", "content": content or ""}
                    # normalize tool_calls to list of dicts with id/function
                    norm_calls = []
                    for tc in tool_calls:
                        if isinstance(tc, dict):
                            tid = tc.get("id", "")
                            fn = tc.get("function", {})
                            name = fn.get("name", "")
                            args = fn.get("arguments", "{}")
                            norm_calls.append({"id": tid, "function": {"name": name, "arguments": args}})
                        else:
                            tid = getattr(tc, "id", "")
                            fn = getattr(tc, "function", None)
                            if fn is not None:
                                name = getattr(fn, "name", "")
                                args = getattr(fn, "arguments", "{}")
                            else:
                                name = ""
                                args = "{}"
                            norm_calls.append({"id": tid, "function": {"name": name, "arguments": args}})
                    assistant_msg["tool_calls"] = [
                        {"id": c["id"], "type": "function", "function": c["function"]} for c in norm_calls
                    ]
                    cur_messages.append(assistant_msg)
                    # execute each tool
                    for c in norm_calls:
                        name = c["function"]["name"]
                        arg_str = c["function"]["arguments"]
                        try:
                            args = json.loads(arg_str) if isinstance(arg_str, str) else dict(arg_str)
                        except Exception:
                            args = {}
                        result = await _execute_tool(name, args)
                        # truncate to 4000 already done
                        cur_messages.append({"role": "tool", "tool_call_id": c["id"], "content": str(result)[:4000]})
                    # continue loop
                    continue
                except Exception as e:
                    # if parsing failed, return stringified resp
                    return str(resp)
            # after loop, return last content if any
            # Find last assistant content
            for m in reversed(cur_messages):
                if m.get("role") == "assistant" and m.get("content"):
                    return str(m["content"])
            return ""

        return await _loop()

    async def plan_and_execute(
        self,
        request: Any,
        *,
        plan_model_id: str,
        worker_model_id: str,
        worker_model_ladder: list[str] | None = None,
        max_workers: int | None = None,
        orch_settings: Any | None = None,
        verifier_model_id: str | None = None,
        exploration_model_id: str | None = None,
    ) -> OrchestratorResult:
        # settings resolution
        settings = orch_settings or OrchestrationSettings()
        worker_ladder = worker_model_ladder or [worker_model_id]
        # sanitize ladder: filter empty
        worker_ladder = [m for m in worker_ladder if m]
        if not worker_ladder:
            worker_ladder = [worker_model_id]
        effective_max_rounds = getattr(settings, "max_rounds", 2)
        try:
            effective_max_rounds = int(effective_max_rounds)
        except Exception:
            effective_max_rounds = 2
        effective_max_rounds = max(1, min(4, effective_max_rounds))
        effective_tools_on = bool(getattr(settings, "worker_tools", True))

        messages = getattr(request, "messages", None)
        if messages is None and isinstance(request, dict):
            messages = request.get("messages", [])
        # normalize messages to dicts for LLM
        norm_msgs: list[dict] = []
        for m in messages or []:
            if isinstance(m, dict):
                norm_msgs.append(m)
            else:
                norm_msgs.append({"role": getattr(m, "role", "user"), "content": str(getattr(m, "content", ""))})

        # last user message for planning context
        last_user = ""
        for m in reversed(norm_msgs):
            if m.get("role") == "user":
                last_user = str(m.get("content", ""))
                break
        if not last_user and norm_msgs:
            last_user = str(norm_msgs[-1].get("content", ""))

        # STAGE 0 — EXPLORATION
        exploration_context = ""
        if bool(getattr(settings, "exploration", False)):
            # only if no context already? We treat as always try unless last_user empty?
            # spec: only if settings.exploration AND no context already — we have no prior context, so do it
            try:
                exp_model = exploration_model_id or plan_model_id
                exp_prompt = (
                    f"You are a repo scout. Given this request: {last_user}. "
                    "Return a compact list of: (a) relevant files/paths you expect to be involved (guesses ok), "
                    "(b) any obvious risks or unknowns, (c) 2-4 concrete things subagents should verify in the repo. "
                    "Max 300 words. Do not modify anything."
                )
                exploration_context = await asyncio.wait_for(
                    self._call_llm(
                        model=exp_model,
                        messages=[{"role": "user", "content": exp_prompt}],
                        temperature=0.2,
                        max_tokens=500,
                    ),
                    timeout=30,
                )
                exploration_context = str(exploration_context or "")[:2000]
            except Exception:
                exploration_context = ""

        # ---- planning phase (STAGE 1) ----
        plan: TaskPlan | None = None
        plan_error = ""
        for attempt in range(2):  # initial + one retry
            system = PLANNER_SYSTEM if attempt == 0 else REPAIR_SYSTEM
            # include exploration_context in user prompt if present
            user_content = last_user
            if exploration_context:
                user_content = f"Request: {last_user}\n\nExploration context:\n{exploration_context}"
            try:
                content = await self._call_llm(
                    model=plan_model_id,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user_content}],
                    temperature=0.2,
                    max_tokens=1000,
                )
                # extract JSON
                json_str = _extract_json(content)
                data = json.loads(json_str)
                plan = TaskPlan.model_validate(data)
                break
            except Exception as e:
                plan_error = str(e)
                if attempt == 0:
                    continue
                else:
                    plan = None

        if plan is None:
            return OrchestratorResult(
                compacted_context=None,
                plan=None,
                degraded_to_fast=True,
                reason=f"plan_failed:{plan_error[:200]}",
                worker_ok=0,
                worker_fail=0,
                plan_model_id=plan_model_id,
                rounds_used=0,
                round_history="",
                accepted_tasks=[],
                partial_tasks=[],
            )

        # ---- STAGE 2 — ROUND LOOP ----
        # Setup playbook (honor custom playbook_path)
        try:
            pb_path = getattr(settings, "playbook_path", None)
            if pb_path is None:
                try:
                    from autoconduck.config import get_config

                    pb_path = get_config().playbook_path
                except Exception:
                    pb_path = None
            pb = Playbook.load(pb_path)
        except Exception:
            pb = Playbook.load(None)

        tools_defs = getattr(pb, "tools", [])

        # max_workers resolution: explicit arg → get_config().max_workers → MAX_WORKERS
        _mw = max_workers
        if _mw is None:
            try:
                from autoconduck.config import get_config

                _mw = int(get_config().max_workers)
            except Exception:
                _mw = MAX_WORKERS

        pending: list[SubTask] = list(plan.tasks)
        # track per-task state
        accepted: list[str] = []
        partial: list[str] = []
        outputs_by_id: dict[str, str] = {}
        status_by_id: dict[str, str] = {}
        rounds_by_id: dict[str, int] = {}
        prev_outputs: dict[str, str] = {}
        feedbacks: dict[str, str] = {}
        rounds_used_for: dict[str, int] = {}
        round_history_lines: list[str] = []
        total_rounds_used = 0

        # helpers for verification
        def _has_rule_checks(task: SubTask) -> bool:
            for ac in task.acceptance or []:
                kind = getattr(ac, "kind", None) or (ac.get("kind") if isinstance(ac, dict) else None)  # type: ignore
                if kind in ("file_exists", "contains", "not_contains", "regex", "command"):
                    return True
            return False

        def _any_llm_check(task: SubTask) -> bool:
            for ac in task.acceptance or []:
                kind = getattr(ac, "kind", None) or (ac.get("kind") if isinstance(ac, dict) else None)  # type: ignore
                if kind == "llm":
                    return True
            return False

        async def _verify_task(task: SubTask, output: str, settings_obj: Any, verifier_mid: str | None) -> tuple[bool, str]:
            acceptance = task.acceptance or []
            # normalize to AcceptanceCheck objects (if dicts)
            norm: list[Any] = []
            for ac in acceptance:
                if isinstance(ac, dict):
                    try:
                        # try to validate via AcceptanceCheck
                        from autoconduck.playbook import AcceptanceCheck as AC

                        norm.append(AC.model_validate(ac))
                    except Exception:
                        # keep dict
                        norm.append(ac)
                else:
                    norm.append(ac)
            # if no acceptance at all
            if not norm:
                _min_chars = getattr(settings_obj, "min_output_chars", 40)
                try:
                    _min_chars = int(_min_chars)
                except Exception:
                    _min_chars = 40
                if len(output.strip()) >= _min_chars:
                    return True, "no criteria, output length ok"
                else:
                    return False, "output too short"

            # rule-based checks first
            failures: list[str] = []
            for ac in norm:
                kind = getattr(ac, "kind", None) if not isinstance(ac, dict) else ac.get("kind")
                if kind == "llm":
                    continue  # handled later
                if kind == "command":
                    allow = bool(getattr(settings_obj, "allow_command_checks", False))
                    if not allow:
                        continue  # skip, note but not fail? spec: feedback notes skipped
                    # if allowed, we don't actually execute commands for safety; treat as skip with note?
                    # spec says ONLY if allow_command_checks else skip (feedback notes skipped)
                    # So no failure, just skip
                    continue
                # rule kinds
                if kind == "file_exists":
                    path = getattr(ac, "path", None) if not isinstance(ac, dict) else ac.get("path")
                    if not path:
                        failures.append(f"file_exists check missing path")
                        continue
                    resolved, msg = _resolve_sandbox(str(path))
                    if resolved is None:
                        failures.append(f"file_exists {path}: access denied")
                    else:
                        if not os.path.exists(resolved):
                            failures.append(f"file_exists {path}: not found")
                elif kind in ("contains", "not_contains", "regex"):
                    path = getattr(ac, "path", None) if not isinstance(ac, dict) else ac.get("path")
                    pattern = getattr(ac, "pattern", None) if not isinstance(ac, dict) else ac.get("pattern")
                    if not path or pattern is None:
                        failures.append(f"{kind} check missing path/pattern")
                        continue
                    resolved, msg = _resolve_sandbox(str(path))
                    if resolved is None:
                        failures.append(f"{kind} {path}: access denied")
                        continue
                    if not os.path.exists(resolved):
                        failures.append(f"{kind} {path}: file not found")
                        continue
                    try:
                        with open(resolved, "r", encoding="utf-8", errors="replace") as f:
                            text = f.read()
                    except Exception as e:
                        failures.append(f"{kind} {path}: read error {e}")
                        continue
                    if kind == "contains":
                        if pattern not in text:
                            failures.append(f"contains {path}: pattern '{pattern}' not found")
                    elif kind == "not_contains":
                        if pattern in text:
                            failures.append(f"not_contains {path}: pattern '{pattern}' should be absent")
                    elif kind == "regex":
                        try:
                            if not re.search(pattern, text):
                                failures.append(f"regex {path}: pattern '{pattern}' no match")
                        except re.error as e:
                            failures.append(f"regex {path}: invalid regex {e}")
                else:
                    # unknown kind, skip
                    continue
            if failures:
                return False, "; ".join(failures)[:300]

            # decide if LLM verdict needed
            verifier_mode = str(getattr(settings_obj, "verifier", "auto"))
            has_rule = _has_rule_checks(task)
            has_llm = _any_llm_check(task)
            need_llm = False
            if verifier_mode in ("llm", "auto"):
                if not has_rule:
                    need_llm = True
                elif has_llm:
                    need_llm = True
                elif verifier_mode == "llm" and norm:
                    need_llm = True
            if need_llm:
                # call LLM verdict
                v_model = verifier_mid or (worker_ladder[0] if worker_ladder else worker_model_id)
                # build criteria string
                try:
                    criteria_str = pb.render_acceptance(norm) if hasattr(pb, "render_acceptance") else str(norm)  # type: ignore
                except Exception:
                    criteria_str = str(norm)
                user_ver = f"Criteria:\n{criteria_str}\n\nOutput:\n{output[:3000]}"
                try:
                    verdict_raw = await asyncio.wait_for(
                        self._call_llm(
                            model=v_model,
                            messages=[
                                {"role": "system", "content": "You judge whether the subagent output satisfies the acceptance criteria. Reply exactly 'PASS' or 'FAIL' then a one-line reason."},
                                {"role": "user", "content": user_ver},
                            ],
                            temperature=0,
                            max_tokens=24,
                        ),
                        timeout=15,
                    )
                    verdict = str(verdict_raw).strip()
                    if verdict.upper().startswith("PASS"):
                        return True, verdict[:200]
                    elif verdict.upper().startswith("FAIL"):
                        reason = verdict[4:].strip(" :.-") or verdict
                        return False, reason[:300]
                    else:
                        # treat ambiguous as PASS with note
                        return True, f"verifier ambiguous: {verdict[:100]}"
                except Exception as e:
                    return True, "verifier unavailable"
            # if we reach here, rule checks passed and no LLM needed
            return True, "rule checks passed"

        # semaphore for concurrency
        sem = asyncio.Semaphore(_mw)

        # loop rounds
        # pending contains tasks not yet accepted/partial final
        # we need to track which tasks have been tried and still pending for retry
        # initial pending is all; after each round, we recompute pending as those that failed and have retries left
        next_pending: list[SubTask] = list(pending)
        # we will iterate round_no 1..effective_max_rounds but ready is computed per round
        for round_no in range(1, effective_max_rounds + 1):
            # soft deps: satisfied if in accepted ∪ partial
            satisfied = set(accepted) | set(partial)
            # also tasks accepted earlier in this loop that will be added after verification, not yet for this round's ready computation
            # For round 1, satisfied starts empty; for later rounds, accepted/partial from previous rounds
            ready = [t for t in next_pending if all(dep in satisfied for dep in (t.depends_on or []))]
            if not ready:
                # if no ready but still pending tasks with unsatisfied deps, break (cannot be satisfied)
                # also if next_pending empty, break
                if not next_pending:
                    break
                # No ready but pending remains means deps unsatisfied; break to avoid infinite
                break
            # prepare next_pending for next iteration: will be rebuilt from failures that have retries
            current_next_pending: list[SubTask] = []

            # fire workers
            async def run_one(task: SubTask) -> tuple[SubTask, str | Exception, str]:
                async with sem:
                    # determine round_idx for this task
                    prev_rounds = rounds_used_for.get(task.id, 0)
                    round_idx = prev_rounds + 1
                    # model escalation
                    model_for_round = worker_ladder[min(round_idx - 1, len(worker_ladder) - 1)]
                    # build messages via playbook
                    try:
                        brief_messages = pb.render_brief(task, round_idx, prev_output=prev_outputs.get(task.id), feedback=feedbacks.get(task.id))
                    except Exception:
                        # fallback
                        base = f"TASK {task.id}: {task.goal}\nOutput contract: {task.output_contract}"
                        if round_idx >= 2 and feedbacks.get(task.id):
                            base += f"\nFeedback: {feedbacks.get(task.id)}\nPrev: {prev_outputs.get(task.id,'')[:500]}"
                        brief_messages = [{"role": "user", "content": base}]
                    # decide tool usage
                    use_tools = effective_tools_on and (task.tools is not False)
                    if use_tools and tools_defs:
                        try:
                            c = await asyncio.wait_for(
                                self._call_llm_with_tools(brief_messages, model_for_round, tools_defs, max_tool_calls=6, timeout=WORKER_TIMEOUT_S),
                                timeout=WORKER_TIMEOUT_S,
                            )
                            return (task, str(c), model_for_round)
                        except Exception as e:
                            return (task, e, model_for_round)
                    else:
                        try:
                            c = await asyncio.wait_for(
                                self._call_llm(
                                    model=model_for_round,
                                    messages=brief_messages,
                                    temperature=0.3,
                                    max_tokens=1200,
                                ),
                                timeout=WORKER_TIMEOUT_S,
                            )
                            return (task, str(c), model_for_round)
                        except Exception as e:
                            return (task, e, model_for_round)

            # gather
            results = await asyncio.gather(*(run_one(t) for t in ready), return_exceptions=True)

            # process each result with verification
            # Need to handle gather return_exceptions: each result is (task,str|Exception,str) or Exception
            # Flatten
            new_ready_results: list[tuple[SubTask, str | Exception, str]] = []
            for r in results:
                if isinstance(r, Exception):
                    # gather exception (should not happen since run_one catches)
                    continue
                if isinstance(r, tuple) and len(r) == 3:
                    new_ready_results.append(r)  # type: ignore
                else:
                    # unexpected
                    continue

            # For tasks in ready but not in results (if exception), treat as failure
            ready_ids = {t.id for t in ready}
            result_ids = {t.id for t, _, _ in new_ready_results}
            # verification
            ok_count = 0
            fail_count = 0
            failed_worker_ids: list[str] = []

            for task, out, model_used in new_ready_results:
                rounds_used_for[task.id] = rounds_used_for.get(task.id, 0) + 1
                round_idx = rounds_used_for[task.id]
                total_rounds_used = max(total_rounds_used, round_idx, round_no)
                if isinstance(out, Exception):
                    # worker exception
                    fail_count += 1
                    feedback = str(out)[:300]
                    # decide retry or partial
                    task_rounds = task.max_rounds or effective_max_rounds
                    if round_idx < task_rounds and total_rounds_used < effective_max_rounds:
                        # keep pending for next round
                        prev_outputs[task.id] = ""
                        feedbacks[task.id] = feedback
                        current_next_pending.append(task)
                        # keep status not yet, but history?
                        # Not adding to partial yet
                    else:
                        partial.append(task.id)
                        status_by_id[task.id] = "PARTIAL"
                        rounds_by_id[task.id] = round_idx
                        outputs_by_id[task.id] = ""
                        round_history_lines.append(f"[{task.id}] PARTIAL (round {round_idx}): {feedback[:120]}")
                    continue
                output_str = str(out)
                # if output is Exception instance string? already handled
                # verify
                try:
                    passed, feedback = await _verify_task(task, output_str, settings, verifier_model_id)
                except Exception as e:
                    passed, feedback = False, f"verify error: {e}"
                outputs_by_id[task.id] = output_str
                rounds_by_id[task.id] = round_idx
                if passed:
                    accepted.append(task.id)
                    status_by_id[task.id] = "PASS"
                    ok_count += 1
                    round_history_lines.append(f"[{task.id}] PASS (round {round_idx})")
                else:
                    # failed verification
                    task_rounds = task.max_rounds or effective_max_rounds
                    if round_idx < task_rounds and total_rounds_used < effective_max_rounds and round_no < effective_max_rounds:
                        # retry next round
                        prev_outputs[task.id] = output_str
                        feedbacks[task.id] = feedback
                        current_next_pending.append(task)
                        # do not mark partial yet
                    else:
                        partial.append(task.id)
                        status_by_id[task.id] = "PARTIAL"
                        round_history_lines.append(f"[{task.id}] PARTIAL (round {round_idx}): {feedback[:120]}")
                        # keep failure count?
                        fail_count += 1

            # Also handle tasks that were not ready but pending for future deps: keep them
            # next_pending for next round = tasks that were pending but not ready (deps not satisfied) + current_next_pending (retries)
            not_ready_pending = [t for t in next_pending if t.id not in {x.id for x in ready}]
            next_pending = not_ready_pending + current_next_pending

            # update total_rounds_used to at least round_no if we executed
            total_rounds_used = max(total_rounds_used, round_no)

            if not next_pending:
                break  # early exit when everything accepted (or partial exhausted)
            # continue to next round

        # After loop, check if we had zero acceptances but still pending partial? That's fine per spec, continue to compaction
        # But need to handle case where some tasks never got executed due to deps -> mark partial?
        # Any remaining pending that never ran should be partial
        for t in next_pending:
            if t.id not in accepted and t.id not in partial:
                partial.append(t.id)
                status_by_id[t.id] = "PARTIAL"
                rounds_by_id[t.id] = rounds_used_for.get(t.id, effective_max_rounds)
                if t.id not in outputs_by_id:
                    outputs_by_id[t.id] = ""
                # add history if not already
                if not any(t.id in line for line in round_history_lines):
                    round_history_lines.append(f"[{t.id}] PARTIAL (round {rounds_by_id[t.id]}): not executed due to dependencies")

        # Determine worker_ok / worker_fail based on accepted/partial counts
        worker_ok = len(accepted)
        worker_fail = len(partial)  # also counts verification failures, not just exceptions
        # However degraded semantics: only keep degraded for plan failures / total worker exceptions, not partial
        # If ok==0 and no pending retries but partial exists, we still compact (not degrade) — unless all workers threw exceptions?
        # Check if we had any accepted; if none and outputs all empty due to exceptions, we need to degrade if worker exceptions total
        # Preserve today's semantics: if ok==0 after all rounds and all outputs failed due to exceptions, degrade
        # But if we have partial with outputs (verification failures), we don't degrade — we compact partial results
        # So only degrade if accepted empty and all partial outputs empty and we had exception path?
        # Simplify: if worker_ok==0 and worker_fail>0 and all outputs_by_id values are empty -> degrade all_workers_failed
        if worker_ok == 0 and worker_fail > 0:
            # check if we had any outputs with content (partial retries produced outputs)
            has_content = any(v.strip() for v in outputs_by_id.values())
            if not has_content:
                # check if failures were exceptions vs verification
                # if all round_history mentions exception? we treat as all_workers_failed
                # To preserve original semantics: if ok==0 we previously returned degraded_to_fast=True reason all_workers_failed
                # Now spec says All-failed guard: if round 1 completes with zero acceptances AND zero pending remaining (all partial) → continue to compaction (not hard degrade — partial results still compact; only keep degraded_to_fast for plan failures / total worker exceptions)
                # So we only degrade if workers threw exceptions entirely and outputs empty
                # We'll check if has_content False => assume all exceptions
                return OrchestratorResult(
                    compacted_context=None,
                    plan=plan,
                    degraded_to_fast=True,
                    reason="all_workers_failed",
                    worker_ok=0,
                    worker_fail=worker_fail,
                    plan_model_id=plan_model_id,
                    worker_model_ids=worker_ladder[:worker_fail] if worker_fail else [worker_ladder[0]],
                    rounds_used=total_rounds_used,
                    round_history="\n".join(round_history_lines)[:1500],
                    accepted_tasks=accepted,
                    partial_tasks=partial,
                )

        # Build compacted context
        # If no outputs_by_id but we have accepted, use outputs_by_id; else build from outputs
        compacted = _build_compacted_context(plan, outputs_by_id, status_by_id, rounds_by_id, COMPACTION_TOKEN_LIMIT)
        round_history = "\n".join(round_history_lines)[:1500]

        # worker_model_ids: spec says existing field; keep behavior: list of failed worker ids? Original collected failed_worker_ids
        # For new logic, we can return worker_ladder entries for partial/failed? Use ladder first element repeated?
        # Keep simple: return ladder for failures
        failed_ids: list[str] = []
        for pid in partial:
            # assign model that was used last for that task: worker_ladder[min(rounds_by_id[pid]-1, len(worker_ladder)-1)]
            mid = worker_ladder[min(rounds_by_id.get(pid, 1) - 1, len(worker_ladder) - 1)]
            failed_ids.append(mid)

        return OrchestratorResult(
            compacted_context=compacted,
            plan=plan,
            degraded_to_fast=False,
            reason="ok",
            worker_ok=worker_ok,
            worker_fail=worker_fail,
            plan_model_id=plan_model_id,
            worker_model_ids=failed_ids,
            rounds_used=total_rounds_used,
            round_history=round_history,
            accepted_tasks=accepted,
            partial_tasks=partial,
        )


def _extract_json(text: str) -> str:
    text = text.strip()
    # try direct json
    if text.startswith("{"):
        # find balanced json object
        # quick: try to find first { and last }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                json.loads(candidate)
                return candidate
            except Exception:
                pass
    # fallback: look for code fence
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m:
        return m.group(1)
    # last resort return as-is
    return text
