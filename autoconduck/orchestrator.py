from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

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


class TaskPlan(BaseModel):
    tasks: list[SubTask]

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


PLANNER_SYSTEM = """You are a task decomposer. Break the user request into 2-6 parallel subtasks.
Each subtask must have isolated file context and an output_contract (what it must return).
Return JSON matching TaskPlan schema: {"tasks":[{"id":"t1","goal":"...","files":[],"depends_on":[],"output_contract":"..."}]}
Prefer file-disjoint subtasks. IDs must be unique like t1,t2.
"""

REPAIR_SYSTEM = """Your previous JSON was invalid. Return ONLY valid JSON matching {"tasks":[{"id":"t1","goal":"...","files":[],"depends_on":[],"output_contract":"..."}]} with 2-6 tasks."""


def _compact_texts(texts: list[str], limit_tokens: int = COMPACTION_TOKEN_LIMIT) -> str:
    # deterministic template compaction; approx 4 chars per token
    max_chars = limit_tokens * 4
    header = "AutoConduck subagent findings:\n"
    bullets = []
    for i, t in enumerate(texts, 1):
        snippet = t.strip().replace("\n", " ")[:800]
        bullets.append(f"- Task {i}: {snippet}")
    body = "\n".join(bullets)
    full = header + body
    if len(full) > max_chars:
        full = full[: max_chars - 12] + "\n[truncated]"
    return full


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

    async def plan_and_execute(
        self,
        request: Any,
        *,
        plan_model_id: str,
        worker_model_id: str,
        max_workers: int | None = None,
    ) -> OrchestratorResult:
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

        # ---- planning phase ----
        plan: TaskPlan | None = None
        plan_error = ""
        for attempt in range(2):  # initial + one retry
            system = PLANNER_SYSTEM if attempt == 0 else REPAIR_SYSTEM
            try:
                content = await self._call_llm(
                    model=plan_model_id,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": last_user}],
                    temperature=0.2,
                    max_tokens=800,
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
            )

        # ---- worker pool ----
        _mw = max_workers
        if _mw is None:
            try:
                from autoconduck.config import get_config

                _mw = int(get_config().max_workers)
            except Exception:
                _mw = MAX_WORKERS
        sem = asyncio.Semaphore(_mw)
        # sort by depends_on to respect DAG order partially; for simplicity run in order but still parallel within levels
        # Here we just run all with semaphore; dependencies are soft (prompt includes depends_on note)
        async def run_one(task: SubTask) -> str | Exception:
            async with sem:
                prompt = (
                    f"Subtask {task.id}: {task.goal}\n"
                    f"Files: {', '.join(task.files) if task.files else '(none)'}\n"
                    f"Depends on: {', '.join(task.depends_on) if task.depends_on else 'none'}\n"
                    f"Output contract: {task.output_contract}\n"
                    f"Return concise result for this subtask only."
                )
                try:
                    c = await asyncio.wait_for(
                        self._call_llm(
                            model=worker_model_id,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.3,
                            max_tokens=600,
                        ),
                        timeout=WORKER_TIMEOUT_S,
                    )
                    return str(c)
                except Exception as e:
                    return e

        results = await asyncio.gather(*(run_one(t) for t in plan.tasks), return_exceptions=True)

        texts: list[str] = []
        ok = 0
        fail = 0
        # Collect model ids only for failed workers (one entry per failure)
        failed_worker_ids: list[str] = []
        for r in results:
            if isinstance(r, str):
                texts.append(r)
                ok += 1
            else:
                # Exception / non-str return from run_one or gather
                fail += 1
                failed_worker_ids.append(worker_model_id)

        if ok == 0:
            return OrchestratorResult(
                compacted_context=None,
                plan=plan,
                degraded_to_fast=True,
                reason="all_workers_failed",
                worker_ok=0,
                worker_fail=fail,
                plan_model_id=plan_model_id,
                worker_model_ids=failed_worker_ids or [worker_model_id] * max(fail, len(plan.tasks)),
            )

        compacted = _compact_texts(texts, COMPACTION_TOKEN_LIMIT)
        # Optional LLM compaction if many workers: use cheap model to summarize if >2
        # For now deterministic is fine; keep token limit

        return OrchestratorResult(
            compacted_context=compacted,
            plan=plan,
            degraded_to_fast=False,
            reason="ok",
            worker_ok=ok,
            worker_fail=fail,
            plan_model_id=plan_model_id,
            worker_model_ids=failed_worker_ids,
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
    import re

    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m:
        return m.group(1)
    # last resort return as-is
    return text
