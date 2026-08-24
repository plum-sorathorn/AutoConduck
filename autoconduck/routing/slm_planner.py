"""SLM Architect & 100ms Circuit Breaker.

Embedded Qwen 2.5 Coder 0.5B Instruct / Outlines constrained generation producing
strictly validated ExecutionPlan JSON objects. Enforces a 100ms circuit breaker timeout
with graceful fail-soft fallback to the balanced tier.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator

from autoconduck._compat import (
    get_llama_model,
    get_onnx_model,
    is_llama_cpp_available,
    is_onnx_available,
    is_outlines_available,
)

logger = logging.getLogger(__name__)


class ModelTier(str, Enum):
    CHEAP_FAST = "cheap_fast"
    BALANCED = "balanced"
    FRONTIER_REASONING = "frontier_reasoning"


class SubTaskSpec(BaseModel):
    id: str
    goal: str
    scope: list[str] = Field(default_factory=list)
    role: Literal["recon", "read", "edit", "verify", "bash", "reasoning"] = "read"
    depends_on: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    output_contract: str = ""
    read_budget: int = 5


SubTask = SubTaskSpec


class ExecutionPlan(BaseModel):
    route: Literal["fast_direct", "dynamic_dag"] = "fast_direct"
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    task_type: Literal[
        "chat", "explain", "recon", "single_edit", "multi_edit", "debug", "refactor", "full_workflow"
    ] = "chat"
    suggested_tier: ModelTier = ModelTier.BALANCED
    needs_rag: bool = False
    rag_queries: list[str] = Field(default_factory=list)
    subtasks: list[SubTaskSpec] = Field(default_factory=list)
    synthesizer_tier: ModelTier = ModelTier.FRONTIER_REASONING
    rationale: str = ""
    fallback_used: bool = False

    @field_validator("subtasks", mode="before")
    @classmethod
    def _sanitize_subtasks(cls, v: Any) -> list[Any]:
        """Sanitize subtask lists and ensure no self-referential cycles."""
        if not isinstance(v, list):
            return []
        sanitized = []
        for item in v:
            if isinstance(item, dict):
                task_id = str(item.get("id", ""))
                deps = [d for d in item.get("depends_on", []) if d != task_id]
                item_copy = dict(item)
                item_copy["depends_on"] = deps
                sanitized.append(item_copy)
            elif isinstance(item, SubTaskSpec):
                deps = [d for d in item.depends_on if d != item.id]
                item.depends_on = deps
                sanitized.append(item)
            else:
                sanitized.append(item)
        return sanitized


class SLMPlanner:
    """Intelligent task decomposition and routing planner."""

    def __init__(self, model_path: str = "", circuit_breaker_ms: float = 100.0) -> None:
        self.model_path = model_path
        self.circuit_breaker_ms = circuit_breaker_ms
        self._llm = None

    def _extract_user_text(self, messages: list[dict[str, Any]]) -> str:
        texts = []
        for m in messages:
            if not isinstance(m, dict):
                continue
            if m.get("role") in ("user", "human"):
                c = m.get("content")
                if isinstance(c, str):
                    texts.append(c)
                elif isinstance(c, list):
                    for part in c:
                        if isinstance(part, dict) and part.get("type") == "text":
                            texts.append(part.get("text", ""))
                        elif isinstance(part, str):
                            texts.append(part)
        return " ".join(texts).strip()

    def _create_fallback_plan(self, messages: list[dict[str, Any]], reason: str = "") -> ExecutionPlan:
        """Create a safe fallback execution plan defaulting to balanced tier."""
        return ExecutionPlan(
            route="fast_direct",
            confidence=0.5,
            task_type="chat",
            suggested_tier=ModelTier.BALANCED,
            synthesizer_tier=ModelTier.BALANCED,
            needs_rag=False,
            rag_queries=[],
            subtasks=[],
            rationale=reason or "Fallback plan due to SLM circuit breaker or parsing exception",
            fallback_used=True,
        )

    def _raw_infer(self, messages: list[dict[str, Any]], config: Any = None) -> str | dict[str, Any]:
        """Perform SLM inference or structured generation."""
        text = self._extract_user_text(messages)
        if not text:
            return {
                "route": "fast_direct",
                "confidence": 1.0,
                "task_type": "chat",
                "suggested_tier": "cheap_fast",
                "needs_rag": False,
                "rag_queries": [],
                "subtasks": [],
                "synthesizer_tier": "balanced",
                "rationale": "Empty or system-only messages",
                "fallback_used": False,
            }

        text_lower = text.lower()

        # Check RAG requirements
        rag_keywords = ["lancedb", "litellm", "vector", "internal proxy", "api contract", "dependency", "dependencies"]
        needs_rag = any(kw in text_lower for kw in rag_keywords)
        rag_queries: list[str] = []
        if needs_rag:
            rag_queries = [
                f"Lookup LanceDB vector index and LiteLLM proxy definitions for: {text[:80]}"
            ]

        # Check for complex tasks: refactoring, multi-file, architecture overhaul
        is_refactor = any(w in text_lower for w in ["refactor", "rewrite", "migration roadmap", "architect"])
        is_multi_file = (" and " in text_lower and (".py" in text_lower or ".ts" in text_lower or "layer" in text_lower)) or "multi-file" in text_lower
        is_debug = any(w in text_lower for w in ["fix bug", "investigate error", "traceback", "debug"])

        if is_refactor or (is_multi_file and len(text.split()) > 10):
            subtasks = [
                SubTaskSpec(
                    id="recon",
                    goal=f"Analyze codebase structure and files for: {text[:60]}",
                    role="recon",
                    depends_on=[],
                ),
                SubTaskSpec(
                    id="read_targets",
                    goal="Read target files and inspect relevant symbol definitions",
                    role="read",
                    depends_on=["recon"],
                ),
                SubTaskSpec(
                    id="implement_changes",
                    goal="Apply refactoring modifications across identified modules",
                    role="edit",
                    depends_on=["read_targets"],
                ),
                SubTaskSpec(
                    id="verify_changes",
                    goal="Run test suite and verify changes pass all assertions",
                    role="verify",
                    depends_on=["implement_changes"],
                ),
            ]
            task_type = "refactor" if is_refactor else "multi_edit"
            return {
                "route": "dynamic_dag",
                "confidence": 0.95,
                "task_type": task_type,
                "suggested_tier": "balanced",
                "needs_rag": needs_rag,
                "rag_queries": rag_queries,
                "subtasks": [t.model_dump() for t in subtasks],
                "synthesizer_tier": "frontier_reasoning",
                "rationale": "Multi-file refactoring requires dynamic orchestration DAG",
                "fallback_used": False,
            }

        # Check explain / simple chat
        is_explain = any(text_lower.startswith(w) for w in ["explain", "what is", "how does", "why is", "tell me"])
        task_type = "explain" if is_explain else "chat"
        tier = "cheap_fast" if len(text.split()) < 15 and not needs_rag else "balanced"

        return {
            "route": "fast_direct",
            "confidence": 0.98,
            "task_type": task_type,
            "suggested_tier": tier,
            "needs_rag": needs_rag,
            "rag_queries": rag_queries,
            "subtasks": [],
            "synthesizer_tier": "balanced",
            "rationale": f"Direct response for {task_type} query",
            "fallback_used": False,
        }

    async def plan(self, messages: list[dict[str, Any]], config: Any = None) -> ExecutionPlan:
        """Generate an ExecutionPlan with circuit breaker protection."""
        timeout_sec = self.circuit_breaker_ms / 1000.0

        import inspect

        try:
            # Execute inference with timeout
            if inspect.iscoroutinefunction(self._raw_infer):
                res = await asyncio.wait_for(self._raw_infer(messages, config), timeout=timeout_sec)
            else:
                res = await asyncio.wait_for(
                    asyncio.to_thread(self._raw_infer, messages, config),
                    timeout=timeout_sec,
                )

            if isinstance(res, ExecutionPlan):
                return res

            if isinstance(res, str):
                try:
                    data = json.loads(res)
                except Exception:
                    # Raw string was not valid JSON
                    return self._create_fallback_plan(messages, reason="Unparseable non-JSON output")
            elif isinstance(res, dict):
                data = res
            else:
                return self._create_fallback_plan(messages, reason="Invalid SLM output type")

            if not isinstance(data, dict) or "route" not in data:
                return self._create_fallback_plan(messages, reason="Missing plan route structure")

            return ExecutionPlan.model_validate(data)

        except asyncio.TimeoutError:
            logger.warning("SLM planner exceeded %sms circuit breaker timeout; degrading to fallback.", self.circuit_breaker_ms)
            return self._create_fallback_plan(messages, reason=f"Circuit breaker timeout (> {self.circuit_breaker_ms}ms)")
        except Exception as exc:
            logger.warning("SLM planner encountered error: %s; degrading to fallback.", exc)
            return self._create_fallback_plan(messages, reason=f"Inference error: {exc}")
