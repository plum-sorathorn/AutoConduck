"""Structured task planning prompts and validation."""

import json
from typing import Any

from pydantic import BaseModel, Field


class SubTask(BaseModel):
    id: str
    goal: str
    scope: list[str]
    output_contract: str
    constraints: list[str]
    depends_on: list[str] = Field(default_factory=list)


class TaskPlan(BaseModel):
    subtasks: list[SubTask]
    summary: str = ""


PLANNER_SYSTEM_PROMPT = """You are a coding-task planner. Return only JSON matching the supplied schema.
Create a realistic DAG of read-only analysis tasks. Goals are single imperative sentences; scope contains
resolved file paths, never vague descriptions; constraints explicitly say what the analyst must not do.

Worked example — request: refactor auth flow to support refresh tokens
{"subtasks":[{"id":"auth-model","goal":"Inspect the authentication models and token persistence interfaces.","scope":["autoconduck/auth/models.py","autoconduck/auth/tokens.py"],"output_contract":"List relevant classes, fields, and current token lifecycle with file:line references.","constraints":["Do not propose code changes.","Do not inspect files outside the listed scope."],"depends_on":[]},{"id":"auth-api","goal":"Trace login and refresh request handling through the HTTP boundary.","scope":["autoconduck/api/auth.py","tests/test_auth.py"],"output_contract":"Summarize endpoints, validation, and test coverage as evidence bullets.","constraints":["Do not modify files.","Do not infer behavior without a file:line reference."],"depends_on":["auth-model"]}],"summary":"Inspect token state before API behavior."}

Worked example — request: review payment retry behavior
{"subtasks":[{"id":"retry-code","goal":"Map payment retry and backoff control flow.","scope":["src/payments/retry.py","src/payments/client.py"],"output_contract":"Provide a numbered control-flow summary and risks with file:line references.","constraints":["Do not write a patch.","Do not include unrelated modules."],"depends_on":[]}]}
"""


def _model_name() -> str:
    try:
        from autoconduck import config  # type: ignore
        for name in ("FAST_MODEL", "fast_model", "DEFAULT_FAST_MODEL"):
            value = getattr(config, name, None)
            if isinstance(value, str) and value:
                return value
    except Exception:
        pass
    return "gpt-4o-mini"


def _completion(client: Any, messages: list[dict[str, str]], **kwargs: Any) -> Any:
    if client is not None:
        if hasattr(client, "completion"):
            return client.completion(model=_model_name(), messages=messages, **kwargs)
        if hasattr(client, "chat") and hasattr(client.chat, "completions"):
            return client.chat.completions.create(model=_model_name(), messages=messages, **kwargs)
    import litellm
    return litellm.completion(model=_model_name(), messages=messages, **kwargs)


def _content(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        return str(response["choices"][0]["message"]["content"])
    return str(response.choices[0].message.content)


def build_task_plan(messages: list, client=None) -> TaskPlan | None:
    """Ask the fast model for a plan; tolerate unavailable dependencies and bad models."""
    try:
        schema = TaskPlan.model_json_schema()
        prompt_messages = [{"role": "system", "content": PLANNER_SYSTEM_PROMPT}, *messages]
        for _ in range(2):
            try:
                response = _completion(client, prompt_messages, response_format={
                    "type": "json_schema", "json_schema": {"name": "TaskPlan", "schema": schema, "strict": True}
                })
                raw = _content(response)
                return TaskPlan.model_validate(json.loads(raw))
            except Exception:
                continue
    except (ImportError, ModuleNotFoundError):
        return None
    return None
