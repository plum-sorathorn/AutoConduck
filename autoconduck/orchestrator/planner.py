"""Structured task planning prompts and validation."""

import json
import logging
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class OutputContract(BaseModel):
    description: str = ""
    verify: list[str] = Field(default_factory=list)

    def __str__(self) -> str:
        return self.description


class SubTask(BaseModel):
    id: str
    goal: str
    scope: list[str]
    output_contract: OutputContract = Field(default_factory=OutputContract)
    constraints: list[str]
    depends_on: list[str] = Field(default_factory=list)
    verified_context: list[str] = Field(default_factory=list)
    read_budget: int = 5
    role: str = "read"

    @field_validator("output_contract", mode="before")
    @classmethod
    def _coerce_output_contract(cls, value: Any) -> Any:
        if isinstance(value, str):
            return OutputContract(description=value)
        return value


class TaskPlan(BaseModel):
    subtasks: list[SubTask]
    summary: str = ""
    budget_hint: float | None = None


PLANNER_SYSTEM_PROMPT = """You are a coding-task planner. Return only JSON matching the supplied schema.
Create a realistic DAG of read-only analysis tasks. Goals are single imperative sentences; scope contains
resolved file paths, never vague descriptions; constraints explicitly say what the analyst must not do.

When FILE CONTENTS are provided below, base each subtask's scope paths on those files and the request.
For every subtask, populate verified_context with up to 8 short factual bullet strings (<15 words each)
extracted directly from the FILE CONTENTS block (e.g. "line 96: SOURCES list is hardcoded, must extend not replace").
For implementation-flavored subtasks, put literal shell commands in output_contract.verify
(e.g. "pytest", "python -m compileall autoconduck"); leave verify empty for pure read/analysis subtasks.
output_contract is an object with description (string) and verify (list of strings).

Worked example — request: refactor auth flow to support refresh tokens
{"subtasks":[{"id":"auth-model","goal":"Inspect the authentication models and token persistence interfaces.","scope":["autoconduck/auth/models.py","autoconduck/auth/tokens.py"],"output_contract":{"description":"List relevant classes, fields, and current token lifecycle with file:line references.","verify":[]},"constraints":["Do not propose code changes.","Do not inspect files outside the listed scope."],"depends_on":[],"verified_context":[],"read_budget":5},{"id":"auth-api","goal":"Trace login and refresh request handling through the HTTP boundary.","scope":["autoconduck/api/auth.py","tests/test_auth.py"],"output_contract":{"description":"Summarize endpoints, validation, and test coverage as evidence bullets.","verify":[]},"constraints":["Do not modify files.","Do not infer behavior without a file:line reference."],"depends_on":["auth-model"],"verified_context":[],"read_budget":5}],"summary":"Inspect token state before API behavior."}

Worked example — request: review payment retry behavior
{"subtasks":[{"id":"retry-code","goal":"Map payment retry and backoff control flow.","scope":["src/payments/retry.py","src/payments/client.py"],"output_contract":{"description":"Provide a numbered control-flow summary and risks with file:line references.","verify":[]},"constraints":["Do not write a patch.","Do not include unrelated modules."],"depends_on":[],"verified_context":[],"read_budget":5}]}
"""

_PATH_RE = re.compile(
    r"""(?x)
    (?P<q>["'`])(?P<pquoted>[^"'`\s]+?\.(?:py|md|json|toml|txt))(?P=q)
    |
    (?P<pplain>(?:[\w.-]+/)+[\w.-]+\.(?:py|md|json|toml|txt))
    """
)


def _extract_file_paths(messages: list[dict]) -> list[str]:
    """Regex-scan message texts for plausible relative paths that exist on disk."""
    texts: list[str] = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    texts.append(part["text"])
                elif isinstance(part, str):
                    texts.append(part)
    found: list[str] = []
    seen: set[str] = set()
    root = Path.cwd()
    for text in texts:
        for m in _PATH_RE.finditer(text):
            path = m.group("pquoted") or m.group("pplain")
            if not path or path in seen:
                continue
            candidate = Path(path)
            if candidate.is_absolute():
                continue
            full = root / candidate
            try:
                if full.is_file():
                    found.append(path.replace("\\", "/"))
                    seen.add(path)
            except OSError:
                continue
    return found


def _read_files(paths: list[str]) -> dict[str, str]:
    """Plain file I/O; skip unreadable paths. Keys are the original path strings."""
    out: dict[str, str] = {}
    root = Path.cwd()
    for path in paths:
        try:
            full = root / path
            if full.is_file():
                out[path] = full.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
    return out


def _format_file_contents(files: dict[str, str]) -> str:
    if not files:
        return ""
    parts = ["\n\nFILE CONTENTS (ground truth for planning):"]
    for path, content in files.items():
        parts.append(f"{path}:\n{content}")
    return "\n".join(parts)


def _model_name(cfg=None, task_value=0.5, config=None) -> str:
    try:
        from autoconduck import pricing

        config = config or cfg
        if config is None:
            from autoconduck.config import get_config

            config = get_config()
        lo, hi = config.selection.phase_bands["planner"]
        return pricing.select_closest(
            pricing.pool_ids(config), lo + (hi - lo) * task_value, config, band=(lo, hi)
        )
    except Exception:
        pass
    return "gpt-4o"


def _completion(
    client: Any,
    messages: list[dict[str, str]],
    cfg=None,
    task_value: float = 0.5,
    **kwargs: Any,
) -> Any:
    from autoconduck.config import orchestrator_litellm_params
    from autoconduck.messages_api import normalize_messages_for_llm

    messages = normalize_messages_for_llm(messages)
    kwargs = {**orchestrator_litellm_params(cfg), **kwargs}
    kwargs.setdefault("max_tokens", 500)
    from autoconduck.config import qualify_model

    kwargs["model"] = qualify_model(_model_name(cfg, task_value=task_value))
    if client is not None:
        if hasattr(client, "completion"):
            return client.completion(messages=messages, **kwargs)
        if hasattr(client, "chat") and hasattr(client.chat, "completions"):
            return client.chat.completions.create(messages=messages, **kwargs)
    import litellm

    return litellm.completion(messages=messages, **kwargs)


def _content(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        return str(response["choices"][0]["message"]["content"])
    return str(response.choices[0].message.content)


def build_task_plan(
    messages: list, client=None, cfg=None, task_value: float = 0.5, ground_truth: str = ""
) -> TaskPlan | None:
    """Ask the planner model for a structured task plan.

    Returns None on any failure so the orchestrator degrades gracefully to the
    direct-executor path rather than spending a second LLM call on a retry.
    """
    try:
        from autoconduck.messages_api import normalize_messages_for_llm

        messages = normalize_messages_for_llm(messages if isinstance(messages, list) else [])
        schema = TaskPlan.model_json_schema()
        paths = _extract_file_paths(messages if isinstance(messages, list) else [])
        file_block = _format_file_contents(_read_files(paths))
        if ground_truth:
            file_block += f"\n\nRECON GROUND TRUTH EVIDENCE:\n{ground_truth}"
        system_content = PLANNER_SYSTEM_PROMPT + file_block
        prompt_messages = [{"role": "system", "content": system_content}, *messages]
        user_msg = "\n".join(
            str(message.get("content", ""))
            for message in messages
            if isinstance(message, dict)
        )
        try:
            logging.getLogger("autoconduck.orchestrator").debug(
                "PLANNER PROMPT:\n%s\n---\n%s", system_content, user_msg
            )
            response = _completion(
                client,
                prompt_messages,
                cfg=cfg,
                task_value=task_value,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "TaskPlan",
                        "schema": schema,
                        "strict": True,
                    },
                },
            )
            raw = _content(response)
            return TaskPlan.model_validate(json.loads(raw))
        except Exception:
            return None
    except (ImportError, ModuleNotFoundError):
        return None
    return None

