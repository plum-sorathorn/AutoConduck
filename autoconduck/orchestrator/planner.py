"""Structured task planning prompts and validation."""

import logging
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator
from autoconduck.jsonutil import _extract_json, parse_json_text


class OutputContract(BaseModel):
    description: str = ""
    verify: list[str] = Field(default_factory=list)

    @field_validator("description", mode="before")
    @classmethod
    def _coerce_description(cls, value: Any) -> str:
        if isinstance(value, list):
            return " ".join(str(x) for x in value if x is not None)
        return str(value or "")

    @field_validator("verify", mode="before")
    @classmethod
    def _coerce_verify(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            return [value] if value.strip() else []
        if isinstance(value, list):
            return [str(x).strip() for x in value if x is not None and str(x).strip()]
        return []

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

    @field_validator("id", mode="before")
    @classmethod
    def _coerce_id(cls, value: Any) -> str:
        return str(value or "task")

    @field_validator("goal", mode="before")
    @classmethod
    def _coerce_goal(cls, value: Any) -> str:
        return str(value or "Analyze task area")

    @field_validator("scope", "constraints", "depends_on", "verified_context", mode="before")
    @classmethod
    def _coerce_str_list(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            return [s.strip() for s in value.split(",") if s.strip()] if value else []
        if isinstance(value, list):
            return [str(item).strip() for item in value if item is not None and str(item).strip()]
        return []

    @field_validator("output_contract", mode="before")
    @classmethod
    def _coerce_output_contract(cls, value: Any) -> Any:
        if isinstance(value, OutputContract):
            return value
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            return OutputContract(description=value)
        if isinstance(value, list):
            return OutputContract(description=" ".join(str(item) for item in value if item is not None))
        return OutputContract(description=str(value or ""))

    @field_validator("read_budget", mode="before")
    @classmethod
    def _coerce_read_budget(cls, value: Any) -> int:
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            digits = re.findall(r"\d+", value)
            if digits:
                return int(digits[0])
        return 5

    @field_validator("role", mode="before")
    @classmethod
    def _coerce_role(cls, value: Any) -> str:
        return str(value or "read")


class TaskPlan(BaseModel):
    subtasks: list[SubTask] = Field(default_factory=list)
    summary: str = ""
    budget_hint: float | None = None

    @field_validator("subtasks", mode="before")
    @classmethod
    def _coerce_subtasks(cls, value: Any) -> list[Any]:
        if isinstance(value, dict):
            return list(value.values())
        if isinstance(value, list):
            return value
        return []


PLANNER_SYSTEM_PROMPT = """You are a coding-task planner. Return only JSON matching the supplied schema.
Create a realistic, focused DAG of at most 3 to 5 critical read-only analysis tasks (do not generate more than 5 subtasks).
Goals are single imperative sentences; scope contains resolved file paths, never vague descriptions;
constraints explicitly say what the analyst must not do.

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
    (?P<q>["'`])(?P<pquoted>[^"'`\s]+?\.(?:py|js|ts|tsx|jsx|md|json|toml|yaml|yml|txt|tmp))(?P=q)
    |
    (?P<pplain>[\w./\\-]+\.(?:py|js|ts|tsx|jsx|md|json|toml|yaml|yml|txt|tmp))\b
    """
)


from .skeletons import is_ignored_path, load_gitignore_patterns, format_structural_context


def _extract_file_paths(
    messages: list[dict], root: Path | None = None, max_paths: int = 6
) -> list[str]:
    """Regex-scan recent message texts for plausible relative paths that exist on disk, filtering gitignore."""
    if not messages:
        return []
    root = root or Path.cwd()
    patterns = load_gitignore_patterns(root)

    # Focus on the most recent user turn (and at most last 4 messages) to avoid ancient transcript tool leaks
    recent_msgs = messages[-4:] if len(messages) > 4 else messages
    texts: list[str] = []
    for msg in recent_msgs:
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
    for text in texts:
        for m in _PATH_RE.finditer(text):
            path = m.group("pquoted") or m.group("pplain")
            if not path or path in seen:
                continue
            candidate = Path(path)
            if candidate.is_absolute():
                continue
            norm = path.replace("\\", "/")
            if is_ignored_path(norm, root, patterns):
                continue
            full = root / candidate
            try:
                if full.is_file():
                    found.append(norm)
                    seen.add(norm)
                    if len(found) >= max_paths:
                        return found
            except OSError:
                continue
    return found


def _read_files(
    paths: list[str], root: Path | None = None, max_file_bytes: int = 150_000
) -> dict[str, str]:
    """Plain file I/O; skip unreadable or excessively large files."""
    out: dict[str, str] = {}
    root = root or Path.cwd()
    for path in paths:
        try:
            full = root / path
            if full.is_file():
                if full.stat().st_size > max_file_bytes:
                    continue
                out[path] = full.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
    return out


def _format_file_contents(files: dict[str, str], ground_truth: str = "") -> str:
    """Format AST skeletons, cross-file dependency maps, and recon scout ground truth."""
    return format_structural_context(files, ground_truth=ground_truth)


def _select_planner_model(retry: bool = False, cfg=None, task_value=0.5, config=None) -> str:
    try:
        from autoconduck.routing import pricing

        config = config or cfg
        if config is None:
            from autoconduck.config import get_config

            config = get_config()
        override = getattr(config.selection, "planner_model_override", None)
        if override:
            from autoconduck.config import qualify_model

            return qualify_model(str(override).strip())
        if retry and getattr(config.selection, "planner_retry_cheaper", False):
            return pricing.cheapest_enabled(config) or "gpt-4o"
        lo, hi = config.selection.phase_bands["planner"]
        target = hi if retry else (lo + (hi - lo) * task_value)
        from autoconduck.config import resolve_orchestrator_model

        return pricing.select_closest(
            pricing.pool_ids(config), target, config, band=(lo, hi)
        ) or resolve_orchestrator_model(config)
    except Exception:
        pass
    return "gpt-4o"


def _model_name(cfg=None, task_value=0.5, config=None) -> str:
    return _select_planner_model(False, cfg, task_value, config)


def _completion(
    client: Any,
    messages: list[dict[str, str]],
    cfg=None,
    task_value: float = 0.5,
    retry: bool = False,
    **kwargs: Any,
) -> Any:
    from autoconduck.server.messages_api import normalize_messages_for_llm, litellm_params_for

    messages = normalize_messages_for_llm(messages)
    planner_model = _select_planner_model(retry, cfg, task_value=task_value)
    params = litellm_params_for(planner_model, cfg)
    kwargs = {**params, **kwargs}
    kwargs.setdefault("temperature", 0.0)
    kwargs["drop_params"] = True
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
        choices = response.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {})
        content = message.get("content")
        if content is None:
            content = message.get("reasoning_content") or ""
        return str(content)
    if hasattr(response, "choices") and response.choices:
        msg = response.choices[0].message
        content = getattr(msg, "content", None)
        if content is None:
            content = getattr(msg, "reasoning_content", None) or ""
        return str(content)
    return ""


def build_task_plan(
    messages: list, client=None, cfg=None, task_value: float = 0.5, ground_truth: str = ""
) -> TaskPlan | None:
    """Ask the planner model for a structured task plan.

    Returns None on any failure so the orchestrator degrades gracefully to the
    direct-executor path rather than spending a second LLM call on a retry.
    """
    try:
        from autoconduck.server.messages_api import normalize_messages_for_llm

        messages = normalize_messages_for_llm(messages if isinstance(messages, list) else [])
        schema = TaskPlan.model_json_schema()
        paths = _extract_file_paths(messages if isinstance(messages, list) else [])
        file_block = _format_file_contents(_read_files(paths), ground_truth=ground_truth)
        from .roles import role_card
        system_content = (role_card("planner") + "\n" if getattr(getattr(cfg, "selection", None), "phase_role_cards", True) else "") + PLANNER_SYSTEM_PROMPT + file_block
        user_messages = [
            message for message in messages
            if isinstance(message, dict) and message.get("role") == "user"
        ]
        prompt_messages = [{"role": "system", "content": system_content}, *user_messages]
        user_msg = "\n".join(
            str(message.get("content", ""))
            for message in messages
            if isinstance(message, dict)
        )
        logger = logging.getLogger("autoconduck.orchestrator")
        first_model = _select_planner_model(False, cfg, task_value=task_value)
        logger.info("Planning with model %s", first_model)
        prompt_log = logger.info if getattr(getattr(cfg, "selection", None), "dump_prompts", True) else logger.debug
        prompt_log("PLANNER PROMPT:\n%s\n---\n%s", system_content, user_msg)
        raw = ""
        for attempt in range(2):
            try:
                kwargs = {}
                mode = getattr(getattr(cfg, "selection", None), "planner_response_format", "json_object")
                if mode == "json_object":
                    kwargs["response_format"] = {"type": "json_object"}
                elif mode == "json_schema":
                    kwargs["response_format"] = {"type": "json_schema", "json_schema": {"name": "TaskPlan", "schema": schema, "strict": True}}
                retry = attempt == 1
                model = _select_planner_model(retry, cfg, task_value=task_value)
                if retry and model != first_model:
                    logger.info("Planner attempt 2 using fallback model %s (was %s)", model, first_model)
                response = _completion(client, prompt_messages, cfg=cfg, task_value=task_value, retry=retry, **kwargs)
                raw = _content(response)
                parsed, repair_error, preview = parse_json_text(raw)
                if parsed is None:
                    raise ValueError(repair_error or "no JSON object found")
                if repair_error:
                    logger.info("Planner JSON repaired by %s", repair_error)
                return TaskPlan.model_validate(parsed)
            except Exception as exc:
                logger.info("Planner attempt %d failed: %s; response preview: %s", attempt + 1, exc, raw[:300])
        return None
    except (ImportError, ModuleNotFoundError):
        return None
    return None
