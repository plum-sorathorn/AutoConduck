from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = (
    "You are a coding subagent executing a delegated task inside a repository. "
    "Follow the brief exactly. "
    "Respect all constraints. "
    "Produce output matching the report format. "
    "You may use the provided tools (read-only) to inspect files. "
    "Never claim work you did not do; if blocked, say so explicitly."
)

DEFAULT_BRIEF_TEMPLATE = (
    "TASK {task_id}: {goal}\n"
    "CONTEXT: {context}\n"
    "FILES: {files}\n"
    "{constraints_block}\n"
    "{acceptance_block}\n"
    "{report_block}\n"
    "{round_note}"
)

DEFAULT_FOLLOWUP_TEMPLATE = (
    "TASK {task_id}: {goal}\n"
    "Your previous attempt did not pass verification. Feedback: {feedback}. "
    "Here is your previous output: {prev_output}. "
    "Revise it, addressing every point of the feedback. Do not repeat unchanged.\n"
    "CONTEXT: {context}\n"
    "FILES: {files}\n"
    "{constraints_block}\n"
    "{acceptance_block}\n"
    "{report_block}\n"
    "{round_note}"
)

DEFAULT_ACCEPTANCE_HEADER = "ACCEPTANCE CRITERIA (your result will be checked against these):"

DEFAULT_REPORT_FORMAT_TEMPLATE = "REPORT FORMAT: return a structured result matching: {contract}"

DEFAULT_CONSTRAINTS: list[str] = [
    "Do not modify any files unless the task explicitly requires it",
    "Do not invent file paths — verify with tools if unsure",
    "Return only your findings, not commentary",
]

DEFAULT_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file's content (read-only).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to read"},
                    "offset": {"type": "integer", "description": "Line offset to start reading from", "default": 0},
                    "limit": {"type": "integer", "description": "Maximum number of lines to read", "default": 2000},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search for a regex pattern in files (read-only).",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for"},
                    "path": {"type": "string", "description": "Directory to search in", "default": "."},
                    "include": {"type": ["string", "null"], "description": "File pattern to include (e.g. '*.py')"},
                    "max_matches": {"type": "integer", "description": "Maximum matches to return", "default": 20},
                },
                "required": ["pattern"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files matching a glob pattern (read-only).",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern to match"},
                },
                "required": ["pattern"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List directory entries (read-only).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path to list", "default": "."},
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
]

_ALLOWED_YAML_KEYS = {
    "system_prompt",
    "brief_template",
    "followup_template",
    "acceptance_header",
    "report_format_template",
    "default_constraints",
    "tools",
}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class AcceptanceCheck(BaseModel):
    kind: Literal["file_exists", "contains", "not_contains", "regex", "command", "llm"]
    path: str | None = None
    pattern: str | None = None
    command: str | None = None
    desc: str = ""

    @model_validator(mode="after")
    def _soft_check(self) -> "AcceptanceCheck":
        # Soft validation: do not crash if required per-kind field is missing.
        # This is intentionally non-strict to allow flexible authoring.
        return self


class Playbook(BaseModel):
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    brief_template: str = DEFAULT_BRIEF_TEMPLATE
    followup_template: str = DEFAULT_FOLLOWUP_TEMPLATE
    acceptance_header: str = DEFAULT_ACCEPTANCE_HEADER
    report_format_template: str = DEFAULT_REPORT_FORMAT_TEMPLATE
    default_constraints: list[str] = Field(default_factory=lambda: list(DEFAULT_CONSTRAINTS))
    tools: list[dict[str, Any]] = Field(default_factory=lambda: [dict(t) for t in DEFAULT_TOOLS])

    @classmethod
    def load(cls, path: str | None = None) -> "Playbook":
        """Load playbook from YAML file or fall back to built-in defaults."""
        if path is None:
            return cls()
        p = Path(path)
        if not p.exists():
            return cls()
        try:
            raw = p.read_text(encoding="utf-8")
            data = yaml.safe_load(raw) or {}
            if not isinstance(data, dict):
                logger.warning("playbook YAML at %s is not a mapping; using defaults", p)
                return cls()
            # filter to allowed keys, ignore unknown
            filtered: dict[str, Any] = {k: v for k, v in data.items() if k in _ALLOWED_YAML_KEYS}
            # merge with defaults: missing keys fall back
            # Build instance from defaults then override filtered keys
            defaults = cls()
            merged: dict[str, Any] = defaults.model_dump()
            merged.update(filtered)
            return cls.model_validate(merged)
        except Exception as e:
            logger.warning("failed to load playbook at %s: %s — using defaults", p, e)
            return cls()

    # ------------------------------------------------------------------
    # Render helpers
    # ------------------------------------------------------------------

    def render_constraints(self, constraints: list[str]) -> str:
        items = constraints if constraints else self.default_constraints
        lines = ["CONSTRAINTS:"]
        for i, c in enumerate(items, 1):
            lines.append(f"{i}. {c}")
        return "\n".join(lines)

    def render_acceptance(self, criteria: list[AcceptanceCheck]) -> str:
        if not criteria:
            return ""
        lines = [self.acceptance_header]
        for idx, c in enumerate(criteria, 1):
            # normalize dicts if caller passes dicts (defensive)
            if isinstance(c, dict):
                try:
                    c = AcceptanceCheck.model_validate(c)
                except Exception:
                    # fallback: stringify dict
                    lines.append(f"{idx}. [unknown] — {c}")
                    continue
            parts: list[str] = [f"{idx}. [{c.kind}]"]
            # include relevant fields per kind
            if c.kind == "file_exists":
                if c.path:
                    parts.append(f"path={c.path}")
            elif c.kind in ("contains", "not_contains", "regex"):
                if c.path:
                    parts.append(f"path={c.path}")
                if c.pattern:
                    parts.append(f"pattern={c.pattern}")
            elif c.kind == "command":
                if c.command:
                    parts.append(f"command={c.command}")
                if c.path:
                    parts.append(f"path={c.path}")
                if c.pattern:
                    parts.append(f"pattern={c.pattern}")
            elif c.kind == "llm":
                if c.pattern:
                    parts.append(f"pattern={c.pattern}")
                if c.path:
                    parts.append(f"path={c.path}")
            else:
                # generic fallback
                if c.path:
                    parts.append(f"path={c.path}")
                if c.pattern:
                    parts.append(f"pattern={c.pattern}")
                if c.command:
                    parts.append(f"command={c.command}")
            desc = c.desc or ""
            line = " ".join(parts)
            if desc:
                line += f" — {desc}"
            lines.append(line)
        return "\n".join(lines)

    def render_report_format(self, output_contract: str) -> str:
        return self.report_format_template.format(contract=output_contract)

    def render_brief(
        self,
        task: Any,
        round_no: int,
        prev_output: str | None = None,
        feedback: str | None = None,
    ) -> list[dict[str, str]]:
        # Extract task fields via getattr with defaults; support dicts too
        def _get(obj: Any, key: str, default: Any) -> Any:
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        task_id = _get(task, "id", "unknown")
        goal = _get(task, "goal", "")
        context = _get(task, "context", "") or ""
        files = _get(task, "files", []) or []
        constraints = _get(task, "constraints", []) or []
        acceptance = _get(task, "acceptance", []) or []
        output_contract = _get(task, "output_contract", "") or ""
        max_rounds = _get(task, "max_rounds", 3)
        try:
            max_rounds = int(max_rounds)
        except Exception:
            max_rounds = 3

        # Normalize files display
        if isinstance(files, list):
            files_str = ", ".join(str(f) for f in files) if files else "(none)"
        else:
            files_str = str(files) if files else "(none)"

        # Context may be empty — template expects {context} placeholder
        # Pass raw context string; caller template includes CONTEXT: prefix
        context_val = context if context else ""

        # Build blocks
        constraints_block = self.render_constraints(constraints if isinstance(constraints, list) else [])

        # Normalize acceptance list to AcceptanceCheck objects
        norm_acceptance: list[AcceptanceCheck] = []
        for a in acceptance:
            if isinstance(a, AcceptanceCheck):
                norm_acceptance.append(a)
            elif isinstance(a, dict):
                try:
                    norm_acceptance.append(AcceptanceCheck.model_validate(a))
                except Exception:
                    continue
            else:
                # try getattr-based dict conversion
                try:
                    norm_acceptance.append(AcceptanceCheck.model_validate(dict(a)))
                except Exception:
                    continue
        acceptance_block = self.render_acceptance(norm_acceptance)

        if isinstance(output_contract, str) and output_contract.strip():
            report_block = self.render_report_format(output_contract)
        else:
            report_block = "Return a concise result for this subtask only."

        round_note = f"ROUND {round_no} of up to {max_rounds}"

        # Choose template
        if round_no >= 2:
            tpl = self.followup_template
            content = tpl.format(
                task_id=task_id,
                goal=goal,
                context=context_val,
                files=files_str,
                constraints_block=constraints_block,
                acceptance_block=acceptance_block,
                report_block=report_block,
                round_note=round_note,
                prev_output=prev_output or "",
                feedback=feedback or "",
            )
        else:
            tpl = self.brief_template
            content = tpl.format(
                task_id=task_id,
                goal=goal,
                context=context_val,
                files=files_str,
                constraints_block=constraints_block,
                acceptance_block=acceptance_block,
                report_block=report_block,
                round_note=round_note,
            )

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": content},
        ]
