"""Execution handoff formatting for universal client coding harnesses."""
from __future__ import annotations

from typing import Any


class ExecutionHandoff(str):
    """String subclass representing formatted handoff markdown with optional attached tool_calls."""

    content: str
    tool_calls: list[dict[str, Any]] | None

    def __new__(cls, content: str, tool_calls: list[dict[str, Any]] | None = None):
        obj = super().__new__(cls, content)
        obj.content = content
        obj.tool_calls = tool_calls
        return obj

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"content": self.content}
        if self.tool_calls:
            data["tool_calls"] = self.tool_calls
        return data


def format_execution_handoff(
    plan: Any,
    subagent_outputs: dict[str, str],
    compacted: str,
    user_agent: str = "",
    client_type: str | None = None,
    is_nested: bool = False,
) -> ExecutionHandoff:
    """Format a clean, structured implementation plan with verified context for the client agent.

    All subagent analysis is orchestrated server-side within AutoConduck. The synthesized
    handoff is delivered as structured markdown directives for the client harness.
    """
    sections: list[str] = []

    summary_header = (plan.summary if plan and plan.summary else "Multi-agent task analysis completed.").strip()
    sections.append(f"## Implementation Plan & Verified Context\n\n{summary_header}")

    subtasks = plan.subtasks if plan and plan.subtasks else []

    if subtasks:
        sections.append("### Subtask Breakdown")
        for i, st in enumerate(subtasks, 1):
            st_id = getattr(st, "id", f"task_{i}")
            findings = subagent_outputs.get(st_id, "").strip()
            scope = getattr(st, "scope", []) or []
            scope_str = ", ".join(f"`{s}`" for s in scope) if scope else "General workspace"
            deps = getattr(st, "depends_on", []) or []
            deps_str = ", ".join(f"`{d}`" for d in deps) if deps else "None (independent)"
            contract = getattr(st, "output_contract", None)
            verify_list = getattr(contract, "verify", None) if contract and not isinstance(contract, str) else None
            verify_cmds = ", ".join(f"`{v}`" for v in verify_list) if verify_list else ""

            goal = getattr(st, "goal", "")
            st_block = [f"#### {i}. `{st_id}`: {goal}"]
            st_block.append(f"- **Scope**: {scope_str}")
            st_block.append(f"- **Dependencies**: {deps_str}")
            if verify_cmds:
                st_block.append(f"- **Verification**: {verify_cmds}")
            constraints = getattr(st, "constraints", []) or []
            if constraints:
                st_block.append(f"- **Constraints**: {'; '.join(constraints)}")
            verified_ctx = getattr(st, "verified_context", None)
            if findings:
                st_block.append(f"- **Analyst Findings & Key Symbols**:\n  {findings}")
            elif verified_ctx:
                st_block.append(f"- **Verified Context**:\n  " + "\n  ".join(f"• {c}" for c in verified_ctx))
            sections.append("\n".join(st_block))

    if compacted and not subtasks:
        sections.append(f"### Key Findings & Architecture\n\n{compacted}")

    sections.append(
        "### Execution Directives\n"
        "Proceed with implementation of the subtasks sequentially using available tools (`read`, `edit`, `write`, `bash`)."
    )

    return ExecutionHandoff("\n\n".join(sections), tool_calls=None)
