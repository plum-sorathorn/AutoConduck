"""Execution handoff formatting and agent extension detection."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def check_subagent_support() -> tuple[bool, bool]:
    """Check if Pi is configured and whether the pi-subagents package/extension is installed.

    Returns (is_pi_configured, has_subagents).
    """
    try:
        agent_dir = (
            Path(os.environ.get("PI_CODING_AGENT_DIR", "")).expanduser()
            if os.environ.get("PI_CODING_AGENT_DIR")
            else Path.home() / ".pi" / "agent"
        )
        settings_file = agent_dir / "settings.json"
        if not settings_file.is_file():
            return False, False
        data = json.loads(settings_file.read_text(encoding="utf-8"))
        packages = data.get("packages", [])
        if not isinstance(packages, list):
            packages = []
        has_sub = any("pi-subagents" in str(p) for p in packages)
        if not has_sub:
            ext_dir = agent_dir / "extensions"
            if ext_dir.is_dir() and any("subagent" in f.name.lower() for f in ext_dir.iterdir()):
                has_sub = True
        return True, has_sub
    except Exception:
        return False, False


class ExecutionHandoff(str):
    """String subclass representing formatted handoff markdown with attached tool_calls."""

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


def build_subagent_workflow_script(plan: Any, subagent_outputs: dict[str, str]) -> str | None:
    """Generate a clean JavaScript workflowScript for pi-subagents."""
    subtasks = plan.subtasks if plan and hasattr(plan, "subtasks") and plan.subtasks else []
    if not subtasks:
        return None

    # Organize subtasks into dependency waves
    subtask_map = {st.id: st for st in subtasks}
    completed_ids: set[str] = set()
    waves: list[list[Any]] = []
    remaining = list(subtasks)

    while remaining:
        current_wave = [
            st
            for st in remaining
            if all(dep in completed_ids or dep not in subtask_map for dep in getattr(st, "depends_on", []))
        ]
        if not current_wave:
            current_wave = remaining
            remaining = []
        else:
            remaining = [st for st in remaining if st not in current_wave]

        for st in current_wave:
            completed_ids.add(st.id)
        waves.append(current_wave)

    def _format_subtask_prompt(st: Any) -> str:
        findings = subagent_outputs.get(st.id, "").strip()
        lines = [f"Goal: {st.goal}"]
        if getattr(st, "scope", None):
            lines.append(f"Target Scope: {', '.join(st.scope)}")
        if getattr(st, "constraints", None):
            lines.append(f"Constraints: {'; '.join(st.constraints)}")
        if findings:
            lines.append(f"Analyst Findings & Evidence:\n{findings}")
        elif getattr(st, "verified_context", None):
            lines.append("Verified Facts:\n" + "\n".join(f"• {c}" for c in st.verified_context))
        return "\n".join(lines)

    lines: list[str] = []
    return_vars: list[str] = []

    if len(waves) == 1 and len(waves[0]) == 1:
        st = waves[0][0]
        task_prompt = _format_subtask_prompt(st)
        verify_cmd = (
            st.output_contract.verify[0]
            if getattr(st, "output_contract", None) and getattr(st.output_contract, "verify", None)
            else None
        )
        item_dict: dict[str, Any] = {
            "agent": "worker",
            "task": task_prompt,
        }
        if verify_cmd:
            item_dict["gate"] = verify_cmd
        item_json = json.dumps(item_dict, indent=2)
        indented_item = "\n".join("  " + l if i else l for i, l in enumerate(item_json.splitlines()))
        return f'return (await runs.run("{st.id}", {indented_item})).output;'

    for wave_idx, wave in enumerate(waves, 1):
        wave_var = f"wave{wave_idx}" if len(waves) > 1 else "results"
        return_vars.append(wave_var)
        items = []
        for st in wave:
            task_prompt = _format_subtask_prompt(st)
            verify_cmd = (
                st.output_contract.verify[0]
                if getattr(st, "output_contract", None) and getattr(st.output_contract, "verify", None)
                else None
            )
            item_dict = {
                "key": st.id,
                "agent": "worker",
                "task": task_prompt,
            }
            if verify_cmd:
                item_dict["gate"] = verify_cmd
            items.append(item_dict)

        items_json = json.dumps(items, indent=2)
        indented_items = "\n".join("  " + l for l in items_json.splitlines())
        lines.append(f"const {wave_var} = await runs.all(\n{indented_items}\n);")

    if len(return_vars) == 1:
        lines.append(f"return {return_vars[0]}.map(r => r.output);")
    else:
        combined = ", ".join(f"...{v}" for v in return_vars)
        lines.append(f"return [{combined}].map(r => r.output);")

    return "\n\n".join(lines)


def build_subagent_tool_call(plan: Any, subagent_outputs: dict[str, str]) -> dict[str, Any] | None:
    """Build a tool_call dict invoking pi-subagents if supported."""
    is_pi, has_subagents = check_subagent_support()
    if not (is_pi and has_subagents):
        return None
    script = build_subagent_workflow_script(plan, subagent_outputs)
    if not script:
        return None
    import time

    call_id = f"call_subagent_{int(time.time())}"
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "subagent",
            "arguments": json.dumps({"workflowScript": script}),
        },
    }


def format_execution_handoff(
    plan: Any,
    subagent_outputs: dict[str, str],
    compacted: str,
) -> ExecutionHandoff:
    """Format a clean, structured implementation plan with verified context for the client agent."""
    is_pi, has_subagents = check_subagent_support()
    sections: list[str] = []

    # Warning / Notice if Pi is configured but pi-subagents extension is missing
    if is_pi and not has_subagents:
        sections.append(
            "> [!NOTE]\n"
            "> **Linear Execution Mode**: The `pi-subagents` extension is not detected in your Pi configuration. "
            "Executing subtasks linearly in the main session. "
            "(To enable parallel subagent delegation, add `\"npm:pi-subagents\"` to `~/.pi/agent/settings.json`)."
        )

    summary_header = (plan.summary if plan and plan.summary else "Multi-agent task analysis completed.").strip()
    sections.append(f"## Implementation Plan & Verified Context\n\n{summary_header}")

    subtasks = plan.subtasks if plan and plan.subtasks else []
    if subtasks:
        sections.append("### Subtask Breakdown")
        for i, st in enumerate(subtasks, 1):
            findings = subagent_outputs.get(st.id, "").strip()
            scope_str = ", ".join(f"`{s}`" for s in st.scope) if st.scope else "General workspace"
            deps_str = ", ".join(f"`{d}`" for d in st.depends_on) if st.depends_on else "None (independent)"
            verify_cmds = (
                ", ".join(f"`{v}`" for v in st.output_contract.verify)
                if hasattr(st, "output_contract") and st.output_contract and getattr(st.output_contract, "verify", None)
                else ""
            )

            st_block = [f"#### {i}. `{st.id}`: {st.goal}"]
            st_block.append(f"- **Scope**: {scope_str}")
            st_block.append(f"- **Dependencies**: {deps_str}")
            if verify_cmds:
                st_block.append(f"- **Verification**: {verify_cmds}")
            if st.constraints:
                st_block.append(f"- **Constraints**: {'; '.join(st.constraints)}")
            if findings:
                st_block.append(f"- **Analyst Findings & Key Symbols**:\n  {findings}")
            elif st.verified_context:
                st_block.append(f"- **Verified Context**:\n  " + "\n  ".join(f"• {c}" for c in st.verified_context))
            sections.append("\n".join(st_block))

    if compacted and not subtasks:
        sections.append(f"### Key Findings & Architecture\n\n{compacted}")

    tool_call = build_subagent_tool_call(plan, subagent_outputs) if (is_pi and has_subagents and subtasks) else None

    if is_pi and has_subagents and len(subtasks) >= 1:
        sections.append(
            "### Execution Directives\n"
            "Delegating subtasks to `pi-subagents` via `workflowScript` execution."
        )
    else:
        sections.append(
            "### Execution Directives\n"
            "Proceed with implementation of the subtasks sequentially using available tools (`read`, `edit`, `write`, `bash`)."
        )

    return ExecutionHandoff("\n\n".join(sections), tool_calls=[tool_call] if tool_call else None)

