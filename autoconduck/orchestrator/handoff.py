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


def format_execution_handoff(
    plan: Any,
    subagent_outputs: dict[str, str],
    compacted: str,
) -> str:
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

    if is_pi and has_subagents and len(subtasks) >= 2:
        sections.append(
            "### Execution Directives\n"
            "You may delegate independent subtasks to `pi-subagents` (e.g. using `workflowScript` / `runs.all(...)` or `subagent`), "
            "or execute them sequentially in the parent session. Proceed with implementation."
        )
    else:
        sections.append(
            "### Execution Directives\n"
            "Proceed with implementation of the subtasks sequentially using available tools (`read`, `edit`, `write`, `bash`)."
        )

    return "\n\n".join(sections)
