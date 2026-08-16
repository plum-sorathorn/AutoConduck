"""Strict slow-path role cards. Keywords: review/check/validate/diff/audit -> reviewer;
research/source/docs/verify-external/find-docs -> researcher; opinion/decision/risk/challenge/advise -> oracle;
investigate/explore/scout/general -> delegate; otherwise worker."""
from dataclasses import dataclass
@dataclass(frozen=True)
class RoleConfig:
    name: str; description: str; aliases: tuple[str,...]=(); tools: tuple[str,...]=(); thinking: str="medium"; band_bias: float=1.0; system_prompt: str=""
_S="You are a scouting subagent. Map the minimum context another agent needs to act: relevant entry points, key types/functions, data flow, files likely to need changes, constraints, risks, open questions. Prefer targeted search over whole-file reads. Cite exact file paths and line ranges. Do not propose or perform edits."
_R="You are a research subagent. Produce a concise, well-sourced brief answering the question directly. Prefer primary sources over commentary. Drop stale or redundant material. Do not edit files."
_W="You are `worker`: the implementation subagent. You are the single writer thread. Execute the assigned task or approved plan with narrow, coherent edits. The main agent and user remain the decision authority. Treat an approved plan as the contract: validate it against the actual code, but do not silently make new product, architecture, or scope decisions."
_V="You are a disciplined review subagent. Inspect, evaluate, and report findings with evidence. You do not guess; you verify from code, tests, docs, or requirements. Do not use shell commands or write files. Do not invent issues — only report problems you can justify from evidence."
_O="You are the oracle: a decision-consistency advisor. Reconstruct the key decisions, constraints, and open questions; treat them as the baseline contract and prevent drift. You are not the executor and do not silently become a second decision-maker. Do not edit files or write code."
_D="You are a delegated agent. Execute the assigned task directly and efficiently, staying close to the parent session's intent. Keep the response focused on the requested work."
ROLES={
"scout":RoleConfig("scout","Recon",("recon","explorer"),("read","grep","glob","list"),band_bias=.85,system_prompt=_S),
"researcher":RoleConfig("researcher","Research",tools=("read","glob","grep","fetch"),system_prompt=_R),
"worker":RoleConfig("worker","Implementation",("developer","coder","implementer"),("read","grep","glob","list","edit","write","bash"),system_prompt=_W),
"reviewer":RoleConfig("reviewer","Review",("code-reviewer",),("read","grep","glob","list"),band_bias=.85,system_prompt=_V),
"oracle":RoleConfig("oracle","Advice",("advisor",),("read","grep","glob","list"),system_prompt=_O),
"delegate":RoleConfig("delegate","Delegation",tools=("read","grep","glob","list","edit","write","bash"),system_prompt=_D),
"planner":RoleConfig("planner","Planning",system_prompt="You are the planner: translate the task into an ordered subtask plan (TaskPlan schema). Preserve constraints. Return only JSON."),
"compactor":RoleConfig("compactor","Compaction",system_prompt="You are the compactor: summarize the subagent findings into a concise, faithful context for the executor. Preserve constraints, risks, and open questions. Do not add new decisions or requirements."),
"executor":RoleConfig("executor","Execution",tools=("read","grep","glob","list","edit","write","bash"),system_prompt="You are the executor: synthesize the comprehensive implementation blueprint. Provide concrete, narrow, coherent edits and directives to all affected files, including exact code snippets and sequential subtasks for the host agent and its subagents to execute."),}
def assign_subagent_role(goal:str)->str:
    t=goal.lower()
    for role, words in (("reviewer",("review","check","validate","diff","audit")),("researcher",("research","source","docs","verify-external","find-docs")),("oracle",("opinion","decision","risk","challenge","advise")),("delegate",("investigate","explore","scout","general"))):
        if any(w in t for w in words): return role
    return "worker"
def role_card(name:str)->str: return ROLES.get(name,ROLES["delegate"]).system_prompt
