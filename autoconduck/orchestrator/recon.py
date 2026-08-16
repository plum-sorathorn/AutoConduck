"""Lightweight pre-planner reconnaissance module for target file discovery."""

import logging
from typing import Any
from pydantic import BaseModel, Field

from .planner import _extract_file_paths
from autoconduck.jsonutil import parse_json_text


class ReconTarget(BaseModel):
    files: list[str] = Field(default_factory=list)
    query: str = ""
    reasoning: str = ""


RECON_SYSTEM_PROMPT = """You are a codebase reconnaissance assistant. Return ONLY JSON matching the schema below.
Given a user request, identify up to 5 plausible relative file paths in a standard codebase that should be inspected to answer or fulfill the request.
Example: {"files": ["autoconduck/routing/dispatcher.py", "autoconduck/config.py"], "query": "dispatcher route logic", "reasoning": "Inspecting routing dispatcher and config."}
"""


def _recon_model_name(cfg=None, task_value: float = 0.5) -> str:
    try:
        from autoconduck.routing import pricing

        if cfg is None:
            from autoconduck.config import get_config

            cfg = get_config()
        bands = getattr(getattr(cfg, "selection", None), "phase_bands", {}) or {}
        lo, hi = bands.get("recon", [0.10, 0.45])
        max_cost = float(getattr(getattr(cfg, "selection", None), "max_file_read_scaled_cost", 0.55))
        from autoconduck.config import resolve_orchestrator_model

        return pricing.select_closest(
            pricing.pool_ids(cfg), lo + (hi - lo) * task_value, cfg, band=(lo, hi), max_scaled_cost=max_cost
        ) or resolve_orchestrator_model(cfg)
    except Exception:
        pass
    return "gpt-4o-mini"


def build_recon_plan(
    messages: list, client=None, cfg=None, task_value: float = 0.5
) -> ReconTarget:
    """Build a lightweight recon plan identifying candidate files to inspect.

    Always asks a cheap/medium model for candidate file targets. Explicit paths are
    supplied as candidates so the model can verify and shortlist them.
    """
    explicit = _extract_file_paths(messages if isinstance(messages, list) else [])
    try:
        from autoconduck.config import get_config
        from autoconduck.server.messages_api import normalize_messages_for_llm, litellm_params_for

        cfg = cfg or get_config()
        recon_model = _recon_model_name(cfg, task_value)
        params = litellm_params_for(recon_model, cfg)
        params["max_tokens"] = 300
        params.setdefault("temperature", 0.0)
        params["drop_params"] = True

        user_text = "\n".join(
            str(m.get("content", "")) if isinstance(m, dict) else str(m)
            for m in messages
            if not isinstance(m, dict) or m.get("role", "user") == "user"
        )

        candidate_hint = (
            f"\n\nCandidate files (verify and select the most relevant up to 5): "
            f"{', '.join(explicit[:12])}"
            if explicit
            else ""
        )
        prompt_messages = normalize_messages_for_llm(
            [
                {"role": "system", "content": __import__("autoconduck.orchestrator.roles", fromlist=["role_card"]).role_card("scout") + "\n" + RECON_SYSTEM_PROMPT if getattr(cfg.selection, "phase_role_cards", True) else RECON_SYSTEM_PROMPT},
                {"role": "user", "content": f"User request:\n{user_text}{candidate_hint}"},
            ]
        )

        schema = ReconTarget.model_json_schema()
        logger = logging.getLogger("autoconduck.orchestrator")
        prompt_log = logger.info if getattr(getattr(cfg, "selection", None), "dump_prompts", True) else logger.debug
        prompt_log("RECON PROMPT:\n%s\n---\n%s", prompt_messages[0]["content"], prompt_messages[1]["content"])
        for attempt in range(2):
            call_params = dict(params)
            if attempt == 1:
                mode = getattr(getattr(cfg, "selection", None), "planner_response_format", "json_object")
                if mode == "json_object":
                    call_params["response_format"] = {"type": "json_object"}
                elif mode == "json_schema":
                    call_params["response_format"] = {"type": "json_schema", "json_schema": {"name": "ReconTarget", "schema": schema, "strict": True}}
            if client is not None and hasattr(client, "completion"):
                resp = client.completion(messages=prompt_messages, **call_params)
            elif client is not None and hasattr(client, "chat") and hasattr(client.chat, "completions"):
                resp = client.chat.completions.create(messages=prompt_messages, **call_params)
            else:
                import litellm
                resp = litellm.completion(messages=prompt_messages, **call_params)
            raw = resp["choices"][0]["message"]["content"] if isinstance(resp, dict) else resp.choices[0].message.content
            parsed, error, _ = parse_json_text(raw)
            if parsed is not None:
                return ReconTarget.model_validate(parsed)
            logging.getLogger("autoconduck.orchestrator").info("Recon attempt %d failed: %s; response preview: %s", attempt + 1, error, raw[:200])
        return ReconTarget()
    except Exception as exc:
        logging.getLogger("autoconduck.orchestrator").debug("Recon LLM fallback: %s", exc)
        return ReconTarget()
