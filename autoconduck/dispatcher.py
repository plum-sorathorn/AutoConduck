from dataclasses import dataclass
from typing import Literal
from . import semantic_router, evaluator

def _user_messages(messages: list) -> list:
    return [message for message in messages if not isinstance(message, dict) or message.get("role", "user") == "user"]

@dataclass(frozen=True)
class RoutingDecision:
    path: Literal["fast", "slow"]
    confidence_band: Literal["fast", "slow", "ambiguous"]
    confidence: float
    complexity: float
    model: str | None
    reason: str

def _default_tiebreaker(message, pseudo_model, config):
    try:
        import litellm
        from .config import get_config, orchestrator_litellm_params
        cfg = config or get_config()
        content = getattr(message, "content", str(message))
        params = orchestrator_litellm_params(cfg)
        result = litellm.completion(messages=[{"role":"user", "content": f"Reply only FAST or SLOW. Classify: {content}"}], max_tokens=2, **params)
        answer = result.choices[0].message.content.strip().upper()
        return "slow" if "SLOW" in answer else "fast"
    except Exception:
        return "fast"

def route(messages: list, history, pseudo_model: str = "autoconduck", tiebreaker=None, config=None) -> RoutingDecision:
    if config is None:
        from .config import get_config
        config = get_config()
    enabled = [entry for entry in (getattr(config, "model_list", []) or []) if entry.get("enabled", True)]
    user_messages = _user_messages(messages)
    last = user_messages[-1] if user_messages else ""
    text = last.get("content", "") if isinstance(last, dict) else getattr(last, "content", str(last))
    text = evaluator.clean_routing_text(text)
    match = semantic_router.route(text)
    result = evaluator.score(user_messages, history, match, pseudo_model, config)
    if len(enabled) <= 1 and result.confidence_band == "ambiguous":
        path = "slow" if match.route == "slow_path" else "fast"
        return RoutingDecision(path, "ambiguous", result.confidence, result.complexity, None, f"single-model, router-resolved: {path}")
    if result.confidence_band == "ambiguous":
        try: path = str((tiebreaker or _default_tiebreaker)(messages[-1], pseudo_model, config)).lower()
        except Exception: path = "fast"
        path = "slow" if path.startswith("slow") else "fast"
        return RoutingDecision(path, "ambiguous", result.confidence, result.complexity, None, "tiebreaker: " + path)
    return RoutingDecision(result.path, result.confidence_band, result.confidence, result.complexity, None, result.reason)
