from dataclasses import dataclass
from typing import Literal
from . import semantic_router, evaluator, pricing

def _user_messages(messages: list) -> list:
    return [message for message in messages if not isinstance(message, dict) or message.get("role", "user") == "user"]

@dataclass(frozen=True)
class RoutingDecision:
    path: Literal["fast", "slow"]
    confidence_band: Literal["fast", "slow", "ambiguous"]
    confidence: float
    complexity: float
    reason: str
    model: str | None = None

def _default_tiebreaker(message, pseudo_model, config):
    try:
        import litellm
        from .config import get_config, orchestrator_litellm_params
        cfg = config or get_config()
        content = getattr(message, "content", str(message))
        params = orchestrator_litellm_params(cfg)
        result = litellm.completion(messages=[{"role":"user", "content": f"Reply with FAST or SLOW, then a space, then a complexity digit 1-9 (1=trivial, 9=very complex). Example: 'FAST 3'. Classify: {content}"}], max_tokens=3, **params)
        answer = result.choices[0].message.content.strip().upper()
        match = __import__("re").match(r"^(FAST|SLOW)\s*(\d)?", answer)
        return answer if match else "FAST"
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
        model = pricing.select_closest(pricing.pool_ids(config), result.complexity, config, pseudo_model=pseudo_model) if path == "fast" else None
        if model:
            from .stats import record_selection
            record_selection(result.complexity, pricing.target_scaled_cost(result.complexity, pseudo_model, config), model, config)
        return RoutingDecision(path=path, confidence_band="ambiguous", confidence=result.confidence, complexity=result.complexity, reason=f"single-model, router-resolved: {path}", model=model)
    if result.confidence_band == "ambiguous":
        try:
            answer = str((tiebreaker or _default_tiebreaker)(messages[-1], pseudo_model, config)).upper()
            import re
            match = re.match(r"^(FAST|SLOW)\s*(\d)?", answer)
            digit = int(match.group(2)) if match and match.group(2) else None
            complexity = .5 * result.complexity + .5 * (digit / 9) if digit is not None else result.complexity
            path = "slow" if match and match.group(1) == "SLOW" else "fast"
        except Exception:
            path, complexity = "fast", result.complexity
        model = pricing.select_closest(pricing.pool_ids(config), complexity, config, pseudo_model=pseudo_model) if path == "fast" else None
        if model:
            from .stats import record_selection
            record_selection(complexity, pricing.target_scaled_cost(complexity, pseudo_model, config), model, config)
        return RoutingDecision(path=path, confidence_band="ambiguous", confidence=result.confidence, complexity=complexity, reason="tiebreaker: " + path, model=model)
    model = pricing.select_closest(pricing.pool_ids(config), result.complexity, config, pseudo_model=pseudo_model) if result.path == "fast" else None
    if model:
        from .stats import record_selection
        record_selection(result.complexity, pricing.target_scaled_cost(result.complexity, pseudo_model, config), model, config)
    return RoutingDecision(path=result.path, confidence_band=result.confidence_band, confidence=result.confidence, complexity=result.complexity, reason=result.reason, model=model)
