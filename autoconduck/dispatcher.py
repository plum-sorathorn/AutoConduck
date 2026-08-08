from dataclasses import dataclass
from typing import Literal
from . import semantic_router, evaluator

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
        content = getattr(message, "content", str(message))
        result = litellm.completion(model="gpt-5-mini", messages=[{"role":"user", "content": f"Reply only FAST or SLOW. Classify: {content}"}], max_tokens=2)
        answer = result.choices[0].message.content.strip().upper()
        return "slow" if "SLOW" in answer else "fast"
    except Exception:
        return "fast"

def route(messages: list, history, pseudo_model: str = "autoconduck", tiebreaker=None, config=None) -> RoutingDecision:
    text = getattr(messages[-1], "content", str(messages[-1]))
    result = evaluator.score(messages, history, semantic_router.route(text), pseudo_model, config)
    if result.confidence_band == "ambiguous":
        try: path = str((tiebreaker or _default_tiebreaker)(messages[-1], pseudo_model, config)).lower()
        except Exception: path = "fast"
        path = "slow" if path.startswith("slow") else "fast"
        return RoutingDecision(path, "ambiguous", result.confidence, result.complexity, None, "tiebreaker: " + path)
    return RoutingDecision(result.path, result.confidence_band, result.confidence, result.complexity, None, result.reason)
