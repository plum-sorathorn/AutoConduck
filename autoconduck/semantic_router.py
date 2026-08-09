from dataclasses import dataclass
from typing import Literal
import re

RouteName = Literal["fast_path", "slow_path"]

FAST_EXAMPLES = ["fix this typo", "rename this function", "where is X defined", "add a docstring", "what does this line do", "update this comment", "delete this file", "fix the syntax error", "explain this function"]
SLOW_EXAMPLES = ["refactor the application", "review the backend", "build a feature", "implement multi-file change", "migrate the database", "redesign the API", "write integration tests for the whole system", "optimize the performance of the codebase"]

@dataclass(frozen=True)
class RouteMatch:
    route: RouteName
    confidence: float

def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


# Pre-tokenize all static examples once at import (~zero per-request cost, eliminates 16 regex calls/example batch).
_TOKENIZED_EXAMPLES = {e: _tokens(e) for e in FAST_EXAMPLES + SLOW_EXAMPLES}


class SemanticRouter:
    def __init__(self) -> None:
        self._layer = None
        try:
            from semantic_router import Route, RouteLayer
            try:
                from semantic_router.encoders import FastEmbedEncoder
                encoder = FastEmbedEncoder()
            except Exception:
                encoder = None
            if encoder is not None:
                self._layer = RouteLayer(routes=[Route(name="fast_path", utterances=FAST_EXAMPLES), Route(name="slow_path", utterances=SLOW_EXAMPLES)], encoder=encoder)
        except Exception:
            self._layer = None

    def route(self, text: str) -> RouteMatch:
        text = str(text or "")
        if self._layer is not None:
            try:
                result = self._layer(text)
                name = getattr(result, "name", getattr(result, "route", "fast_path"))
                confidence = float(getattr(result, "similarity_score", getattr(result, "confidence", 0.0)))
                return RouteMatch("slow_path" if name == "slow_path" else "fast_path", max(0.0, min(1.0, confidence)))
            except Exception:
                pass
        words = _tokens(text)
        def best(examples: list[str]) -> float:
            # Use pre-tokenized sets so we never re-run regex on example strings.
            return max((len(words & _TOKENIZED_EXAMPLES[e]) / max(1, len(words | _TOKENIZED_EXAMPLES[e])) for e in examples), default=0.0)
        fast, slow = best(FAST_EXAMPLES), best(SLOW_EXAMPLES)
        if fast == slow == 0:
            return RouteMatch("fast_path", 0.0)
        return RouteMatch("slow_path" if slow > fast else "fast_path", max(fast, slow))

semantic_router = SemanticRouter()
def route(text: str) -> RouteMatch:
    return semantic_router.route(text)
