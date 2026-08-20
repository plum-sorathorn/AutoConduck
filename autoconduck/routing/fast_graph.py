"""Fast-path routing execution pipeline and backwards-compatibility facade."""

from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class FastGraphState:
    messages: list
    history: Any
    pseudo_model: str = "autoconduck"
    config: Any = None
    tiebreaker: Any = None

    # Internal state
    raw_text: str = ""
    route_match: Any = None
    eval_score: Any = None
    path: Literal["fast", "slow"] = "fast"
    confidence_band: Literal["fast", "slow", "ambiguous"] = "fast"
    confidence: float = 0.0
    complexity: float = 0.0
    reason: str = ""
    model: str | None = None


class FastGraph:
    """Micro-pipeline facade for fast path routing.

    Delegates directly to dispatcher.route() for zero-overhead synchronous execution.
    """

    def execute(self, state: FastGraphState) -> FastGraphState:
        from .dispatcher import route

        decision = route(
            messages=state.messages,
            history=state.history,
            pseudo_model=state.pseudo_model,
            tiebreaker=state.tiebreaker,
            config=state.config,
        )
        state.path = decision.path
        state.confidence_band = decision.confidence_band
        state.confidence = decision.confidence
        state.complexity = decision.complexity
        state.reason = decision.reason
        state.model = decision.model
        return state

    def _node_model_select(self, state: Any) -> None:
        """Backwards-compatibility model selection helper for existing test suites."""
        from ..config import resolve_orchestrator_model
        from . import pricing

        if getattr(state, "path", "fast") == "fast":
            cfg = getattr(state, "config", None)
            comp = getattr(state, "complexity", 0.0)
            pseudo = getattr(state, "pseudo_model", "autoconduck")
            model = pricing.select_closest(
                pricing.pool_ids(cfg),
                comp,
                cfg,
                pseudo_model=pseudo,
            ) or resolve_orchestrator_model(cfg)
            state.model = model
        else:
            state.model = None


_DEFAULT_FAST_GRAPH = FastGraph()


def execute_fast_graph(state: FastGraphState) -> FastGraphState:
    return _DEFAULT_FAST_GRAPH.execute(state)

