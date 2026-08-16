"""Zero-overhead compiled mini graph for fast-path routing execution."""

from dataclasses import dataclass
from typing import Any, Callable, Literal

from . import semantic_router, evaluator, pricing


@dataclass
class FastGraphState:
    messages: list
    history: Any
    pseudo_model: str = "autoconduck"
    config: Any = None
    tiebreaker: Any = None

    # Internal node state
    raw_text: str = ""
    route_match: Any = None
    eval_score: Any = None
    path: Literal["fast", "slow"] = "fast"
    confidence_band: Literal["fast", "slow", "ambiguous"] = "fast"
    confidence: float = 0.0
    complexity: float = 0.0
    reason: str = ""
    model: str | None = None


class FastGraphNode:
    """Synchronous fast graph node execution primitive."""

    def __init__(self, name: str, action: Callable[[FastGraphState], None]):
        self.name = name
        self.action = action

    def __call__(self, state: FastGraphState) -> None:
        self.action(state)


class FastGraph:
    """Micro-DAG execution graph for fast path routing.

    Executes in under 0.1 ms with zero async overhead or dynamic reflection.
    Nodes:
      1. input_sanitize: Extract user text and clean routing input
      2. route_match: Match against Aurelio semantic router
      3. evaluate_score: Score complexity, check stack trace/escalation/de-escalation
      4. tiebreaker_resolve: Resolve ambiguous zone if needed
      5. model_select: Dynamic closest-cost model selection (enforces hard limits)
    """

    def __init__(self):
        self._nodes = [
            FastGraphNode("input_sanitize", self._node_input_sanitize),
            FastGraphNode("route_match", self._node_route_match),
            FastGraphNode("evaluate_score", self._node_evaluate_score),
            FastGraphNode("tiebreaker_resolve", self._node_tiebreaker_resolve),
            FastGraphNode("model_select", self._node_model_select),
        ]

    def execute(self, state: FastGraphState) -> FastGraphState:
        for node in self._nodes:
            node(state)
        return state

    def _node_input_sanitize(self, state: FastGraphState) -> None:
        user_msgs = [
            m
            for m in state.messages
            if not isinstance(m, dict) or m.get("role", "user") == "user"
        ]
        last = user_msgs[-1] if user_msgs else ""
        text = (
            last.get("content", "")
            if isinstance(last, dict)
            else getattr(last, "content", str(last))
        )
        state.raw_text = evaluator.clean_routing_text(text)

    def _node_route_match(self, state: FastGraphState) -> None:
        state.route_match = semantic_router.route(state.raw_text)

    def _node_evaluate_score(self, state: FastGraphState) -> None:
        score = evaluator.score(
            state.messages,
            state.history,
            state.route_match,
            state.pseudo_model,
            state.config,
        )
        state.eval_score = score
        state.path = getattr(score, "path", "fast")
        state.confidence_band = getattr(score, "confidence_band", "fast")
        state.confidence = getattr(score, "confidence", 0.0)
        state.complexity = getattr(score, "complexity", 0.0)
        state.reason = getattr(score, "reason", "")

    def _node_tiebreaker_resolve(self, state: FastGraphState) -> None:
        if state.confidence_band != "ambiguous":
            return

        cfg = state.config
        enabled = [
            entry
            for entry in (getattr(cfg, "model_list", []) or [])
            if entry.get("enabled", True)
        ]
        if len(enabled) <= 1:
            state.path = "slow" if state.route_match.route == "slow_path" else "fast"
            state.reason = f"single-model, router-resolved: {state.path}"
            return

        selection = getattr(cfg, "selection", cfg)
        tiebreaker_floor = float(getattr(selection, "tiebreaker_min_complexity", 0.45))
        if state.pseudo_model.endswith("budget"):
            tiebreaker_floor = float(
                getattr(selection, "budget_tiebreaker_min_complexity", 0.65)
            )
        use_tiebreaker = state.tiebreaker is not None or (
            bool(getattr(selection, "tiebreaker_enabled", True))
            and state.complexity >= tiebreaker_floor
        )

        if not use_tiebreaker:
            state.path = "fast"
            state.reason = "tiebreaker: fast (below-floor)"
            return

        from .dispatcher import _default_tiebreaker

        try:
            tb_fn = state.tiebreaker or _default_tiebreaker
            answer = str(tb_fn(state.messages[-1], state.pseudo_model, cfg)).upper()
        except Exception:
            answer = "NONE"

        import re

        m = re.match(r"^(FAST|SLOW)(?:\s+([1-9]))?\s*$", answer)
        if answer == "NONE" or not m:
            slow_threshold = float(getattr(selection, "slow_threshold", 0.75))
            tiebreaker_model = pricing.cheapest_enabled(cfg)
            degraded = bool(
                tiebreaker_model
                and pricing.is_degraded(
                    tiebreaker_model,
                    getattr(cfg, "degraded_window_s", 300),
                    getattr(cfg, "degraded_error_rate", 0.2),
                )
            )
            if degraded:
                state.path = "fast"
                reason_suffix = "unavailable: degraded-provider"
            else:
                state.path = "slow" if state.complexity >= slow_threshold else "fast"
                reason_suffix = "unavailable: complexity-fallback"
            state.reason = f"tiebreaker_{reason_suffix}"
        else:
            digit = int(m.group(2)) if m.group(2) else None
            if digit is not None:
                state.complexity = 0.5 * state.complexity + 0.5 * (digit / 9)
            state.path = "slow" if m.group(1) == "SLOW" else "fast"
            state.reason = f"tiebreaker: {state.path}"

    def _node_model_select(self, state: FastGraphState) -> None:
        if state.path == "fast":
            from ..config import resolve_orchestrator_model

            selection = getattr(state.config, "selection", state.config)
            max_fast_cost = getattr(selection, "fast_path_max_scaled_cost", 0.50)
            try:
                max_fast_cost = float(max_fast_cost)
            except (TypeError, ValueError):
                max_fast_cost = 0.50

            model = pricing.select_closest(
                pricing.pool_ids(state.config),
                state.complexity,
                state.config,
                pseudo_model=state.pseudo_model,
                max_scaled_cost=max_fast_cost,
            ) or resolve_orchestrator_model(state.config)
            state.model = model
            if model:
                from ..stats import record_selection

                record_selection(
                    state.complexity,
                    pricing.target_scaled_cost(
                        state.complexity, state.pseudo_model, state.config
                    ),
                    model,
                    state.config,
                )
        else:
            state.model = None


_DEFAULT_FAST_GRAPH = FastGraph()


def execute_fast_graph(state: FastGraphState) -> FastGraphState:
    return _DEFAULT_FAST_GRAPH.execute(state)
