from dataclasses import dataclass
from typing import Literal
import re
import math

RouteName = Literal["fast_path", "slow_path"]

# Cross-domain examples — covers software, data/ML, writing, ops, math/reasoning.
# Deliberately diverse to reduce token-overlap false-positives in the Jaccard fallback.
FAST_EXAMPLES = [
    # Software
    "fix this typo",
    "rename this function",
    "where is X defined",
    "add a docstring",
    "what does this line do",
    "update this comment",
    "delete this file",
    "fix the syntax error",
    "explain this function",
    # Data / ML
    "what does this column mean",
    "show me the first 10 rows",
    "explain what this metric measures",
    "rename this feature",
    # Writing
    "fix the grammar in this sentence",
    "correct this typo in the paragraph",
    "what does this word mean",
    # Ops
    "check if the service is running",
    "what is the current log level",
    "show the last 20 lines of the log",
    # Math / Reasoning
    "what is the formula for this",
    "explain this equation",
    "verify this calculation",
]

SLOW_EXAMPLES = [
    # Software
    "refactor the application",
    "review the backend",
    "build a feature",
    "implement multi-file change",
    "migrate the database",
    "redesign the API",
    "write integration tests for the whole system",
    "optimize the performance of the codebase",
    "fix the race condition across the async workers",
    "implement authentication and role-based permissions",
    "architect a distributed caching layer",
    # Data / ML
    "design a feature engineering pipeline",
    "train a neural network on the customer dataset",
    "build an end-to-end ETL workflow",
    "implement cross-validation and hyperparameter tuning",
    "set up model monitoring and drift detection",
    # Writing
    "rewrite the entire argument structure of the thesis",
    "redesign the narrative arc for the report",
    "restructure the essay with a new rhetorical approach",
    # Ops / Infrastructure
    "set up high-availability failover across regions",
    "provision the Kubernetes cluster with Helm charts",
    "design the incident response runbook",
    "implement infrastructure as code with Terraform",
    # Math / Reasoning
    "prove this theorem using mathematical induction",
    "derive the closed-form solution for this optimization problem",
    "formally verify this logical argument",
]


@dataclass(frozen=True)
class RouteMatch:
    route: RouteName
    confidence: float


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


# Pre-tokenize all static examples once at import (eliminates regex cost per request).
_TOKENIZED_EXAMPLES: dict[str, set[str]] = {
    e: _tokens(e) for e in FAST_EXAMPLES + SLOW_EXAMPLES
}

# Build a lightweight IDF table over the example corpus so common words (the,
# this, of) carry much less weight than rare discriminative words (hyperparameter,
# failover, theorem). This replaces pure Jaccard with a soft tf-idf-weighted
# overlap, giving much better signal for out-of-domain prompts.
def _build_idf(examples: list[str]) -> dict[str, float]:
    N = len(examples)
    df: dict[str, int] = {}
    for ex in examples:
        for tok in _TOKENIZED_EXAMPLES[ex]:
            df[tok] = df.get(tok, 0) + 1
    # idf(t) = log(1 + N / df(t)); +1 smoothing so unseen terms get log(1+N)
    return {tok: math.log(1 + N / count) for tok, count in df.items()}

_ALL_EXAMPLES = FAST_EXAMPLES + SLOW_EXAMPLES
_IDF: dict[str, float] = _build_idf(_ALL_EXAMPLES)
_MAX_IDF = math.log(1 + len(_ALL_EXAMPLES))  # used to normalise similarity


def _idf_weighted_similarity(query_tokens: set[str], example_tokens: set[str]) -> float:
    """Soft cosine-like similarity weighted by IDF; replaces raw Jaccard."""
    common = query_tokens & example_tokens
    if not common:
        return 0.0
    union = query_tokens | example_tokens
    # Weighted intersection over weighted union
    w_inter = sum(_IDF.get(t, _MAX_IDF) for t in common)
    w_union = sum(_IDF.get(t, _MAX_IDF) for t in union)
    return w_inter / w_union if w_union else 0.0


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
                self._layer = RouteLayer(
                    routes=[
                        Route(name="fast_path", utterances=FAST_EXAMPLES),
                        Route(name="slow_path", utterances=SLOW_EXAMPLES),
                    ],
                    encoder=encoder,
                )
        except Exception:
            self._layer = None

    def route(self, text: str) -> RouteMatch:
        text = str(text or "")
        if self._layer is not None:
            try:
                result = self._layer(text)
                name = getattr(result, "name", getattr(result, "route", "fast_path"))
                confidence = float(
                    getattr(result, "similarity_score", getattr(result, "confidence", 0.0))
                )
                return RouteMatch(
                    "slow_path" if name == "slow_path" else "fast_path",
                    max(0.0, min(1.0, confidence)),
                )
            except Exception:
                pass

        # IDF-weighted fallback (replaces plain Jaccard over fixed toy examples)
        words = _tokens(text)

        def best(examples: list[str]) -> float:
            return max(
                (_idf_weighted_similarity(words, _TOKENIZED_EXAMPLES[e]) for e in examples),
                default=0.0,
            )

        fast_score = best(FAST_EXAMPLES)
        slow_score = best(SLOW_EXAMPLES)
        if fast_score == slow_score == 0:
            return RouteMatch("fast_path", 0.0)
        return RouteMatch(
            "slow_path" if slow_score > fast_score else "fast_path",
            max(fast_score, slow_score),
        )


semantic_router = SemanticRouter()


def route(text: str) -> RouteMatch:
    return semantic_router.route(text)
