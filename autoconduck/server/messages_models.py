"""Model identifiers exposed by the compatibility surface."""

# Generic pseudo model names used by most agents (Pi, Claude Code, OpenCode)
_GENERIC_PSEUDO_MODELS = {"autoconduck", "autoconduck-budget", "autoconduck-expensive"}

# Variant suffixes recognised for the autoconduck namespace.
# OMP registers four variants; we accept them all so autoconduck/fast,
# autoconduck/balanced, autoconduck/frontier and autoconduck/smart-dag
# are routed through the AutoConduck dispatcher rather than passed verbatim
# to an upstream LLM that doesn't know those IDs.
_AUTOCONDUCK_VARIANTS = {"fast", "balanced", "frontier", "smart-dag"}


def _build_pseudo_models():
    """Return the union of generic pseudo-models plus any autoconduck/<variant> and variant names."""
    variants = set(_AUTOCONDUCK_VARIANTS)
    prefixed_slash = {f"autoconduck/{v}" for v in variants}
    prefixed_dash = {f"autoconduck-{v}" for v in variants}
    prefixed_space = {f"autoconduck {v}" for v in variants}
    return (
        set(_GENERIC_PSEUDO_MODELS)
        | variants
        | prefixed_slash
        | prefixed_dash
        | prefixed_space
    )


PSEUDO_MODELS: frozenset[str] = frozenset(_build_pseudo_models())
