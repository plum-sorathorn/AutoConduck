from dataclasses import dataclass
from typing import Literal
import re
import math
from .semantic_router import RouteMatch

STACK_TRACE_BOOST = 0.25
ESCALATION_THRESHOLD = 0.80
HYSTERESIS_FLOOR = 0.50

# ---------------------------------------------------------------------------
# Pre-compiled regex patterns — all compiled once at import, zero per-request
# regex compilation cost. Target: <0.5ms total per call on a 2KB prompt.
# ---------------------------------------------------------------------------

_SYSTEM_REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>", re.IGNORECASE | re.DOTALL)

# ── Layer 1: Surface signals ────────────────────────────────────────────────

# Structural formatting markers (bullets, numbered items, code fences, headers)
# NOTE: does NOT include keyword hits — that was the double-counting bug.
_STRUCTURAL_PATTERN = re.compile(
    r"(?m)^\s*(?:[-*]|\\d+[.)])\s+|```|^\s*##\s+|"
    r"\b(?:\[Context\]|\[Skills\]|\[Prompts\]|\[Extensions\]|\[Themes\])\b|"
    r"\b(?:refactor|migrate|redesign|architecture|overhaul|workflow|pipeline|orchestrat|subagent|multi-agent|end-to-end|e2e|entire|whole|all files|multiple files|integration|codebase)\b",
    re.IGNORECASE,
)

# Cross-references: @mentions, #issues, URLs (external coordination density)
_CROSS_REFS_PATTERN = re.compile(r"@\w+|#\d+|https?://\S+")

# Code density: backtick spans, code fences, CLI flags, env vars
_CODE_DENSITY_PATTERN = re.compile(r"`[^`]*`|```[\s\S]*?```|\b(?:--[a-z0-9_-]+|-\w)\b|\$[A-Z0-9_]+|%[A-Z0-9_]+%")

# Scope breadth: filenames, CamelCase names (classes/components), path-like refs,
# totality markers. Uses a set to count *distinct* entities for a breadth signal.
_SCOPE_ENTITIES_PATTERN = re.compile(
    r"[\w./\\-]+\.\w{1,4}\b|"          # filenames
    r"\b[A-Z][a-zA-Z]{2,}\w*\b|"       # CamelCase identifiers (class/component names)
    r"/[\w/]{3,}\b|"                    # path-like refs
    r"\b(?:all|every|entire|whole|each|across|throughout)\b",  # totality markers
    re.IGNORECASE,
)

# Numbered list items
_NUMBERED_LIST = re.compile(r"(?m)^\s*\d+[.)]\s+")

# Multi-step transition markers
_TRANSITION_MARKERS = re.compile(
    r"\b(?:first|second|third|then|next|after that|also|finally|additionally|furthermore|moreover|step \d+|item \d+)\b",
    re.IGNORECASE,
)

# ── Layer 2: Domain-agnostic semantic signals ───────────────────────────────

# 2a. Abstraction level — abstract concepts vs concrete one-off operations
_ABSTRACT_SIGNALS = re.compile(
    r"\b(?:design|architect|strategy|approach|paradigm|pattern|framework|structure|"
    r"system|pipeline|workflow|scheme|protocol|model|abstraction|concept|principle|"
    r"methodology|blueprint|topology|hierarchy|taxonomy|ontology|"
    r"re-?think|re-?imagine|re-?design|re-?structure|re-?organize)\b",
    re.IGNORECASE,
)
_CONCRETE_SIGNALS = re.compile(
    r"\b(?:line|character|word|typo|comma|bracket|quote|space|tab|indent|"
    r"variable name|function name|rename|move|copy|delete|print|echo|log|"
    r"one.?liner|single.?line)\b",
    re.IGNORECASE,
)

# 2b. Uncertainty hedge — discovery/diagnostic work requires model capability
_UNCERTAINTY_SIGNALS = re.compile(
    r"\b(?:investigate|diagnose|debug|find out|figure out|explore|understand why|"
    r"not sure|unclear|seems like|might be|could be|possibly|perhaps|"
    r"why (?:is|does|did|would|are|isn.t|doesn.t|can.t)|"
    r"how (?:does|do|did|would|can|should)|"
    r"what (?:causes|is wrong|happened|broke|went wrong)|"
    r"root cause|trace back|trace the|where does|when does)\b",
    re.IGNORECASE,
)

# 2c. Cross-domain complexity — hard signals across 5 domains, easy signals universal
_DOMAIN_HARD_PATTERN = re.compile(
    r"\b(?:"
    # Software / Systems
    r"race condition|concurrency|async|deadlock|memory leak|profil|benchmark|"
    r"performance|distributed|security|auth|permission|schema|migration|"
    r"optimization|profiling|packaging|dependency|build system|compilation|"
    r"latency|throughput|cache|eviction|sharding|partitioning|consensus|replication|"
    # Data / ML
    r"aggregation|normalization|etl|transform|feature engineer|"
    r"model train|hyperparameter|dimensionality|embedding|clustering|"
    r"cross.?validation|overfitting|regularization|gradient|backpropagation|"
    r"data pipeline|inference|fine.?tun|pre.?train|"
    # Writing / Rhetoric
    r"narrative arc|thesis|argument structure|tone shift|voice consistency|"
    r"rhetorical|persuasive|academic|synthesis|coherence|discourse|"
    # Formal Reasoning / Math
    r"prove|derive|theorem|lemma|infer|deduce|formal proof|axiom|"
    r"logical consequence|contrapositive|mathematical induction|"
    r"complexity class|big.?o|np.?hard|"
    # Ops / Infrastructure — explicit ops keywords that don't overlap with easy terms
    r"high.?availability|failover|load balanc|orchestrat|"
    r"provision|infrastructure.as.code|terraform|kubernetes|helm|"
    r"deploy|rollback|capacity plan|sla|incident response|"
    r"container|docker|replicate|sharding|horizontal scal"
    r")\b",
    re.IGNORECASE,
)
_DOMAIN_EASY_PATTERN = re.compile(
    r"\b(?:"
    r"typo|rename|format|comment|lint|simple|quick|small|one.?line|"
    r"fix error|where is|what is|explain|docstring|update text|print|"
    r"show me|list|display|echo|check if|verify|confirm|does it"
    r")\b",
    re.IGNORECASE,
)

# 2d. Task novelty — creating new things vs operating on existing ones
_HIGH_NOVELTY = re.compile(
    r"\b(?:from scratch|brand.?new|ground up|"
    r"new (?:system|design|approach|feature|module|service|component|"
    r"algorithm|protocol|interface|schema|architecture|library|tool|endpoint)|"
    r"create|invent|devise|propose|prototype|proof of concept|"
    r"novel|innovative|custom|bespoke|greenfield|spin up|bootstrap)\b",
    re.IGNORECASE,
)
_LOW_NOVELTY = re.compile(
    r"\b(?:existing|current|already|standard|conventional|usual|typical|"
    r"same as before|as before|like (?:we|you|i) did|copy|duplicate|"
    r"based on|following|per the|using the existing)\b",
    re.IGNORECASE,
)

# 2e. Imperative strength bands — ordered from highest to lowest strength
# Each entry: (score, compiled_regex)
_IMPERATIVE_BANDS = [
    (1.00, re.compile(
        r"\b(?:completely rewrite|overhaul|rebuild|from the ground up|full rewrite|"
        r"tear down|blow up|rethink|ground.?up redesign)\b",
        re.IGNORECASE,
    )),
    (0.85, re.compile(
        r"\b(?:implement|build|create|write|develop|architect|design|engineer|"
        r"construct|generate|produce|draft)\b",
        re.IGNORECASE,
    )),
    (0.60, re.compile(
        r"\b(?:fix|add|update|change|modify|improve|extend|refactor|migrate|"
        r"adjust|tweak|patch|correct|resolve|address|handle)\b",
        re.IGNORECASE,
    )),
    (0.15, re.compile(
        r"\b(?:explain|what|why|describe|review|show|list|summarize|check|"
        r"find|inspect|explore|look|tell me|where is|how does)\b",
        re.IGNORECASE,
    )),
]
_NEUTRAL_IMPERATIVE = 0.40  # fallback when no verb detected

# ── Escalation & stack trace ────────────────────────────────────────────────

# Expanded stack trace detection — covers Python, Java, Go, Rust, C/C++,
# Ruby, generic compiler/runtime errors, and system signals.
_STACK_TRACE_PATTERNS = re.compile(
    r"Traceback \(most recent call last\)|"              # Python
    r'File "[^"]+", line \d+|'                          # Python frame
    r"at [\w.$]+\([\w.]+:\d+\)|"                        # Java/Kotlin
    r"Exception in thread [\"']?[\w.]+[\"']?|"          # Java
    r"Caused by: [\w.]+Exception|"                      # Java chained
    r"goroutine \d+ \[|"                                # Go goroutine header
    r"\bpanic: |"                                       # Go/Rust panic keyword
    r"thread '[^']+' panicked at|"                      # Rust thread panic
    r"\bSegmentation fault\b|\bBus error\b|"            # C/C++ signals
    r"\bAbort trap\b|\bIllegal instruction\b|"          # C/C++ signals
    r"error: (?:linker|undefined reference to|multiple definition|"
    r"use of undeclared|expected|cannot find)|"         # C/C++ compiler errors
    r"\b\w+(?:\.\w+)*:\d+:\d+:\s*(?:error|warning|fatal error):|"  # gcc/clang format
    r"\b(?:RubyError|NoMethodError|NameError):|"        # Ruby
    r"\b(?:Segfault|SIGSEGV|SIGABRT|SIGILL|SIGFPE)\b|" # POSIX signals
    r"\b(?:fatal error|unhandled exception|build failed|"
    r"command failed|exit code [1-9]\d*|"
    r"core dumped|killed by signal|assertion failed)\b",
    re.IGNORECASE,
)

# Expanded escalation signal detection — explicit directives + natural agent language.
# Written as a flat list of alternations to avoid regex parsing pitfalls with
# multi-line string continuation.
_ESCALATION_SIGNAL_PATTERN = re.compile(
    "|".join([
        # Explicit AutoConduck directives
        r"\bautoconduck:\s*escalate\b",
        r"\[escalate\]",
        r"\bescalate:\s*slow\b",
        r"\btask_too_complex\b",
        r"\bneeds_planning\b",
        r"\bescalate_routing\b",
        r"<autoconduck-escalate>",
        # Natural agent self-assessment: scope/complexity has grown
        r"\b(?:scope|complexity|requirements|task)\s+(?:has|have)\s+(?:become|grown|expanded|increased|gotten)\s+(?:\S+\s+)*(?:more\s+)?(?:complex|complicated|difficult|involved|intricate|large|broad)\b",
        # this is more complex
        r"\bthis\s+(?:is|has become)\s+(?:\S+\s+)*(?:complex|complicated|difficult|involved)\b",
        # I need to rethink
        r"\b(?:I|we)\s+(?:need\s+to|should|must)\s+(?:rethink|reconsider|reassess|replan|reevaluate)\b",
        # I underestimated
        r"\b(?:I|we)\s+(?:underestimated|misjudged|miscalculated)\s+(?:the|this)\b",
        # more planning is needed
        r"\b(?:more|additional|further)\s+(?:planning|analysis|investigation|decomposition)(?:\s+and\s+\w+)?\s+(?:is|are)\s+(?:needed|required|necessary)\b",
        # turns out to be more complex
        r"\b(?:this|it)\s+turns?\s+out\s+to\s+be\s+(?:more|harder|larger|bigger|more complex|more involved)\b",
    ]),
    re.IGNORECASE,
)


def clean_routing_text(text: object) -> str:
    """Remove Claude Code's injected reminders before measuring user intent."""
    return _SYSTEM_REMINDER.sub("", str(text or "")).strip()


@dataclass(frozen=True)
class Score:
    confidence_band: Literal["fast", "slow", "ambiguous"]
    path: Literal["fast", "slow"]
    confidence: float
    complexity: float
    reason: str


def has_stack_trace(text: str) -> bool:
    return bool(_STACK_TRACE_PATTERNS.search(str(text or "")))


def has_escalation_signal(text: str) -> bool:
    """Check if the text contains an explicit or natural-language escalation directive."""
    return bool(_ESCALATION_SIGNAL_PATTERN.search(str(text or "")))


def _tokens(text: str) -> set[str]:
    """Tokenise text into a lowercase word set (shared with semantic_router)."""
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def _intent_drift(messages: list) -> float:
    """Jaccard distance between first and last user messages.

    Returns 0.0 when there is only one user turn (no drift possible).
    High value → conversation has moved far from original intent → scope expansion.
    """
    user_msgs = [
        m for m in messages
        if isinstance(m, dict) and m.get("role") == "user"
        and "<system-reminder>" not in str(m.get("content", ""))
    ]
    if len(user_msgs) < 2:
        return 0.0
    first = _tokens(str(user_msgs[0].get("content", "")))
    last = _tokens(str(user_msgs[-1].get("content", "")))
    if not first or not last:
        return 0.0
    union = first | last
    jaccard = len(first & last) / len(union)
    return 1.0 - jaccard


def complexity_of(text: str, config=None) -> float:
    """Compute a complexity score ∈ [0, 1] for a prompt string.

    Four-layer scoring:
      Layer 1 — Surface signals (length, structural, scope, code density)
      Layer 2 — Domain-agnostic semantic signals (abstraction, uncertainty,
                 cross-domain difficulty, novelty, imperative strength, multi-step)
      Layer 3 — Additive boosts (stack trace, escalation)
      Weight calibration from config.selection.complexity_weights.

    Context-aware signals (conversation depth, tool chain length, intent drift)
    are applied in score() as additive soft boosts so that complexity_of()
    remains pure-text and independently testable.
    """
    t = str(text or "")

    # ── Layer 1: Surface signals ──────────────────────────────────────────

    # length: soft log-curve so 400-char and 1200-char prompts are distinguishable
    # log(1+n/400)/log(4) ≈ 0 at n=0, 0.5 at n=600, 1.0 at n≈1600
    length = min(1.0, math.log1p(len(t) / 400) / math.log(4))

    # structural: formatting markers only — no keyword hits (fixes double-count bug)
    structural = min(1.0, len(_STRUCTURAL_PATTERN.findall(t)) / 3)

    # scope_breadth: distinct named entities (filenames, CamelCase, paths, totality words)
    scope_breadth = min(1.0, len(set(_SCOPE_ENTITIES_PATTERN.findall(t))) / 6)

    # code_density: inline/block code, CLI flags, env vars (split out from old refs)
    code_density = min(1.0, len(_CODE_DENSITY_PATTERN.findall(t)) / 4)

    # ── Layer 2: Semantic signals ─────────────────────────────────────────

    # 2a. abstraction_level: abstract concepts vs concrete micro-operations
    abstract_hits = len(_ABSTRACT_SIGNALS.findall(t))
    concrete_hits = len(_CONCRETE_SIGNALS.findall(t))
    abstraction_level = (max(-4, min(4, abstract_hits - concrete_hits)) + 4) / 8

    # 2b. uncertainty_hedge: discovery/diagnostic work
    uncertainty_hedge = min(1.0, len(_UNCERTAINTY_SIGNALS.findall(t)) / 3)

    # 2c. cross_domain_complexity: multi-domain hard/easy keyword balance
    hard_count = len(_DOMAIN_HARD_PATTERN.findall(t))
    easy_count = len(_DOMAIN_EASY_PATTERN.findall(t))
    cross_domain = (max(-3, min(3, hard_count - easy_count)) + 3) / 6

    # 2d. task_novelty: building new vs operating on existing
    high_novelty = len(_HIGH_NOVELTY.findall(t))
    low_novelty = len(_LOW_NOVELTY.findall(t))
    task_novelty = min(1.0, max(0.0, (high_novelty - low_novelty) / 3 + 0.5))

    # 2e. imperative_strength: graded action intensity (replaces binary edit_intent)
    imperative_strength = _NEUTRAL_IMPERATIVE
    for score_val, pattern in _IMPERATIVE_BANDS:
        if pattern.search(t):
            imperative_strength = score_val
            break  # highest-priority band wins first match

    # 2f. multi_step: sequential complexity via transition markers + numbered items
    numbered = len(_NUMBERED_LIST.findall(t))
    markers = len(_TRANSITION_MARKERS.findall(t)) + max(0, numbered - 1)
    # Normalize over 5 (not 3) so 10-step workflows aren't capped at 3
    multi_step = min(1.0, markers / 5)

    # ── Weight application ────────────────────────────────────────────────

    weights = getattr(
        getattr(config, "selection", None), "complexity_weights", None
    ) or {
        "length":             0.08,
        "structural":         0.12,
        "scope_breadth":      0.12,
        "code_density":       0.05,
        "abstraction_level":  0.12,
        "uncertainty_hedge":  0.08,
        "cross_domain":       0.12,
        "task_novelty":       0.08,
        "imperative_strength":0.15,
        "multi_step":         0.08,
    }

    factor_values = {
        "length":             length,
        "structural":         structural,
        "scope_breadth":      scope_breadth,
        "code_density":       code_density,
        "abstraction_level":  abstraction_level,
        "uncertainty_hedge":  uncertainty_hedge,
        "cross_domain":       cross_domain,
        "task_novelty":       task_novelty,
        "imperative_strength":imperative_strength,
        "multi_step":         multi_step,
    }

    # Sum only the factors present in the weight dict (forward-compat: unknown
    # factors in config are silently ignored; missing factors default to 0 weight).
    value = sum(weights.get(k, 0.0) * v for k, v in factor_values.items())

    # ── Layer 3: Additive boosts ──────────────────────────────────────────
    trace_boost = STACK_TRACE_BOOST if has_stack_trace(t) else 0.0
    escalation_boost = 0.30 if has_escalation_signal(t) else 0.0

    return min(1.0, value + trace_boost + escalation_boost)


def is_tool_loop(messages: list) -> bool:
    """Return True if the message sequence is an in-flight tool loop turn.

    A turn is an active tool loop if the latest non-system message is a tool
    call or tool result — the agent is mid-execution and re-routing would break
    the tool contract.

    ESCALATION EXCEPTIONS (return False, allow full scoring):
      - Explicit escalation signal in the tool result.
      - Stack trace / fatal error in the tool result.
      - Tool chain length > 12 (long-running tool loops may need more capability).
    """
    if not isinstance(messages, list) or not messages:
        return False

    last_msg = None
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            last_msg = msg
            break
        role = msg.get("role", "user")
        content = str(msg.get("content", ""))
        if role != "system" and "<system-reminder>" not in content:
            last_msg = msg
            break

    if not last_msg or not isinstance(last_msg, dict):
        return False

    role = last_msg.get("role", "user")
    is_active_tool = (
        role in ("tool", "function")
        or "tool_calls" in last_msg
        or "function_call" in last_msg
    )
    if not is_active_tool:
        content = last_msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") in (
                    "tool_use",
                    "tool_result",
                ):
                    is_active_tool = True
                    break

    if not is_active_tool:
        return False

    # Check for in-flight escalation exceptions
    last_text = str(last_msg.get("content", ""))
    if has_escalation_signal(last_text) or has_stack_trace(last_text):
        return False

    # Long tool chain soft-escalation: if >12 tool turns have fired, allow re-scoring
    tool_turn_count = sum(
        1 for m in messages
        if isinstance(m, dict) and (
            m.get("role") in ("tool", "function") or "tool_calls" in m
        )
    )
    if tool_turn_count > 12:
        return False

    return True


def _last(messages: list) -> str:
    if not isinstance(messages, list) or not messages:
        return ""
    for item in reversed(messages):
        if not isinstance(item, dict) or item.get("role", "user") == "user":
            content = (
                item.get("content", "")
                if isinstance(item, dict)
                else getattr(item, "content", item)
            )
            if "<system-reminder>" not in str(content):
                return str(content or "")
            return clean_routing_text(content)
    return ""


def _routing_text(messages: list) -> str:
    user_text = _last(messages)
    tool_text = ""
    if isinstance(messages, list) and messages:
        last = messages[-1]
        if isinstance(last, dict) and last.get("role") in ("tool", "function"):
            tool_text = str(last.get("content", ""))
    return f"{user_text}\n{tool_text}".strip()


def _context_boost(messages: list) -> float:
    """Compute an additive context-aware complexity boost ∈ [0, 0.20].

    Three sub-signals, all O(m) over messages (typically <20):
      conversation_depth  — how many user turns have accumulated
      tool_chain_length   — how many tool calls/results have fired
      intent_drift        — Jaccard distance between first and last user message

    The boost is capped at 0.20 so it cannot dominate the base complexity score.
    It is applied in score() *after* complexity_of() so complexity_of() stays
    pure-text and independently testable.
    """
    if not isinstance(messages, list) or not messages:
        return 0.0

    user_turn_count = sum(
        1 for m in messages
        if isinstance(m, dict) and m.get("role") == "user"
        and "<system-reminder>" not in str(m.get("content", ""))
    )
    # 0.0 on first turn, 1.0 at 10+ turns
    conversation_depth = min(1.0, max(0.0, (user_turn_count - 1) / 10))

    tool_call_count = sum(
        1 for m in messages
        if isinstance(m, dict) and (
            m.get("role") in ("tool", "function") or "tool_calls" in m
        )
    )
    # 0.0 at 0 calls, 1.0 at 8+ calls
    tool_chain_length = min(1.0, tool_call_count / 8)

    drift = _intent_drift(messages)

    boost = 0.08 * conversation_depth + 0.08 * tool_chain_length + 0.04 * drift
    return min(0.20, boost)


def score(
    messages: list,
    history,
    match: RouteMatch,
    pseudo_model: str = "autoconduck",
    config=None,
) -> Score:
    cfg = config
    low = float(getattr(cfg, "ambiguous_low", 0.55) if cfg else 0.55)
    high = float(getattr(cfg, "ambiguous_high", 0.70) if cfg else 0.70)
    stack_trace_boost = float(
        getattr(cfg, "stack_trace_boost", STACK_TRACE_BOOST) if cfg else STACK_TRACE_BOOST
    )
    hysteresis_floor = float(
        getattr(cfg, "hysteresis_floor", HYSTERESIS_FLOOR) if cfg else HYSTERESIS_FLOOR
    )

    text = _routing_text(messages)
    complexity = complexity_of(text, cfg)
    trace = has_stack_trace(text)
    escalation = has_escalation_signal(text)

    # Active tool loops stay on the fast path UNLESS an escalation or stack trace
    # trigger fired (handled inside is_tool_loop) or the tool chain is very long.
    if is_tool_loop(messages):
        # Apply context boost to complexity even on the forced fast path so that
        # the persisted complexity value reflects accumulated session difficulty.
        ctx = _context_boost(messages)
        return Score("fast", "fast", 0.0, min(1.0, complexity + ctx), "interactive agent tool loop")

    # Apply context-aware boost (conversation depth, tool chain, intent drift)
    ctx = _context_boost(messages)
    complexity = min(1.0, complexity + ctx)

    confidence = min(
        1.0,
        max(float(match.confidence), complexity * 0.75)
        + (stack_trace_boost if trace else 0)
        + (0.30 if escalation else 0),
    )

    if trace or escalation:
        return Score(
            "slow",
            "slow",
            confidence,
            max(complexity, 0.85 if escalation else complexity),
            "agent complexity escalation" if escalation else "stack trace boost",
        )

    previous = history[-1] if isinstance(history, list) and history else history
    escalated = bool(
        getattr(previous, "complexity", 0) >= ESCALATION_THRESHOLD
        or (
            isinstance(previous, dict)
            and (
                previous.get("complexity", 0) >= ESCALATION_THRESHOLD
                or previous.get("confidence", 0) >= ESCALATION_THRESHOLD
            )
        )
    )
    if escalated:
        complexity = min(complexity, hysteresis_floor)

    multiplier = 1.0
    if pseudo_model.endswith("budget"):
        multiplier = 1.15
    elif pseudo_model.endswith("expensive"):
        multiplier = 0.85

    boundary_low, boundary_high = min(1.0, low * multiplier), min(
        1.0, high * multiplier
    )

    if confidence < boundary_low or (boundary_low <= confidence <= boundary_high):
        return Score(
            "ambiguous",
            "fast",
            confidence,
            complexity,
            "confidence is in the ambiguous zone",
        )

    sel = getattr(cfg, "selection", cfg)
    slow_threshold = float(getattr(sel, "slow_threshold", 0.75) if sel else 0.75)
    slow = complexity >= slow_threshold or (
        match.route == "slow_path" and confidence >= boundary_high
    )

    return Score(
        "slow" if slow else "fast",
        "slow" if slow else "fast",
        confidence,
        complexity,
        "semantic route and complexity",
    )
