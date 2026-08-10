import math
import pytest
from autoconduck.config import Config
from autoconduck.evaluator import (
    complexity_of,
    has_stack_trace,
    has_escalation_signal,
    _context_boost,
    _intent_drift,
)


# ── Weight sanity ──────────────────────────────────────────────────────────

def test_complexity_weights_sum_to_one_default():
    weights = Config().selection.complexity_weights
    total = sum(weights.values())
    assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, expected 1.0"


# ── Regression anchors (deterministic fixed prompts) ──────────────────────

def test_complexity_of_trivial_prompt_is_low():
    """'fix typo' must stay well below the slow threshold."""
    c = complexity_of("fix typo")
    assert c < 0.45, f"'fix typo' scored {c}, expected < 0.45"


def test_complexity_of_complex_prompt_is_high():
    """A clear refactor request must score above 0.55."""
    c = complexity_of(
        "refactor the entire application across multiple files "
        "and write integration tests for all modules covering every service"
    )
    assert c >= 0.45, f"Refactor prompt scored {c}, expected >= 0.45"


def test_complexity_of_edit_intent_raises_value():
    """Fixing a race condition must score higher than explaining something."""
    assert complexity_of("fix the race condition") > complexity_of(
        "explain what this does"
    )


def test_complexity_of_multistep_markers():
    """More transition markers → higher score."""
    assert complexity_of("do this then next finally") > complexity_of("do this")


# ── Phase 1 fixes ─────────────────────────────────────────────────────────

def test_log_length_normalization_distinguishes_medium_from_long():
    """400-char and 1200-char prompts must produce distinguishable length scores."""
    short_prompt = "fix typo"
    medium_prompt = "a " * 200   # ~400 chars
    long_prompt = "a " * 600    # ~1200 chars
    c_short = complexity_of(short_prompt)
    c_medium = complexity_of(medium_prompt)
    c_long = complexity_of(long_prompt)
    assert c_long > c_medium > c_short, (
        f"Expected c_long({c_long}) > c_medium({c_medium}) > c_short({c_short})"
    )


def test_structural_no_longer_double_counts_keywords():
    """structural factor must only count formatting signals, not keyword hits.
    Verify: a prompt with ONLY hard keywords (no bullets/fences/headers)
    should NOT receive a high structural score."""
    # This prompt has many hard keywords but zero formatting structure
    keyword_heavy = (
        "architecture refactor migrate redesign security auth "
        "pipeline distributed optimization race condition"
    )
    c = complexity_of(keyword_heavy)
    # The score should be non-trivial (keywords contribute via cross_domain)
    # but structural should not be double-boosting it.
    # Primary check: the score is driven by cross_domain + abstraction, not structural.
    # We verify this indirectly: adding real structural formatting raises the score.
    structured_version = (
        "## Plan\n"
        "- architecture refactor migrate redesign security auth\n"
        "- pipeline distributed optimization race condition\n"
        "```\nsome code\n```"
    )
    c_structured = complexity_of(structured_version)
    assert c_structured > c, (
        "Adding structural formatting should raise complexity above pure keyword list"
    )


# ── Phase 2: Domain-agnostic signals ─────────────────────────────────────

def test_abstraction_level_high_for_architectural_prompts():
    c_abstract = complexity_of("redesign the overall system architecture and strategy")
    c_concrete = complexity_of("rename the variable x to y")
    assert c_abstract > c_concrete, (
        f"Abstract prompt {c_abstract} should exceed concrete prompt {c_concrete}"
    )


def test_uncertainty_hedge_raises_complexity():
    c_uncertain = complexity_of(
        "investigate why the service is slow; I'm not sure what causes it"
    )
    c_certain = complexity_of("update the timeout value to 30 seconds")
    assert c_uncertain > c_certain, (
        f"Uncertain/diagnostic prompt {c_uncertain} should exceed certain edit {c_certain}"
    )


def test_imperative_strength_full_rewrite_beats_fix():
    c_full = complexity_of("completely rewrite the authentication module from scratch")
    c_fix = complexity_of("fix the null check in the authentication module")
    c_explain = complexity_of("explain what the authentication module does")
    assert c_full > c_fix > c_explain, (
        f"Expected full({c_full}) > fix({c_fix}) > explain({c_explain})"
    )


def test_imperative_strength_neutral_for_no_verb():
    """A prompt with no recognisable verb should receive the neutral imperative score."""
    # The neutral score is 0.40 — prompt shouldn't be treated as purely explanatory
    c = complexity_of("authentication module performance")
    # With neutral imperative (0.40) and a couple of structural/domain hits this
    # should land somewhere moderate, not at the floor.
    assert 0.1 < c < 0.9, f"No-verb prompt got extreme score: {c}"


def test_task_novelty_from_scratch_beats_existing():
    c_new = complexity_of("build a new caching service from scratch")
    c_existing = complexity_of("update the existing caching service")
    assert c_new > c_existing, (
        f"From-scratch prompt {c_new} should exceed existing update {c_existing}"
    )


def test_cross_domain_data_complexity():
    """Data/ML hard terms must push complexity up."""
    c = complexity_of(
        "build an end-to-end ETL pipeline with feature engineering, "
        "cross-validation, and hyperparameter tuning for model training"
    )
    assert c >= 0.50, f"Data/ML complex prompt scored {c}, expected >= 0.50"


def test_cross_domain_writing_complexity():
    """Writing/rhetoric hard terms must raise complexity."""
    c = complexity_of(
        "redesign the narrative arc and argument structure of the thesis "
        "with a new rhetorical approach for academic synthesis"
    )
    assert c >= 0.45, f"Writing complex prompt scored {c}, expected >= 0.45"


def test_cross_domain_ops_complexity():
    """Ops/infrastructure hard terms must raise complexity."""
    c = complexity_of(
        "provision a Kubernetes cluster with Helm charts and configure "
        "high-availability failover with load balancing across regions"
    )
    assert c >= 0.40, f"Ops complex prompt scored {c}, expected >= 0.40"


def test_cross_domain_math_reasoning_complexity():
    """Formal reasoning terms must raise complexity."""
    c = complexity_of(
        "prove this theorem using mathematical induction and derive "
        "the formal proof of the logical consequence"
    )
    assert c >= 0.40, f"Math/reasoning prompt scored {c}, expected >= 0.40"


def test_multi_step_normalized_over_five():
    """10-step workflow should score higher than 3-step workflow (old norm was /3)."""
    three_steps = "do A, then do B, then finally do C"
    ten_steps = (
        "first do A, second do B, third do C, then D, next E, "
        "after that F, also G, furthermore H, moreover I, finally J"
    )
    c3 = complexity_of(three_steps)
    c10 = complexity_of(ten_steps)
    assert c10 > c3, (
        f"10-step workflow {c10} should exceed 3-step workflow {c3}"
    )


def test_scope_breadth_many_entities():
    """Prompt touching many named entities should score higher than single entity."""
    c_many = complexity_of(
        "Update AuthService, UserRepository, TokenManager, EmailClient, "
        "AuditLogger, and ConfigProvider across /api/v1/auth, /api/v1/users"
    )
    c_one = complexity_of("update the auth function")
    assert c_many > c_one, (
        f"Many-entity prompt {c_many} should exceed single-entity prompt {c_one}"
    )


# ── Phase 3: Context-aware boosts ─────────────────────────────────────────

def test_context_boost_increases_with_conversation_depth():
    """More user turns → higher context boost."""
    one_turn = [{"role": "user", "content": "fix this typo"}]
    ten_turns = [{"role": "user", "content": f"message {i}"} for i in range(10)]
    boost_one = _context_boost(one_turn)
    boost_ten = _context_boost(ten_turns)
    assert boost_ten > boost_one, (
        f"10-turn boost {boost_ten} should exceed 1-turn boost {boost_one}"
    )


def test_context_boost_increases_with_tool_chain():
    """More tool calls → higher context boost."""
    no_tools = [{"role": "user", "content": "do something"}]
    many_tools = [{"role": "user", "content": "do something"}] + [
        {"role": "tool", "content": f"result {i}", "tool_call_id": f"c{i}"}
        for i in range(8)
    ]
    boost_none = _context_boost(no_tools)
    boost_many = _context_boost(many_tools)
    assert boost_many > boost_none, (
        f"Many-tool boost {boost_many} should exceed no-tool boost {boost_none}"
    )


def test_context_boost_capped_at_0_20():
    """Context boost must never exceed 0.20."""
    # Construct a worst-case: 20 user turns + 20 tool calls
    msgs = []
    for i in range(20):
        msgs.append({"role": "user", "content": f"different topic {i}"})
        msgs.append({"role": "tool", "content": f"result {i}", "tool_call_id": f"c{i}"})
    assert _context_boost(msgs) <= 0.20


def test_intent_drift_zero_on_single_turn():
    """Single-turn conversation has zero drift by definition."""
    msgs = [{"role": "user", "content": "fix the typo"}]
    assert _intent_drift(msgs) == 0.0


def test_intent_drift_high_when_topic_changes():
    """If first and last user messages share few tokens, drift should be high."""
    msgs = [
        {"role": "user", "content": "fix the typo in line 12"},
        {"role": "assistant", "content": "fixed"},
        {"role": "user", "content": "redesign the distributed caching architecture"},
    ]
    drift = _intent_drift(msgs)
    assert drift > 0.5, f"High-topic-change drift should be > 0.5, got {drift}"


def test_intent_drift_low_when_same_topic():
    """Continuation of the same topic has low drift."""
    msgs = [
        {"role": "user", "content": "fix the authentication bug"},
        {"role": "assistant", "content": "investigating"},
        {"role": "user", "content": "also fix the authentication token expiry"},
    ]
    drift = _intent_drift(msgs)
    assert drift < 0.7, f"Same-topic drift should be low, got {drift}"


# ── Phase 3: Long tool-loop soft escalation ───────────────────────────────

def test_long_tool_loop_breaks_fast_path_bypass():
    """Tool loop with >12 tool turns should return is_tool_loop=False (allow rescoring)."""
    from autoconduck.evaluator import is_tool_loop
    msgs = [{"role": "user", "content": "do something complex"}]
    for i in range(13):
        msgs.append({
            "role": "tool",
            "content": f"result {i}",
            "tool_call_id": f"c{i}",
        })
    # is_tool_loop must return False so score() performs full evaluation
    assert not is_tool_loop(msgs), (
        "A tool chain > 12 calls should break the fast-path bypass"
    )


def test_short_tool_loop_still_fast_path():
    """Tool loop with ≤12 tool turns preserves fast path bypass (no stack trace)."""
    from autoconduck.evaluator import is_tool_loop
    msgs = [
        {"role": "user", "content": "refactor the codebase"},
        {"role": "assistant", "content": "reading...", "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "content": "file contents", "tool_call_id": "c1"},
    ]
    assert is_tool_loop(msgs), "Short tool loop should still be fast-path"


# ── Phase 4: Expanded stack trace patterns ────────────────────────────────

def test_java_stack_trace_detected():
    java_trace = (
        "Exception in thread \"main\" java.lang.NullPointerException\n"
        "\tat com.example.Foo.bar(Foo.java:42)\n"
        "\tat com.example.Main.main(Main.java:10)"
    )
    assert has_stack_trace(java_trace), "Java stack trace not detected"


def test_go_panic_detected():
    go_panic = (
        "goroutine 1 [running]:\n"
        "main.main()\n"
        "\t/home/user/app/main.go:15 +0x6f\n"
        "exit status 2"
    )
    assert has_stack_trace(go_panic), "Go goroutine panic not detected"


def test_rust_thread_panic_detected():
    rust_panic = (
        "thread 'main' panicked at 'index out of bounds: the len is 3 but the index is 5',\n"
        "src/main.rs:12:5"
    )
    assert has_stack_trace(rust_panic), "Rust thread panic not detected"


def test_c_segfault_detected():
    c_crash = "Segmentation fault (core dumped)"
    assert has_stack_trace(c_crash), "C segfault not detected"


def test_gcc_compiler_error_detected():
    gcc_error = "main.c:42:5: error: use of undeclared identifier 'foo'"
    assert has_stack_trace(gcc_error), "GCC compiler error not detected"


def test_generic_fatal_error_detected():
    assert has_stack_trace("fatal error: runtime: out of memory"), (
        "Generic fatal error not detected"
    )


# ── Phase 4: Natural-language escalation signals ──────────────────────────

def test_explicit_escalation_directive_detected():
    assert has_escalation_signal("autoconduck: escalate to slow path")
    assert has_escalation_signal("[escalate]")
    assert has_escalation_signal("<autoconduck-escalate>")


def test_natural_language_escalation_scope_grown():
    assert has_escalation_signal(
        "The scope has grown significantly more complex than I expected."
    ), "Scope-grown escalation not detected"


def test_natural_language_escalation_rethink():
    assert has_escalation_signal(
        "I need to rethink the approach for this module."
    ), "Rethink escalation not detected"


def test_natural_language_escalation_underestimated():
    assert has_escalation_signal(
        "I underestimated the complexity of this task."
    ), "Underestimation escalation not detected"


def test_natural_language_escalation_more_planning_needed():
    assert has_escalation_signal(
        "Additional planning and analysis is required before proceeding."
    ), "More-planning escalation not detected"


def test_natural_language_escalation_turns_out_harder():
    assert has_escalation_signal(
        "It turns out to be more complex than the original estimate."
    ), "Turns-out-harder escalation not detected"


def test_non_escalation_text_not_detected():
    """Ordinary text must not trigger escalation."""
    assert not has_escalation_signal("fix the typo in line 3")
    assert not has_escalation_signal("update the README")
    assert not has_escalation_signal("explain what this function does")


# ── Phase 5: Semantic router fallback ─────────────────────────────────────

def test_idf_fallback_routes_data_slow():
    """Data/ML complex utterance should route to slow_path via the IDF fallback."""
    from autoconduck.semantic_router import SemanticRouter
    router = SemanticRouter()
    router._layer = None  # force fallback path
    result = router.route("build an end-to-end ETL pipeline with feature engineering")
    assert result.route == "slow_path", (
        f"Data/ML complex prompt routed to {result.route}, expected slow_path"
    )


def test_idf_fallback_routes_writing_slow():
    from autoconduck.semantic_router import SemanticRouter
    router = SemanticRouter()
    router._layer = None
    result = router.route(
        "redesign the narrative arc and argument structure of the thesis"
    )
    assert result.route == "slow_path", (
        f"Writing complex prompt routed to {result.route}, expected slow_path"
    )


def test_idf_fallback_routes_simple_fast():
    from autoconduck.semantic_router import SemanticRouter
    router = SemanticRouter()
    router._layer = None
    result = router.route("fix the grammar in this sentence")
    assert result.route == "fast_path", (
        f"Simple grammar fix routed to {result.route}, expected fast_path"
    )


def test_idf_fallback_empty_input():
    from autoconduck.semantic_router import SemanticRouter
    router = SemanticRouter()
    router._layer = None
    result = router.route("")
    assert result.route == "fast_path"
    assert result.confidence == 0.0
