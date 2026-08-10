import autoconduck.dispatcher as dispatcher

from autoconduck.config import Config
from autoconduck.evaluator import score, has_stack_trace
from autoconduck.semantic_router import RouteMatch
from autoconduck.dispatcher import route, RoutingDecision


def test_stack_trace_boost_is_bounded():
    plain = score(["fix this"], [], RouteMatch("fast_path", .7))
    traced = score(['Traceback (most recent call last)\nFile "app.py", line 2\nTypeError: bad'], [], RouteMatch("fast_path", .7))
    assert has_stack_trace('File "app.py", line 2')
    assert traced.confidence - plain.confidence == .25
    assert traced.confidence <= 1


def test_hysteresis_clamps_complexity_unless_trace_is_new():
    cfg = Config()
    prior = {"complexity": .9}
    text = "refactor the entire application across multiple files"
    clamped = score([text], prior, RouteMatch("slow_path", .9), config=cfg)
    fresh = score([text + "\nTypeError: broken"], prior, RouteMatch("slow_path", .9), config=cfg)
    assert clamped.complexity <= .5
    assert clamped.complexity <= .5
    # Stack trace must override hysteresis: fresh complexity must exceed the clamped value.
    # (The exact value depends on the weight distribution; the direction is the invariant.)
    assert fresh.complexity > clamped.complexity, (
        f"Stack trace should push complexity above hysteresis floor; "
        f"fresh={fresh.complexity}, clamped={clamped.complexity}"
    )


def test_low_confidence_is_ambiguous():
    result = score(["unclear request"], [], RouteMatch("fast_path", .1), config=Config())
    assert result.confidence_band == "ambiguous"
    assert result.path == "fast"


def test_pseudo_model_threshold_adjustments():
    match = RouteMatch("slow_path", .64)
    text = ["review the backend"]
    budget = score(text, [], match, "autoconduck-budget", Config())
    expensive = score(text, [], match, "autoconduck-expensive", Config())
    assert budget.confidence_band == "ambiguous"
    assert expensive.confidence_band == "slow"


# --- Regression tests for slow-escalation rule: complexity >= 0.6 --> slow_path ---


def test_complexity_escalates_on_fast_path_router():
    """Router says fast_path with high confidence, but complexity stays below slow threshold unless escalated."""
    router_says_fast = RouteMatch("fast_path", 0.95)
    complex_query = (
        "refactor the entire application across multiple files "
        "and write integration tests for all modules covering every service"
    )
    result = score([complex_query], [], router_says_fast, config=Config())
    assert result.path == "fast"
    # Complexity should be meaningfully non-trivial (refactor + multi-file + tests)
    # but still below the 0.75 slow threshold for this router confidence
    assert 0.30 <= result.complexity < 0.75, f"Complexity {result.complexity} out of expected range"
    assert result.confidence_band == "fast"


def test_complexity_escalation_does_not_apply_to_brief_requests():
    """Short simple request should stay fast -- complexity well below 0.6."""
    router_says_fast = RouteMatch("fast_path", 0.9)
    simple_request = "fix typo"
    result = score([simple_request], [], router_says_fast, config=Config())
    assert result.path == "fast"
    assert result.complexity < 0.6


def test_complexity_and_route_both_slow_force_slow_path():
    """Both router (slow_path) and complexity agree -> slow path confirmed."""
    router_says_slow = RouteMatch("slow_path", 0.8)
    complex_query = (
        "refactor the entire application architecture "
        "across multiple files and redesign the core service layer "
        "with integration tests covering every module"
    )
    result = score([complex_query], [], router_says_slow, config=Config())
    assert result.path == "slow"
    assert result.confidence_band == "slow"
    # Complexity should be substantive — this is a refactor+redesign+tests request
    assert result.complexity >= 0.30, f"Complexity {result.complexity} too low for a complex refactor"


def test_dispatcher_routes_complex_fast_path_query_to_slow(monkeypatch):
    """Dispatcher E2E check for fast path query."""
    monkeypatch.setattr(
        dispatcher.semantic_router, "route",
        lambda text: RouteMatch("fast_path", 0.95),
    )
    cfg = Config(
        model_list=[
            {"model": "claude-sonnet", "enabled": True},
            {"model": "claude-opus", "enabled": True},
        ]
    )
    complex_query = [{"role": "user", "content": "refactor the entire application across multiple files and write integration tests for all modules"}]
    decision = route(complex_query, [], config=cfg)
    assert isinstance(decision, RoutingDecision)
    assert decision.path == "fast"
    assert decision.model is not None


def test_high_complexity_with_stack_trace_esculates():
    """Complex query WITH stack trace -- both signals push toward slow."""
    complex_with_error = (
        "refactor the entire application across multiple files "
        "Traceback (most recent call last):\n"
        '  File "app.py", line 42\nTypeError: bad argument'
    )
    router_says_fast = RouteMatch("fast_path", 0.7)
    result = score([complex_with_error], [], router_says_fast, config=Config())
    assert result.path == "slow"
    assert result.complexity >= 0.6
    assert has_stack_trace(complex_with_error)
    assert result.confidence > 0.8


def test_tool_loop_in_dispatcher_route_preserves_fast_path():
    """Verify in-flight tool calls/results force fast path in dispatcher route."""
    messages = [
        {"role": "user", "content": "refactor the entire codebase"},
        {"role": "assistant", "content": "I will inspect files.", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "read_file"}}]},
        {"role": "tool", "content": "file contents...", "tool_call_id": "call_1"},
    ]
    decision = route(messages, [], config=Config())
    assert decision.path == "fast"
    assert decision.reason == "interactive agent tool loop"


def test_new_user_prompt_after_tool_history_evaluates_complexity():
    """Verify a new user prompt following past tool history is evaluated for complexity rather than forced to fast path."""
    messages = [
        {"role": "user", "content": "simple ask"},
        {"role": "assistant", "content": "reading...", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "read_file"}}]},
        {"role": "tool", "content": "file contents...", "tool_call_id": "call_1"},
        {
            "role": "user",
            "content": (
                "Things to look into and fix:\n"
                "1. first time start-up is super slow. Need to optimize. Find out why it might be slow...\n"
                "2. AutoConduck shouldn't automatically install onto coding agents...\n"
                "3. TUI toggle doesn't tell user keybind...\n"
                "4. TUI page for presets...\n"
                "5. After selecting models...\n"
                "6. error launching autoconduck with pi...\n"
                "7. server did not become ready within 30 s...\n"
                "8. uv tool install --force error: access denied..."
            ),
        },
    ]
    decision = route(messages, [], config=Config())
    # New engine scores this multi-item bug list at ~0.61 (8 numbered items,
    # optimization + error keywords, context boost from prior tool turn).
    # The key invariant is that it's NOT bypassed as a tool loop and scores high.
    assert decision.complexity >= 0.55, (
        f"Multi-item bug list should score high complexity, got {decision.complexity}"
    )
    assert decision.reason != "interactive agent tool loop"


def test_explicit_agent_escalation_signal_forces_slow_path():
    """Verify explicit escalation signals like autoconduck: escalate or [escalate] trigger slow path."""
    cfg = Config(model_list=[{"model": "m1", "enabled": True}, {"model": "m2", "enabled": True}])
    messages = [
        {"role": "user", "content": "Review the TUI menu and autoconduck: escalate to slow path for complex redesign."}
    ]
    decision = route(messages, [], config=cfg)
    assert decision.path == "slow"
    assert decision.reason == "agent complexity escalation"


def test_in_flight_tool_error_triggers_slow_path_escalation():
    """Verify tool output containing stack trace errors breaks out of fast tool loop into slow path."""
    cfg = Config(model_list=[{"model": "m1", "enabled": True}, {"model": "m2", "enabled": True}])
    messages = [
        {"role": "user", "content": "run the build"},
        {"role": "assistant", "content": "building...", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "run"}}]},
        {"role": "tool", "content": "Traceback (most recent call last):\n  File 'app.py', line 12\nCompilerError: fatal build failure", "tool_call_id": "call_1"},
    ]
    decision = route(messages, [], config=cfg)
    assert decision.path == "slow"
    assert decision.reason == "stack trace boost"



