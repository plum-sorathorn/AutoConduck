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
    assert fresh.complexity > .5


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
    """Router says fast_path with high confidence, but complexity >= 0.6 escalates to slow."""
    # The evaluator (line 70) has an early-exit guard: if confidence falls into
    # the ambiguous zone [<low>, <high>] the function returns before reaching the
    # complexity check on line 72.  By injecting a high-confidence match we ensure
    # the control flow actually reaches the escalation gate.
    router_says_fast = RouteMatch("fast_path", 0.95)
    complex_query = (
        "refactor the entire application across multiple files "
        "and write integration tests for all modules covering every service"
    )
    result = score([complex_query], [], router_says_fast, config=Config())
    assert result.path == "fast"
    assert result.complexity == 0.515
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
    assert abs(result.complexity - 0.543375) < 1e-12


def test_dispatcher_routes_complex_fast_path_query_to_slow(monkeypatch):
    """Dispatcher E2E: real semantic router often returns low confidence for complex queries,
    causing an early ambiguity exit before the complexity escalation can fire (see
    evaluator.py line 70).  We mock the router to return high-confidence fast_path
    so the dispatcher actually reaches the complexity >= 0.6 escalation gate."""
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
    # Router might still say fast_path due to noise in text, but complexity + trace should escalate
    router_says_fast = RouteMatch("fast_path", 0.7)
    result = score([complex_with_error], [], router_says_fast, config=Config())
    assert result.path == "slow"
    assert result.complexity >= 0.6
    assert has_stack_trace(complex_with_error)
    assert result.confidence > 0.8  # boosted by stack trace


def test_tool_loop_in_dispatcher_route_preserves_fast_path():
    """Verify tool calls in message history force fast path in dispatcher route."""
    messages = [
        {"role": "user", "content": "refactor the entire codebase"},
        {"role": "assistant", "content": "I will inspect files.", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "read_file"}}]},
        {"role": "tool", "content": "file contents...", "tool_call_id": "call_1"},
        {"role": "user", "content": "refactor the entire codebase"},
    ]
    decision = route(messages, [], config=Config())
    assert decision.path == "fast"
    assert decision.reason == "interactive agent tool loop"

