from autoconduck.config import Config
from autoconduck.evaluator import score, has_stack_trace
from autoconduck.semantic_router import RouteMatch


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
