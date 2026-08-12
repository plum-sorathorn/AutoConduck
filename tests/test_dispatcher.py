from autoconduck import dispatcher
from autoconduck.routing.semantic_router import RouteMatch
from autoconduck.config import Config
from types import SimpleNamespace


def test_fast_decision_does_not_call_external_services(monkeypatch):
    calls = []
    monkeypatch.setattr(dispatcher.semantic_router, "route", lambda text: RouteMatch("fast_path", .95))
    monkeypatch.setattr(dispatcher, "_default_tiebreaker", lambda *args: calls.append(args) or "slow")
    decision = dispatcher.route(["fix this typo"], [], config=Config(model_list=[{"id": "one"}, {"id": "two"}]))
    assert decision.path == "fast"
    assert calls == []


def test_complex_query_is_slow(monkeypatch):
    monkeypatch.setattr(dispatcher.semantic_router, "route", lambda text: RouteMatch("slow_path", .95))
    decision = dispatcher.route(["refactor the entire application across multiple files"], [], config=Config(model_list=[{"id": "one"}, {"id": "two"}]))
    assert decision.path == "slow"


def test_ambiguous_uses_injected_tiebreaker(monkeypatch):
    # confidence=0.65 is inside the ambiguous zone [ambiguous_low=0.60, ambiguous_high=0.75]
    monkeypatch.setattr(dispatcher.semantic_router, "route", lambda text: RouteMatch("fast_path", .65))
    decision = dispatcher.route(["unclear request"], [], config=Config(model_list=[{"id": "one"}, {"id": "two"}]), tiebreaker=lambda *args: "slow")
    assert decision.confidence_band == "ambiguous"
    assert decision.path == "slow"


def test_tiebreaker_failure_degrades_to_fast(monkeypatch):
    # confidence=0.65 is in the ambiguous zone, so the tiebreaker is invoked
    monkeypatch.setattr(dispatcher.semantic_router, "route", lambda text: RouteMatch("fast_path", .65))
    def fail(*args):
        raise RuntimeError("offline")
    decision = dispatcher.route(["unclear request"], [], config=Config(model_list=[{"id": "one"}, {"id": "two"}]), tiebreaker=fail)
    assert decision.path == "fast"
    assert decision.reason == "tiebreaker_unavailable: complexity-fallback"


def test_unavailable_tiebreaker_uses_complexity_fallback(monkeypatch):
    monkeypatch.setattr(dispatcher.semantic_router, "route", lambda text: RouteMatch("fast_path", .1))
    monkeypatch.setattr(
        dispatcher.evaluator,
        "score",
        lambda *args: SimpleNamespace(
            confidence_band="ambiguous", confidence=.1, complexity=.8, path="fast"
        ),
    )
    decision = dispatcher.route(
        ["multi-part request"], [],
        config=Config(model_list=[{"id": "one"}, {"id": "two"}]),
        tiebreaker=lambda *args: None,
    )
    assert decision.path == "slow"
    assert decision.reason == "tiebreaker_unavailable: complexity-fallback"


def test_below_floor_has_distinct_reason(monkeypatch):
    calls = []
    monkeypatch.setattr(dispatcher.semantic_router, "route", lambda text: RouteMatch("fast_path", .1))
    monkeypatch.setattr(dispatcher, "_default_tiebreaker", lambda *args: calls.append(args) or "SLOW 9")
    decision = dispatcher.route(
        ["fix typo"], [],
        config=Config(model_list=[{"id": "one"}, {"id": "two"}]),
    )
    assert decision.path == "fast"
    # With the fixed evaluator, confidence < ambiguous_low routes directly to fast
    # without ever touching the tiebreaker — the reason comes from the evaluator.
    assert decision.reason == "below ambiguous floor"
    assert calls == []


def test_single_model_short_circuits_tiebreaker():
    cfg = Config(model_list=[{"id": "deepseek-v4-flash", "enabled": True}])
    def fail(*args):
        raise AssertionError("tiebreaker must not run")
    decision = dispatcher.route(["hi"], [], config=cfg, tiebreaker=fail)
    assert decision.path == "fast"
    # 'hi' has very low complexity/confidence — evaluator fast-exits before
    # the single-model short-circuit in the dispatcher.
    assert decision.reason == "below ambiguous floor"

def test_single_model_complex_resolves_slow_without_tiebreaker(monkeypatch):
    # With high confidence (0.85) from slow_path router and genuine message complexity,
    # the evaluator correctly routes SLOW via 'semantic route and complexity'.
    monkeypatch.setattr(dispatcher.semantic_router, "route", lambda text: RouteMatch("slow_path", .85))
    cfg = Config(model_list=[{"id": "deepseek-v4-flash", "enabled": True}])
    decision = dispatcher.route(
        ["refactor the application into multiple modules across the whole codebase"],
        [], config=cfg, tiebreaker=lambda *args: (_ for _ in ()).throw(AssertionError("must not run")),
    )
    assert decision.path == "slow"
    assert decision.reason in ("single-model, router-resolved: slow", "semantic route and complexity")

def test_system_context_does_not_force_slow(monkeypatch):
    monkeypatch.setattr(dispatcher.semantic_router, "route", lambda text: RouteMatch("slow_path", .01))
    cfg = Config(model_list=[{"id": "one"}, {"id": "two"}])
    decision = dispatcher.route(
        [{"role": "system", "content": "refactor the entire application across multiple files"}, {"role": "user", "content": "hi"}],
        [], config=cfg, tiebreaker=lambda *args: "fast",
    )
    assert decision.path == "fast"


def test_claude_code_system_reminder_is_ignored_for_routing(monkeypatch):
    seen = []
    monkeypatch.setattr(dispatcher.semantic_router, "route", lambda text: seen.append(text) or RouteMatch("fast_path", .95))
    reminder = "<system-reminder>workspace instructions and tool context</system-reminder>"
    decision = dispatcher.route(
        [{"role": "user", "content": reminder + "\n\nhi"}], [],
        config=Config(model_list=[{"id": "one"}, {"id": "two"}]),
    )
    assert decision.path == "fast"
    assert seen == ["hi"]


def test_claude_code_reminder_does_not_hide_complex_user_intent(monkeypatch):
    monkeypatch.setattr(dispatcher.semantic_router, "route", lambda text: RouteMatch("slow_path", .95))
    reminder = "<system-reminder>workspace instructions</system-reminder>"
    decision = dispatcher.route(
        [{"role": "user", "content": reminder + "\n\nRefactor the entire application across multiple files and write integration tests."}],
        [], config=Config(model_list=[{"id": "one"}, {"id": "two"}]),
    )
    assert decision.path == "slow"
