from autoconduck import dispatcher
from autoconduck.semantic_router import RouteMatch
from autoconduck.config import Config


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
    monkeypatch.setattr(dispatcher.semantic_router, "route", lambda text: RouteMatch("fast_path", .1))
    decision = dispatcher.route(["unclear request"], [], config=Config(model_list=[{"id": "one"}, {"id": "two"}]), tiebreaker=lambda *args: "slow")
    assert decision.confidence_band == "ambiguous"
    assert decision.path == "slow"


def test_tiebreaker_failure_degrades_to_fast(monkeypatch):
    monkeypatch.setattr(dispatcher.semantic_router, "route", lambda text: RouteMatch("fast_path", .1))
    def fail(*args):
        raise RuntimeError("offline")
    assert dispatcher.route(["unclear request"], [], config=Config(model_list=[{"id": "one"}, {"id": "two"}]), tiebreaker=fail).path == "fast"

def test_single_model_short_circuits_tiebreaker():
    cfg = Config(model_list=[{"id": "deepseek-v4-flash", "enabled": True}])
    def fail(*args):
        raise AssertionError("tiebreaker must not run")
    decision = dispatcher.route(["hi"], [], config=cfg, tiebreaker=fail)
    assert decision.path == "fast"
    assert decision.reason == "single-model, router-resolved: fast"

def test_single_model_complex_resolves_slow_without_tiebreaker(monkeypatch):
    monkeypatch.setattr(dispatcher.semantic_router, "route", lambda text: RouteMatch("slow_path", .01))
    cfg = Config(model_list=[{"id": "deepseek-v4-flash", "enabled": True}])
    decision = dispatcher.route(
        ["refactor the application into multiple modules across the whole codebase"],
        [], config=cfg, tiebreaker=lambda *args: (_ for _ in ()).throw(AssertionError("must not run")),
    )
    assert decision.path == "slow"
    assert decision.reason == "single-model, router-resolved: slow"

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
