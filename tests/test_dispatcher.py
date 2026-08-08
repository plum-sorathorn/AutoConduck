from autoconduck import dispatcher
from autoconduck.semantic_router import RouteMatch


def test_fast_decision_does_not_call_external_services(monkeypatch):
    calls = []
    monkeypatch.setattr(dispatcher.semantic_router, "route", lambda text: RouteMatch("fast_path", .95))
    monkeypatch.setattr(dispatcher, "_default_tiebreaker", lambda *args: calls.append(args) or "slow")
    decision = dispatcher.route(["fix this typo"], [])
    assert decision.path == "fast"
    assert calls == []


def test_complex_query_is_slow(monkeypatch):
    monkeypatch.setattr(dispatcher.semantic_router, "route", lambda text: RouteMatch("slow_path", .95))
    decision = dispatcher.route(["refactor the entire application across multiple files"], [])
    assert decision.path == "slow"


def test_ambiguous_uses_injected_tiebreaker(monkeypatch):
    monkeypatch.setattr(dispatcher.semantic_router, "route", lambda text: RouteMatch("fast_path", .1))
    decision = dispatcher.route(["unclear request"], [], tiebreaker=lambda *args: "slow")
    assert decision.confidence_band == "ambiguous"
    assert decision.path == "slow"


def test_tiebreaker_failure_degrades_to_fast(monkeypatch):
    monkeypatch.setattr(dispatcher.semantic_router, "route", lambda text: RouteMatch("fast_path", .1))
    def fail(*args):
        raise RuntimeError("offline")
    assert dispatcher.route(["unclear request"], [], tiebreaker=fail).path == "fast"
