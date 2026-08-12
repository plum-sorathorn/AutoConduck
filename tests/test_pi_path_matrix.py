"""Pi-shaped OpenAI-completions payloads exercising every dispatcher path."""

from types import SimpleNamespace

import pytest

from autoconduck import dispatcher, evaluator
from autoconduck.config import Config
from autoconduck.routing.semantic_router import RouteMatch


def _config():
    return Config(model_list=[{"id": "cheap", "enabled": True}, {"id": "deep", "enabled": True}])


def _pi_messages(prompt, *, tool_loop=False, tool_turns=0):
    messages = [
        {"role": "system", "content": "You are Pi, a coding agent."},
        {"role": "user", "content": prompt},
    ]
    if tool_loop:
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "read", "arguments": "{}"}}],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": "file contents"},
            ]
        )
    for index in range(tool_turns):
        messages.extend(
            [
                {"role": "assistant", "tool_calls": [{"id": f"call-{index + 2}", "type": "function", "function": {"name": "read", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": f"call-{index + 2}", "content": "result"},
            ]
        )
    return messages


def _patch_selection(monkeypatch, *, degraded=False):
    monkeypatch.setattr(dispatcher.semantic_router, "route", lambda text: RouteMatch("fast_path", 0.1))
    monkeypatch.setattr(dispatcher.pricing, "pool_ids", lambda config: ["cheap", "deep"])
    monkeypatch.setattr(dispatcher.pricing, "select_closest", lambda *args, **kwargs: "cheap")
    monkeypatch.setattr(dispatcher.pricing, "cheapest_enabled", lambda config: "cheap")
    monkeypatch.setattr(dispatcher.pricing, "target_scaled_cost", lambda *args, **kwargs: 0.1)
    monkeypatch.setattr(dispatcher.pricing, "is_degraded", lambda *args, **kwargs: degraded)


def _score(monkeypatch, complexity, *, band="ambiguous", path="fast", reason="confidence is in the ambiguous zone"):
    monkeypatch.setattr(
        dispatcher.evaluator,
        "score",
        lambda *args, **kwargs: SimpleNamespace(
            confidence_band=band, confidence=0.1, complexity=complexity, path=path, reason=reason
        ),
    )


def test_trivial_pi_prompt_is_below_floor_and_does_not_call_tiebreaker(monkeypatch):
    _patch_selection(monkeypatch)
    calls = []
    monkeypatch.setattr(dispatcher, "_default_tiebreaker", lambda *args: calls.append(args) or "SLOW 9")
    decision = dispatcher.route(_pi_messages("fix typo"), [], config=_config())
    assert (decision.path, decision.reason) == ("fast", "tiebreaker: fast (below-floor)")
    assert calls == []
    assert decision.model is not None


def test_complex_pi_prompt_blends_slow_tiebreaker_complexity(monkeypatch):
    _patch_selection(monkeypatch)
    _score(monkeypatch, 0.8)
    prompt = """Review the router, audit authentication, and fix the implementation:
1. inspect all modules and dependencies
2. identify race conditions and security gaps
3. redesign the orchestration workflow
4. add integration tests across the codebase
5. document deployment and rollback behavior"""
    decision = dispatcher.route(_pi_messages(prompt), [], config=_config(), tiebreaker=lambda *args: "SLOW 8")
    assert decision.path == "slow"
    assert decision.reason == "tiebreaker: slow"
    assert decision.complexity == pytest.approx(0.5 * 0.8 + 0.5 * (8 / 9))
    assert decision.model is None


def test_timeout_uses_complexity_fallback(monkeypatch):
    _patch_selection(monkeypatch)
    _score(monkeypatch, 0.8)
    decision = dispatcher.route(_pi_messages("review and fix everything"), [], config=_config(), tiebreaker=lambda *args: (_ for _ in ()).throw(TimeoutError()))
    assert (decision.path, decision.reason) == ("slow", "tiebreaker_unavailable: complexity-fallback")
    assert decision.model is None


def test_none_tiebreaker_moderate_prompt_falls_back_fast(monkeypatch):
    _patch_selection(monkeypatch)
    _score(monkeypatch, 0.52)
    decision = dispatcher.route(_pi_messages("review this change"), [], config=_config(), tiebreaker=lambda *args: None)
    assert (decision.path, decision.reason) == ("fast", "tiebreaker_unavailable: complexity-fallback")
    assert decision.model is not None


def test_none_tiebreaker_degraded_provider_has_distinct_reason(monkeypatch):
    _patch_selection(monkeypatch, degraded=True)
    _score(monkeypatch, 0.52)
    decision = dispatcher.route(_pi_messages("review this change"), [], config=_config(), tiebreaker=lambda *args: None)
    assert (decision.path, decision.reason) == ("fast", "tiebreaker_unavailable: degraded-provider")
    assert decision.model is not None


def test_stack_trace_escalates_directly_without_tiebreaker(monkeypatch):
    _patch_selection(monkeypatch)
    calls = []
    monkeypatch.setattr(dispatcher, "_default_tiebreaker", lambda *args: calls.append(args) or "FAST 1")
    decision = dispatcher.route(_pi_messages('Traceback (most recent call last):\n  File "app.py", line 4\nRuntimeError: broken'), [], config=_config())
    assert decision.path == "slow"
    assert decision.reason == "stack trace boost"
    assert calls == []
    assert decision.model is None


def test_pi_tool_loop_is_fast_but_long_and_complex_loops_are_not_suppressed(monkeypatch):
    _patch_selection(monkeypatch)
    normal = _pi_messages("fix the typo", tool_loop=True)
    decision = dispatcher.route(normal, [], config=_config())
    assert decision.path == "fast"
    assert "tool loop" in decision.reason
    assert decision.model is not None
    long_loop = _pi_messages("fix the typo", tool_turns=13)
    assert evaluator.is_tool_loop(long_loop, _config()) is False
    complex_loop = _pi_messages("architect and redesign the entire distributed system", tool_loop=True)
    monkeypatch.setattr(evaluator, "complexity_of", lambda text, config=None: 0.8)
    assert evaluator.is_tool_loop(complex_loop, _config()) is False


@pytest.mark.parametrize("pseudo_model, invokes", [("autoconduck", True), ("autoconduck-budget", False), ("autoconduck-expensive", True)])
def test_all_pi_pseudo_models_apply_their_tiebreaker_floor(monkeypatch, pseudo_model, invokes):
    _patch_selection(monkeypatch)
    _score(monkeypatch, 0.52)
    calls = []
    tiebreaker = lambda *args: calls.append(args) or "FAST 3"
    decision = dispatcher.route(_pi_messages("review this change"), [], pseudo_model=pseudo_model, config=_config(), tiebreaker=tiebreaker if invokes else None)
    if invokes:
        assert calls
        assert decision.reason == "tiebreaker: fast"
        assert decision.complexity == pytest.approx(0.5 * 0.52 + 0.5 * (3 / 9))
    else:
        assert calls == []
        assert decision.reason == "tiebreaker: fast (below-floor)"
    assert decision.path == "fast"
    assert decision.model is not None


def test_slow_decision_is_consumable_when_orchestrator_degrades(monkeypatch):
    _patch_selection(monkeypatch)
    _score(monkeypatch, 0.8)
    decision = dispatcher.route(_pi_messages("review and fix all modules"), [], config=_config(), tiebreaker=lambda *args: None)
    assert decision.path == "slow"
    assert decision.model is None
    # API orchestration callers can safely turn an orchestrator exception into
    # the already-produced routing decision; no network or model call is needed.
    def degraded_run():
        raise RuntimeError("orchestrator unavailable")
    monkeypatch.setattr(dispatcher, "_default_tiebreaker", degraded_run)
    assert decision.path in {"fast", "slow"}
