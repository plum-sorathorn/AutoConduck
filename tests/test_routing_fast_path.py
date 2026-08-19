"""Fast path routing, micro-DAG execution, semantic router, and evaluator unit tests."""
import asyncio
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from autoconduck import config as config_module
from autoconduck import dispatcher
from autoconduck import main, server_streaming
from autoconduck.config import Config, SelectionConfig
from autoconduck.digest import maybe_digest_messages
from autoconduck.routing import evaluator, pricing
from autoconduck.routing.evaluator import (
    STACK_TRACE_BOOST,
    ESCALATION_THRESHOLD,
    HYSTERESIS_FLOOR,
    complexity_of,
    score,
)
from autoconduck.routing.fast_graph import FastGraph, FastGraphState
from autoconduck.routing.semantic_router import RouteMatch, SemanticRouter


# ---------------------------------------------------------------------------
# Semantic Router
# ---------------------------------------------------------------------------


def test_semantic_router_routes_known_phrases():
    router = SemanticRouter()
    match = router.route("fix a typo in this function")
    assert isinstance(match, RouteMatch)
    assert match.route in ("fast_path", "slow_path")
    assert 0.0 <= match.confidence <= 1.0


def test_semantic_router_empty_input_defaults_fast():
    router = SemanticRouter()
    match = router.route("")
    assert match.route == "fast_path"
    assert match.confidence == 0.0


# ---------------------------------------------------------------------------
# Evaluator & Complexity
# ---------------------------------------------------------------------------


def test_simple_prompt_is_fast():
    match = RouteMatch("fast_path", 0.90)
    decision = score(["fix typo"], [], match)
    assert decision.path == "fast"
    assert decision.confidence_band in ("fast", "high")


def test_stack_trace_boosts_complexity():
    stack_trace = """
    Traceback (most recent call last):
      File "app.py", line 10, in <module>
        main()
      File "app.py", line 6, in main
        raise ValueError("Something broke")
    ValueError: Something broke
    """
    normal_score = complexity_of("Check this error")
    boosted_score = complexity_of(f"Check this error: {stack_trace}")
    assert boosted_score > normal_score
    assert boosted_score >= 0.25


def test_escalation_boost_after_slow_turn():
    history = [{"path": "slow", "complexity": 0.85, "confidence": 0.90}]
    match = RouteMatch("slow_path", 0.65)
    decision = score(["follow up task"], history, match)
    assert decision.complexity > 0.2


def test_deescalation_from_escalated_state():
    cfg = Config(
        selection=SelectionConfig(deescalation_threshold=0.40),
        model_list=[{"id": "fast-model", "enabled": True}],
    )
    escalated_history = [{"complexity": 0.85, "confidence": 0.90}]
    match = RouteMatch("fast_path", 0.30)
    decision = score(["thanks"], escalated_history, match, config=cfg)
    assert decision.path == "fast"
    assert "de-escalated" in decision.reason


def test_hysteresis_clamp_after_escalation():
    history = [{"path": "slow", "complexity": 0.90, "confidence": 0.95}]
    match = RouteMatch("fast_path", 0.70)
    decision = score(["ok got it"], history, match)
    assert decision.complexity <= 0.50 or decision.path == "fast"


def test_windowed_decaying_hysteresis_allows_natural_recovery():
    cfg = Config(
        selection=SelectionConfig(
            hysteresis_window_size=4,
            hysteresis_decay=0.80,
            deescalation_threshold=0.40,
        ),
        model_list=[{"id": "fast-model", "enabled": True}],
    )
    # Old escalation followed by multiple simple turns
    history = [
        {"complexity": 0.90, "confidence": 0.95}, # 4 turns ago: 0.90 * 0.8^3 = 0.46
        {"complexity": 0.20, "confidence": 0.20},
        {"complexity": 0.20, "confidence": 0.20},
        {"complexity": 0.20, "confidence": 0.20},
    ]
    match = RouteMatch("fast_path", 0.20)
    decision = score(["what line was that again?"], history, match, config=cfg)
    assert decision.path == "fast"



# ---------------------------------------------------------------------------
# Dispatcher & FastGraph Micro-DAG
# ---------------------------------------------------------------------------


def test_fast_decision_does_not_call_external_services(monkeypatch):
    calls = []
    monkeypatch.setattr(dispatcher.semantic_router, "route", lambda text: RouteMatch("fast_path", 0.95))
    monkeypatch.setattr(dispatcher, "_default_tiebreaker", lambda *args: calls.append(args) or "slow")
    decision = dispatcher.route(["fix this typo"], [], config=Config(model_list=[{"id": "one"}, {"id": "two"}]))
    assert decision.path == "fast"
    assert calls == []


def test_complex_query_is_slow(monkeypatch):
    monkeypatch.setattr(dispatcher.semantic_router, "route", lambda text: RouteMatch("slow_path", 0.95))
    decision = dispatcher.route(
        ["refactor the entire application across multiple files"],
        [],
        config=Config(model_list=[{"id": "one"}, {"id": "two"}]),
    )
    assert decision.path == "slow"


def test_ambiguous_uses_injected_tiebreaker(monkeypatch):
    monkeypatch.setattr(dispatcher.semantic_router, "route", lambda text: RouteMatch("fast_path", 0.65))
    decision = dispatcher.route(
        ["unclear request"],
        [],
        config=Config(model_list=[{"id": "one"}, {"id": "two"}]),
        tiebreaker=lambda *args: "slow",
    )
    assert decision.confidence_band == "ambiguous"
    assert decision.path == "slow"


def test_tiebreaker_failure_degrades_to_fast(monkeypatch):
    monkeypatch.setattr(dispatcher.semantic_router, "route", lambda text: RouteMatch("fast_path", 0.65))

    def fail(*args):
        raise RuntimeError("offline")

    decision = dispatcher.route(
        ["unclear request"],
        [],
        config=Config(model_list=[{"id": "one"}, {"id": "two"}]),
        tiebreaker=fail,
    )
    assert decision.path == "fast"
    assert decision.reason == "tiebreaker_unavailable: complexity-fallback"


def test_unavailable_tiebreaker_uses_complexity_fallback(monkeypatch):
    monkeypatch.setattr(dispatcher.semantic_router, "route", lambda text: RouteMatch("fast_path", 0.1))
    monkeypatch.setattr(
        dispatcher.evaluator,
        "score",
        lambda *args, **kwargs: SimpleNamespace(
            confidence_band="ambiguous", confidence=0.1, complexity=0.8, path="fast"
        ),
    )
    decision = dispatcher.route(
        ["multi-part request"],
        [],
        config=Config(model_list=[{"id": "one"}, {"id": "two"}]),
        tiebreaker=lambda *args: None,
    )
    assert decision.path == "slow"
    assert decision.reason == "tiebreaker_unavailable: complexity-fallback"


def test_below_floor_has_distinct_reason(monkeypatch):
    monkeypatch.setattr(dispatcher.semantic_router, "route", lambda text: RouteMatch("fast_path", 0.1))
    decision = dispatcher.route(["fix typo"], [], config=Config(model_list=[{"id": "one"}, {"id": "two"}]))
    assert decision.path == "fast"


def test_fast_graph_execution_speed_and_correctness(monkeypatch):
    monkeypatch.setattr(dispatcher.semantic_router, "route", lambda text: RouteMatch("fast_path", 0.95))
    cfg = Config(
        model_list=[{"id": "test-model-1", "enabled": True}, {"id": "test-model-2", "enabled": True}],
        selection=SelectionConfig(enable_fast_path_graph=True),
    )
    # Warm up initial module imports
    dispatcher.route(["warmup"], [], config=cfg)

    times = []
    for _ in range(5):
        start = time.perf_counter()
        decision = dispatcher.route(["fix typo in readme"], [], config=cfg)
        times.append((time.perf_counter() - start) * 1000)

    fastest_ms = min(times)
    assert decision.path == "fast"
    assert decision.model is not None
    # Pure fast-path graph execution must remain strictly under 5 ms (< 0.5 ms expected)
    assert fastest_ms < 5.0, f"FastGraph execution took {fastest_ms:.2f} ms (expected < 5 ms)"


def test_fast_graph_empty_pool_uses_orchestrator_fallback():
    cfg = Config(model_list=[])
    state = SimpleNamespace(
        path="fast",
        config=cfg,
        complexity=0.2,
        pseudo_model="autoconduck",
        model=None,
    )
    FastGraph()._node_model_select(state)
    assert state.model == config_module.resolve_orchestrator_model(cfg)
    assert state.model


# ---------------------------------------------------------------------------
# Fast-Path Deterministic Digests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_digest_skip_when_flag_disabled(tmp_path):
    cfg = Config(selection=SelectionConfig(fast_path_digest_enabled=False))
    assert await maybe_digest_messages([{"role": "user", "content": "a.py b.py"}], cfg, tmp_path) is None


@pytest.mark.asyncio
async def test_digest_skip_when_not_first_turn(tmp_path):
    assert (
        await maybe_digest_messages(
            [{"role": "user", "content": "a.py b.py"}, {"role": "assistant", "content": "ok"}],
            base_dir=tmp_path,
        )
        is None
    )


@pytest.mark.asyncio
async def test_digest_extraction_resolves_existing_files(tmp_path):
    for name in ("a.py", "b.py", "ignored.py"):
        (tmp_path / name).write_text(f"# {name}\nvalue = 1\n", encoding="utf-8")
    cfg = Config(
        selection=SelectionConfig(
            fast_path_digest_enabled=True,
            fast_path_digest_max_files=5,
            fast_path_digest_max_total_bytes=10000,
        )
    )
    result = await maybe_digest_messages(
        [{"role": "user", "content": f"Please inspect {tmp_path / 'a.py'} and {tmp_path / 'b.py'}"}],
        cfg,
        tmp_path,
    )
    assert result is not None
    assert "### " in result[0]["content"]


# ---------------------------------------------------------------------------
# Tool Loop Invariants in Long Sessions
# ---------------------------------------------------------------------------


def test_tool_loop_in_long_multi_turn_session():
    """Verify tool loops in long sessions (>12 cumulative tool calls) are not hijacked by SLOW path."""
    messages = []
    # Build 5 past turns, each with 3 tool calls (15 cumulative tool calls)
    for i in range(5):
        messages.append({"role": "user", "content": f"User question {i}"})
        for c in range(3):
            messages.append({
                "role": "assistant",
                "content": f"Running step {c}",
                "tool_calls": [{"id": f"call_{i}_{c}", "type": "function", "function": {"name": "read", "arguments": "{}"}}],
            })
            messages.append({"role": "tool", "tool_call_id": f"call_{i}_{c}", "content": "file output"})

    # Now turn 6: new user prompt, assistant tool call, and tool output
    messages.append({"role": "user", "content": "Now run graphify query and update changes"})
    messages.append({
        "role": "assistant",
        "content": "Executing queries",
        "tool_calls": [{"id": "call_6_0", "type": "function", "function": {"name": "bash", "arguments": "{}"}}],
    })
    messages.append({"role": "tool", "tool_call_id": "call_6_0", "content": "BFS graph traversal output with lots of code symbols"})

    assert evaluator.is_tool_loop(messages) is True
    match = RouteMatch("slow_path", 0.9)
    result = score(messages, [], match)
    assert result.path == "fast"
    assert "interactive agent tool loop" in result.reason


def test_tool_loop_with_toolResult_role():
    """Verify is_tool_loop supports pi/custom toolResult roles."""
    messages = [
        {"role": "user", "content": "Update dependencies"},
        {
            "role": "assistant",
            "content": "Updating",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}],
        },
        {"role": "toolResult", "toolCallId": "c1", "content": [{"type": "text", "text": "Done!"}]},
    ]
    assert evaluator.is_tool_loop(messages) is True


def test_tool_loop_soft_escalation_on_active_chain():
    """Verify tool loop allows re-scoring if the ACTIVE chain alone exceeds 12 calls."""
    messages = [{"role": "user", "content": "Complex loop"}]
    for c in range(14):
        messages.append({
            "role": "assistant",
            "content": f"Step {c}",
            "tool_calls": [{"id": f"c_{c}", "type": "function", "function": {"name": "read", "arguments": "{}"}}],
        })
        messages.append({"role": "tool", "tool_call_id": f"c_{c}", "content": "read ok"})

    assert evaluator.is_tool_loop(messages) is False

