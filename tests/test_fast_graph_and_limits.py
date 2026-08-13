import time
from types import SimpleNamespace

from autoconduck import config as config_module
from autoconduck.config import Config, SelectionConfig
from autoconduck.routing import pricing, evaluator, dispatcher
from autoconduck.routing.fast_graph import FastGraph, FastGraphState
from autoconduck.routing.semantic_router import RouteMatch


def test_expensive_model_limit_in_select_closest():
    cfg = Config(
        model_list=[
            {"id": "cheap-fast-model", "price_in": 0.1, "price_out": 0.2, "enabled": True},
            {"id": "expensive-mega-model", "price_in": 100.0, "price_out": 300.0, "enabled": True},
        ],
        selection=SelectionConfig(max_file_read_scaled_cost=0.55),
    )
    # Expensive model scaled cost should be > 0.55
    assert pricing.is_expensive_model("expensive-mega-model", cfg) is True
    assert pricing.is_expensive_model("cheap-fast-model", cfg) is False

    # Calling select_closest with max_scaled_cost=0.55 must filter out the expensive model
    selected = pricing.select_closest(
        pricing.pool_ids(cfg), 0.90, cfg, max_scaled_cost=0.55
    )
    assert selected == "cheap-fast-model"


def test_deescalation_from_escalated_state():
    cfg = Config(
        selection=SelectionConfig(deescalation_threshold=0.40),
        model_list=[{"id": "fast-model", "enabled": True}],
    )
    # Simulate escalated history
    escalated_history = [{"complexity": 0.85, "confidence": 0.90}]
    match = RouteMatch("fast_path", 0.30)

    # Simple input ("thanks") under deescalation threshold
    score = evaluator.score(["thanks"], escalated_history, match, config=cfg)
    assert score.path == "fast"
    assert "de-escalated" in score.reason


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
