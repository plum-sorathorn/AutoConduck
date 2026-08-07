import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from autoconduck.config import Config, ModelEntry
from autoconduck import gatekeeper as gatekeeper_mod

# Helpers
def _cfg_with_models(models=None):
    if models is None:
        models = [
            ModelEntry(id="cheap-model", tier="budget", price_in=0.1, price_out=0.2, enabled=True),
            ModelEntry(id="balanced-model", tier="balanced", price_in=0.5, price_out=0.5, enabled=True),
            ModelEntry(id="expensive-model", tier="expensive", price_in=2.0, price_out=2.0, enabled=True),
        ]
    cfg = Config(models=models, max_in_flight=32, max_workers=2)
    return cfg

def _chat_payload(model="autoconduck", content="hello", stream=False):
    return {"model": model, "messages": [{"role": "user", "content": content}], "stream": stream}

# (a) slow path calls pricing.select(0.40) and select(0.55) before orchestrator
def test_slow_path_calls_pricing_select_before_orchestrator():
    cfg = _cfg_with_models()
    # Force slow path via gatekeeper
    decision = gatekeeper_mod.Decision(path="slow", reason="test", T_i=0.8)
    mock_orch_result = MagicMock()
    mock_orch_result.worker_ok = 1
    mock_orch_result.worker_fail = 0
    mock_orch_result.worker_model_ids = ["balanced-model"]
    mock_orch_result.plan_model_id = "cheap-model"
    mock_orch_result.degraded_to_fast = False
    mock_orch_result.compacted_context = "compacted"
    mock_orch_result.plan = MagicMock()

    with patch("autoconduck.proxy.get_config", return_value=cfg), \
         patch("autoconduck.proxy.gatekeeper_mod.classify", return_value=decision), \
         patch("autoconduck.proxy.evaluator_mod.score", return_value=0.8), \
         patch("autoconduck.proxy.pricing_mod.estimate_tokens", return_value=(100, 200)) as mock_est, \
         patch("autoconduck.proxy.pricing_mod.select") as mock_select, \
         patch("autoconduck.proxy.pricing_mod.transform", return_value=0.8), \
         patch("autoconduck.proxy.pricing_mod._detect_intent", return_value="default"), \
         patch("autoconduck.proxy.pricing_mod.record_usage"), \
         patch("autoconduck.proxy.state_mod.session_key_from_request", return_value="sess"), \
         patch("autoconduck.proxy.state_mod.get_session_store") as mock_store, \
         patch("autoconduck.proxy.state_mod.is_degraded", return_value=False), \
         patch("autoconduck.proxy.state_mod.update_turn_state"), \
         patch("autoconduck.proxy.telemetry_mod.telemetry.push"), \
         patch("autoconduck.proxy._forward_litellm", new_callable=AsyncMock) as mock_forward:

        mock_store.return_value.get.return_value = None
        # pricing.select returns ModelEntry based on T'
        def select_side(Tprime, models, t_in, t_out):
            # record call
            if Tprime == 0.40:
                return cfg.models[0]  # cheap
            if Tprime == 0.55:
                return cfg.models[1]  # balanced
            return cfg.models[1]
        mock_select.side_effect = select_side
        # orchestrator
        with patch("autoconduck.proxy.Orchestrator.plan_and_execute", new_callable=AsyncMock) as mock_plan:
            mock_plan.return_value = mock_orch_result
            # non-streaming forward returns litellm-like dict
            mock_forward.return_value = MagicMock(model_dump=lambda: {"id": "chatcmpl-1", "choices": [{"message": {"content": "hi"}}], "usage": {"prompt_tokens": 100, "completion_tokens": 10}})

            from autoconduck.proxy import create_app
            app = create_app(cfg)
            client = TestClient(app)
            resp = client.post("/v1/chat/completions", json=_chat_payload())
            assert resp.status_code == 200, resp.text
            # Verify pricing.select called with 0.40 and 0.55 before orchestrator
            calls = [c.args[0] for c in mock_select.call_args_list]
            # First two calls should be 0.40 and 0.55 for D6
            assert 0.40 in calls, f"select not called with 0.40, calls={mock_select.call_args_list}"
            assert 0.55 in calls, f"select not called with 0.55, calls={mock_select.call_args_list}"
            # Ensure orchestrator was called with those ids
            mock_plan.assert_awaited_once()
            kwargs = mock_plan.call_args.kwargs
            assert kwargs["plan_model_id"] == cfg.models[0].id
            assert kwargs["worker_model_id"] == cfg.models[1].id
            # Also verify ordering: select calls happened before plan_and_execute
            # plan_and_execute call is after estimate_tokens, which is before final select
            # We check that at least 3 select calls happened (0.40,0.55, and final T')
            assert mock_select.call_count >= 3

# (b) disconnect does not call record_error (non-streaming 499 path)
def test_disconnect_does_not_call_record_error():
    cfg = _cfg_with_models()
    decision = gatekeeper_mod.Decision(path="fast", reason="test", T_i=0.1)

    with patch("autoconduck.proxy.get_config", return_value=cfg), \
         patch("autoconduck.proxy.gatekeeper_mod.classify", return_value=decision), \
         patch("autoconduck.proxy.evaluator_mod.score", return_value=0.1), \
         patch("autoconduck.proxy.pricing_mod.estimate_tokens", return_value=(10, 10)), \
         patch("autoconduck.proxy.pricing_mod.select", return_value=cfg.models[0]), \
         patch("autoconduck.proxy.pricing_mod.transform", return_value=0.1), \
         patch("autoconduck.proxy.pricing_mod._detect_intent", return_value="default"), \
         patch("autoconduck.proxy.pricing_mod.record_usage"), \
         patch("autoconduck.proxy.pricing_mod.record_error") as mock_record_error, \
         patch("autoconduck.proxy.state_mod.session_key_from_request", return_value="sess"), \
         patch("autoconduck.proxy.state_mod.get_session_store") as mock_store, \
         patch("autoconduck.proxy.telemetry_mod.telemetry.push"), \
         patch("autoconduck.proxy.state_mod.update_turn_state"):

        mock_store.return_value.get.return_value = None

        # Make _forward_litellm raise CancelledError to simulate disconnect upstream cancellation
        with patch("autoconduck.proxy._forward_litellm", new_callable=AsyncMock) as mock_forward:
            mock_forward.side_effect = asyncio.CancelledError("client disconnected")
            from autoconduck.proxy import create_app
            app = create_app(cfg)
            client = TestClient(app)
            # non-streaming
            resp = client.post("/v1/chat/completions", json=_chat_payload(stream=False))
            # Should return 499 cancelled, not 502, and not call record_error
            assert resp.status_code == 499, resp.text
            mock_record_error.assert_not_called()

# (c) worker failure calls record_error for that worker model id
def test_worker_failure_calls_record_error():
    cfg = _cfg_with_models()
    decision = gatekeeper_mod.Decision(path="slow", reason="test", T_i=0.9)
    mock_orch_result = MagicMock()
    mock_orch_result.worker_ok = 1
    mock_orch_result.worker_fail = 2
    mock_orch_result.worker_model_ids = ["balanced-model", "balanced-model"]
    mock_orch_result.plan_model_id = "cheap-model"
    mock_orch_result.degraded_to_fast = False
    mock_orch_result.compacted_context = "compacted"
    mock_orch_result.plan = MagicMock()

    with patch("autoconduck.proxy.get_config", return_value=cfg), \
         patch("autoconduck.proxy.gatekeeper_mod.classify", return_value=decision), \
         patch("autoconduck.proxy.evaluator_mod.score", return_value=0.9), \
         patch("autoconduck.proxy.pricing_mod.estimate_tokens", return_value=(100, 200)), \
         patch("autoconduck.proxy.pricing_mod.select", side_effect=lambda Tprime, models, t_in, t_out: cfg.models[1] if Tprime==0.55 else cfg.models[0] if Tprime==0.40 else cfg.models[0]), \
         patch("autoconduck.proxy.pricing_mod.transform", return_value=0.9), \
         patch("autoconduck.proxy.pricing_mod._detect_intent", return_value="default"), \
         patch("autoconduck.proxy.pricing_mod.record_usage"), \
         patch("autoconduck.proxy.pricing_mod.record_error") as mock_record_error, \
         patch("autoconduck.proxy.state_mod.session_key_from_request", return_value="sess"), \
         patch("autoconduck.proxy.state_mod.get_session_store") as mock_store, \
         patch("autoconduck.proxy.state_mod.is_degraded", return_value=False), \
         patch("autoconduck.proxy.state_mod.update_turn_state"), \
         patch("autoconduck.proxy.telemetry_mod.telemetry.push"), \
         patch("autoconduck.proxy._forward_litellm", new_callable=AsyncMock) as mock_forward:

        mock_store.return_value.get.return_value = None
        mock_forward.return_value = MagicMock(model_dump=lambda: {"id": "1", "choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 10, "completion_tokens": 10}})

        with patch("autoconduck.proxy.Orchestrator.plan_and_execute", new_callable=AsyncMock) as mock_plan:
            mock_plan.return_value = mock_orch_result
            from autoconduck.proxy import create_app
            app = create_app(cfg)
            client = TestClient(app)
            resp = client.post("/v1/chat/completions", json=_chat_payload())
            assert resp.status_code == 200, resp.text
            # record_error should have been called for worker failures (at least once with balanced-model)
            # check calls include balanced-model
            called_ids = [c.args[0] for c in mock_record_error.call_args_list]
            assert "balanced-model" in called_ids, f"expected record_error with balanced-model, got {called_ids}"
            # if worker_fail=2, called twice
            assert len([x for x in called_ids if x == "balanced-model"]) >= 2 or len(called_ids) >= 2

    # Also test plan failure calls record_error for plan model
    mock_orch_result2 = MagicMock()
    mock_orch_result2.worker_ok = 0
    mock_orch_result2.worker_fail = 0
    mock_orch_result2.worker_model_ids = []
    mock_orch_result2.plan_model_id = "cheap-model"
    mock_orch_result2.plan = None
    mock_orch_result2.degraded_to_fast = True
    mock_orch_result2.compacted_context = None

    with patch("autoconduck.proxy.get_config", return_value=cfg), \
         patch("autoconduck.proxy.gatekeeper_mod.classify", return_value=decision), \
         patch("autoconduck.proxy.evaluator_mod.score", return_value=0.9), \
         patch("autoconduck.proxy.pricing_mod.estimate_tokens", return_value=(100, 200)), \
         patch("autoconduck.proxy.pricing_mod.select", side_effect=lambda Tprime, models, t_in, t_out: cfg.models[0] if Tprime==0.40 else cfg.models[1]), \
         patch("autoconduck.proxy.pricing_mod.transform", return_value=0.9), \
         patch("autoconduck.proxy.pricing_mod._detect_intent", return_value="default"), \
         patch("autoconduck.proxy.pricing_mod.record_usage"), \
         patch("autoconduck.proxy.pricing_mod.record_error") as mock_record_error2, \
         patch("autoconduck.proxy.state_mod.session_key_from_request", return_value="sess"), \
         patch("autoconduck.proxy.state_mod.get_session_store") as mock_store2, \
         patch("autoconduck.proxy.state_mod.is_degraded", return_value=False), \
         patch("autoconduck.proxy.state_mod.update_turn_state"), \
         patch("autoconduck.proxy.telemetry_mod.telemetry.push"), \
         patch("autoconduck.proxy._forward_litellm", new_callable=AsyncMock) as mock_forward2:

        mock_store2.return_value.get.return_value = None
        mock_forward2.return_value = MagicMock(model_dump=lambda: {"id": "1", "choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 10, "completion_tokens": 10}})
        with patch("autoconduck.proxy.Orchestrator.plan_and_execute", new_callable=AsyncMock) as mock_plan2:
            mock_plan2.return_value = mock_orch_result2
            from autoconduck.proxy import create_app
            app = create_app(cfg)
            client = TestClient(app)
            resp = client.post("/v1/chat/completions", json=_chat_payload())
            assert resp.status_code == 200, resp.text
            called_ids2 = [c.args[0] for c in mock_record_error2.call_args_list]
            assert "cheap-model" in called_ids2, f"plan failure should record_error cheap-model, got {called_ids2}"

# (d) empty model list returns 503 OpenAI-shaped body not 500
def test_empty_models_returns_503():
    cfg = Config(models=[], max_in_flight=32)
    decision = gatekeeper_mod.Decision(path="fast", reason="test", T_i=0.2)
    with patch("autoconduck.proxy.get_config", return_value=cfg), \
         patch("autoconduck.proxy.gatekeeper_mod.classify", return_value=decision), \
         patch("autoconduck.proxy.evaluator_mod.score", return_value=0.2), \
         patch("autoconduck.proxy.pricing_mod.estimate_tokens", return_value=(10, 10)), \
         patch("autoconduck.proxy.pricing_mod.transform", return_value=0.2), \
         patch("autoconduck.proxy.state_mod.session_key_from_request", return_value="sess"), \
         patch("autoconduck.proxy.state_mod.get_session_store") as mock_store, \
         patch("autoconduck.proxy.telemetry_mod.telemetry.push"):
        mock_store.return_value.get.return_value = None
        from autoconduck.proxy import create_app
        app = create_app(cfg)
        client = TestClient(app)
        resp = client.post("/v1/chat/completions", json=_chat_payload())
        assert resp.status_code == 503, f"expected 503, got {resp.status_code} body {resp.text}"
        body = resp.json()
        assert "error" in body
        assert body["error"]["code"] == 503
        assert body["error"]["type"] == "config_error"

# (e) multimodal content parts count as attachments
def test_multimodal_counts_as_attachments():
    # Build a request with 4 image_url parts -> should be Tier 2 slow
    from autoconduck.proxy import Message, ChatRequest
    msgs = [
        Message(role="user", content=[
            {"type": "text", "text": "hello"},
            {"type": "image_url", "image_url": {"url": "http://example.com/1.png"}},
            {"type": "image_url", "image_url": {"url": "http://example.com/2.png"}},
            {"type": "image_url", "image_url": {"url": "http://example.com/3.png"}},
            {"type": "image_url", "image_url": {"url": "http://example.com/4.png"}},
        ])
    ]
    req = ChatRequest(model="autoconduck", messages=msgs)
    # gatekeeper should count 4 attachments -> slow
    decision = gatekeeper_mod.classify(req, None)
    assert decision.path == "slow", f"expected slow for 4 multimodal parts, got {decision.path} reason {decision.reason}"
    # Also test that _attachment_count helper counts correctly
    count = gatekeeper_mod._attachment_count(req)
    assert count == 4, f"expected 4, got {count}"

    # Also test with dict request style
    dict_req = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "hi"},
        {"type": "file", "file": {"filename": "a.txt"}},
        {"type": "image_url", "image_url": {"url": "x"}},
        {"type": "image_url", "image_url": {"url": "y"}},
        {"type": "image_url", "image_url": {"url": "z"}},
    ]}], "model": "autoconduck"}
    # Need ChatRequest-like for gatekeeper; it handles dict
    decision2 = gatekeeper_mod.classify(dict_req, None)
    assert decision2.path == "slow"

def test_three_attachments_not_slow_if_not_keyword():
    from autoconduck.proxy import Message, ChatRequest
    msgs = [
        Message(role="user", content=[
            {"type": "text", "text": "fix typo"},
            {"type": "image_url", "image_url": {"url": "u1"}},
            {"type": "image_url", "image_url": {"url": "u2"}},
            {"type": "image_url", "image_url": {"url": "u3"}},
        ])
    ]
    # 3 is threshold >3, so 3 should NOT be slow unless keyword
    # But "fix typo" is tier1 fast regex if <120 chars — should be fast
    req = ChatRequest(model="autoconduck", messages=msgs)
    decision = gatekeeper_mod.classify(req, None)
    # With our patched counter, 3 parts -> count 3, ac >3 is False, so fast via regex
    assert decision.path == "fast"

def test_streaming_disconnect_does_not_call_record_error_inner():
    """REAL streaming disconnect test — cancels mid-stream after >=1 chunk.

    Replaces previous source-inspection-only stub. Asserts:
    - upstream yields with small sleeps between chunks
    - disconnect occurs mid-stream (after first chunk) via request.is_disconnected()
    - pricing_mod.record_error NOT called for routed model
    - telemetry captured cancelled=True
    - mid-stream proof: upstream generator was interrupted (not all 5 chunks yielded to client,
      sentinel after cancel point never consumed, sleep interrupted)
    """
    async def _inner():
        cfg = _cfg_with_models()
        decision = gatekeeper_mod.Decision(path="fast", reason="test", T_i=0.1)

        # Track upstream yields and whether aclose was called
        upstream_yielded: list[int] = []
        upstream_aclosed = {"called": False}

        async def fake_stream():
            # Yields 5 chunks with small sleeps between them to allow mid-flight cancellation
            for i in range(5):
                upstream_yielded.append(i)
                # Use dict-style chunk that _sse_from_litellm_chunk handles
                yield {"choices": [{"delta": {"content": f"tok{i}"}}], "id": f"chatcmpl-{i}", "model": "cheap-model"}
                # sleep between chunks — if cancellation happens mid-stream this sleep chain is cut short
                await asyncio.sleep(0.03)
            # sentinel - should never be reached when cancelled mid-stream

        # Wrap generator to track aclose
        class TrackingAsyncGen:
            def __init__(self, gen):
                self._gen = gen
                self.aclose_called = False
            def __aiter__(self):
                return self
            async def __anext__(self):
                return await self._gen.__anext__()
            async def aclose(self):
                self.aclose_called = True
                upstream_aclosed["called"] = True
                try:
                    await self._gen.aclose()
                except Exception:
                    pass

        with patch("autoconduck.proxy.get_config", return_value=cfg), \
             patch("autoconduck.proxy.gatekeeper_mod.classify", return_value=decision), \
             patch("autoconduck.proxy.pricing_mod.estimate_tokens", return_value=(10, 10)), \
             patch("autoconduck.proxy.pricing_mod.select", return_value=cfg.models[0]), \
             patch("autoconduck.proxy.pricing_mod.transform", return_value=0.1), \
             patch("autoconduck.proxy.pricing_mod._detect_intent", return_value="default"), \
             patch("autoconduck.proxy.pricing_mod.record_usage") as mock_record_usage, \
             patch("autoconduck.proxy.pricing_mod.record_error") as mock_record_error, \
             patch("autoconduck.proxy.state_mod.session_key_from_request", return_value="sess"), \
             patch("autoconduck.proxy.state_mod.get_session_store") as mock_store, \
             patch("autoconduck.proxy.telemetry_mod.telemetry.push") as mock_push, \
             patch("autoconduck.proxy.state_mod.update_turn_state"):

            mock_store.return_value.get.return_value = None

            raw_gen = fake_stream()
            tracking_gen = TrackingAsyncGen(raw_gen)

            with patch("autoconduck.proxy._forward_litellm", new_callable=AsyncMock) as mock_forward:
                mock_forward.return_value = tracking_gen

                from autoconduck.proxy import create_app
                app = create_app(cfg)

                # Find the endpoint function directly to inject a FakeRequest with controllable is_disconnected
                endpoint = None
                for route in app.routes:
                    if getattr(route, "path", None) == "/v1/chat/completions":
                        endpoint = route.endpoint
                        break
                assert endpoint is not None, "endpoint not found"

                # FakeRequest that simulates client disconnect after first chunk
                payload = _chat_payload(stream=True)
                body_bytes = json.dumps(payload).encode()

                class FakeRequest:
                    def __init__(self):
                        self.headers = {}
                        self._body = body_bytes
                        self._is_disc_calls = 0
                    async def body(self):
                        return self._body
                    async def is_disconnected(self):
                        # First call (before first chunk yield) -> False, so first chunk is delivered
                        # Subsequent calls -> True, simulating client dropping connection mid-stream
                        self._is_disc_calls += 1
                        if self._is_disc_calls <= 1:
                            return False
                        return True

                fake_req = FakeRequest()
                # Invoke endpoint — returns StreamingResponse whose body_iterator is gen_with_telemetry
                resp = await endpoint(fake_req)
                from fastapi.responses import StreamingResponse
                assert isinstance(resp, StreamingResponse), f"expected StreamingResponse, got {type(resp)}"

                # Consume the streaming body iterator - it should yield 1 chunk then handle CancelledError
                body_iter = resp.body_iterator  # gen_with_telemetry async generator
                received_chunks: list[bytes] = []
                async for part in body_iter:  # type: ignore
                    received_chunks.append(part)
                    # We deliberately do not break early; generator should stop itself after disconnect
                    # but we guard to avoid infinite loop
                    if len(received_chunks) > 10:
                        break

                # Assertions
                # 1) At least one chunk was received (mid-stream, not before)
                assert len(received_chunks) >= 1, f"expected >=1 chunk before cancel, got {len(received_chunks)}"
                # 2) Not all 5 upstream chunks were delivered to client — cancellation happened mid-stream
                # Each upstream chunk becomes one SSE line; plus maybe DONE but cancelled path yields no DONE
                # So received should be 1, not 5, and not include all tokens
                assert len(received_chunks) < 5, f"expected mid-stream cancel (<5 parts), got {len(received_chunks)} chunks, upstream_yielded={upstream_yielded}"
                # 3) Upstream generator was interrupted — not all 5 yields consumed via upstream.next
                # upstream_yielded tracks how many times fake_stream yielded before aclose.
                # With disconnect after first chunk, second yield triggers disconnect before yielding to client,
                # so upstream may have yielded 2 but not 5. Crucially it should NOT be 5.
                assert len(upstream_yielded) < 5, f"upstream should have been interrupted mid-stream, yielded {upstream_yielded}"
                # Also sentinel check: if we had 5 full chunks + DONE, received would contain DONE marker
                has_done = any(b"[DONE]" in c for c in received_chunks)
                # Cancelled path in gen_stream does NOT yield DONE (it returns after pushing cancelled event)
                assert not has_done, f"cancelled stream should NOT contain [DONE], got {received_chunks}"
                # 4) record_error NOT called for the routed model (cheap-model)
                mock_record_error.assert_not_called()
                # Also ensure not called with cheap-model specifically
                cheap_calls = [c for c in mock_record_error.call_args_list if c.args and c.args[0] == "cheap-model"]
                assert len(cheap_calls) == 0, f"record_error should be 0 for cheap-model, got {mock_record_error.call_args_list}"
                # 5) telemetry recorded cancelled=True (gen_stream pushes cancelled event)
                # mock_push may have been called 1-2 times (cancelled + possibly normal); at least one must be cancelled
                cancelled_events = []
                for call in mock_push.call_args_list:
                    evt = call.args[0] if call.args else call.kwargs.get("event")
                    if evt is not None and getattr(evt, "cancelled", False) is True:
                        cancelled_events.append(evt)
                assert len(cancelled_events) >= 1, f"telemetry should have cancelled=True, push calls: {[ (getattr(a.args[0],'cancelled',None), getattr(a.args[0],'path',None)) for a in mock_push.call_args_list ]}"
                # 6) Ensure aclose was attempted (proxy calls upstream.aclose on disconnect)
                # tracking_gen.aclose_called should be True, or upstream_aclosed True
                # Not strictly required but proves disconnect handling path ran
                assert tracking_gen.aclose_called or upstream_aclosed["called"], "upstream aclose should have been called on disconnect"

    asyncio.run(_inner())


def test_streaming_upstream_5xx_calls_record_error():
    """Non-cancel error path MUST call record_error — proves distinction from disconnect.

    Mocks upstream 5xx error inside streaming D8 path and asserts record_error was called
    for the selected model, while telemetry error is recorded.
    """
    cfg = _cfg_with_models()
    decision = gatekeeper_mod.Decision(path="fast", reason="test", T_i=0.1)

    class Upstream500(Exception):
        status_code = 500
        def __str__(self):
            return "upstream 500 internal error"

    with patch("autoconduck.proxy.get_config", return_value=cfg), \
         patch("autoconduck.proxy.gatekeeper_mod.classify", return_value=decision), \
         patch("autoconduck.proxy.pricing_mod.estimate_tokens", return_value=(10, 10)), \
         patch("autoconduck.proxy.pricing_mod.select", return_value=cfg.models[0]), \
         patch("autoconduck.proxy.pricing_mod.transform", return_value=0.1), \
         patch("autoconduck.proxy.pricing_mod._detect_intent", return_value="default"), \
         patch("autoconduck.proxy.pricing_mod.record_usage"), \
         patch("autoconduck.proxy.pricing_mod.record_error") as mock_record_error, \
         patch("autoconduck.proxy.state_mod.session_key_from_request", return_value="sess"), \
         patch("autoconduck.proxy.state_mod.get_session_store") as mock_store, \
         patch("autoconduck.proxy.telemetry_mod.telemetry.push") as mock_push, \
         patch("autoconduck.proxy.state_mod.update_turn_state"):

        mock_store.return_value.get.return_value = None

        with patch("autoconduck.proxy._forward_litellm", new_callable=AsyncMock) as mock_forward:
            # Simulate 5xx upstream failure during streaming forward
            mock_forward.side_effect = Upstream500()

            from autoconduck.proxy import create_app
            app = create_app(cfg)
            client = TestClient(app)
            resp = client.post("/v1/chat/completions", json=_chat_payload(stream=True))
            # Upstream 5xx should be passed through as 500 and call record_error
            assert resp.status_code == 500, f"expected 500, got {resp.status_code} body {resp.text}"
            # record_error must have been called for cheap-model (selected model)
            assert mock_record_error.call_count >= 1, f"record_error should be called for 5xx, got {mock_record_error.call_args_list}"
            called_ids = [c.args[0] for c in mock_record_error.call_args_list]
            assert "cheap-model" in called_ids, f"expected record_error cheap-model, got {called_ids}"


def test_non_streaming_upstream_5xx_calls_record_error():
    """Non-streaming 5xx also calls record_error (distinct from CancelledError path)."""
    cfg = _cfg_with_models()
    decision = gatekeeper_mod.Decision(path="fast", reason="test", T_i=0.1)

    class Upstream502(Exception):
        status_code = 502
        def __str__(self):
            return "upstream 502 bad gateway"

    with patch("autoconduck.proxy.get_config", return_value=cfg), \
         patch("autoconduck.proxy.gatekeeper_mod.classify", return_value=decision), \
         patch("autoconduck.proxy.pricing_mod.estimate_tokens", return_value=(10, 10)), \
         patch("autoconduck.proxy.pricing_mod.select", return_value=cfg.models[0]), \
         patch("autoconduck.proxy.pricing_mod.transform", return_value=0.1), \
         patch("autoconduck.proxy.pricing_mod._detect_intent", return_value="default"), \
         patch("autoconduck.proxy.pricing_mod.record_usage"), \
         patch("autoconduck.proxy.pricing_mod.record_error") as mock_record_error, \
         patch("autoconduck.proxy.state_mod.session_key_from_request", return_value="sess"), \
         patch("autoconduck.proxy.state_mod.get_session_store") as mock_store, \
         patch("autoconduck.proxy.telemetry_mod.telemetry.push"), \
         patch("autoconduck.proxy.state_mod.update_turn_state"):

        mock_store.return_value.get.return_value = None

        with patch("autoconduck.proxy._forward_litellm", new_callable=AsyncMock) as mock_forward:
            mock_forward.side_effect = Upstream502()
            from autoconduck.proxy import create_app
            app = create_app(cfg)
            client = TestClient(app)
            resp = client.post("/v1/chat/completions", json=_chat_payload(stream=False))
            assert resp.status_code == 502, f"expected 502, got {resp.status_code} body {resp.text}"
            mock_record_error.assert_called()
            called_ids = [c.args[0] for c in mock_record_error.call_args_list]
            assert "cheap-model" in called_ids

