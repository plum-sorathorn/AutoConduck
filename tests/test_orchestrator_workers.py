import asyncio
import json
from unittest.mock import patch

import autoconduck.orchestrator as orch_mod
from autoconduck.orchestrator import Orchestrator

PLAN_JSON_4 = json.dumps({
    "tasks": [
        {"id": "t1", "goal": "do t1", "files": [], "depends_on": [], "output_contract": "c1"},
        {"id": "t2", "goal": "do t2", "files": [], "depends_on": [], "output_contract": "c2"},
        {"id": "t3", "goal": "do t3", "files": [], "depends_on": [], "output_contract": "c3"},
        {"id": "t4", "goal": "do t4", "files": [], "depends_on": [], "output_contract": "c4"},
    ]
})

PLAN_JSON_6 = json.dumps({
    "tasks": [
        {"id": f"t{i}", "goal": f"do t{i}", "files": [], "depends_on": [], "output_contract": f"c{i}"}
        for i in range(1, 7)
    ]
})


def _make_request():
    return {"messages": [{"role": "user", "content": "build something"}]}


def test_timeout_unchanged():
    assert orch_mod.WORKER_TIMEOUT_S == 30
    assert orch_mod.MAX_WORKERS == 4  # fallback constant unchanged


def test_honors_max_workers():
    """Semaphore must limit concurrent worker starts to max_workers=2."""
    async def _inner():
        orch = Orchestrator()
        concurrent = 0
        max_seen = 0

        async def fake_call(model, messages, **kwargs):
            nonlocal concurrent, max_seen
            if model == "plan-model":
                return PLAN_JSON_4
            # worker
            concurrent += 1
            max_seen = max(max_seen, concurrent)
            await asyncio.sleep(0.05)
            concurrent -= 1
            return f"result-{concurrent}"

        orch._call_llm = fake_call  # type: ignore[method-assign]

        res = await orch.plan_and_execute(
            _make_request(),
            plan_model_id="plan-model",
            worker_model_id="worker-model",
            max_workers=2,
        )
        assert res.worker_ok == 4
        assert res.worker_fail == 0
        assert res.degraded_to_fast is False
        # With 4 tasks and limit 2, peak concurrency should be exactly 2
        assert max_seen == 2, f"expected max concurrency 2, got {max_seen}"
        assert concurrent == 0

    asyncio.run(_inner())


def test_backward_compatibility_falls_back_to_config_default():
    """Calling without max_workers must not raise and must use get_config().max_workers."""
    async def _inner():
        orch = Orchestrator()
        concurrent = 0
        max_seen = 0

        async def fake_call(model, messages, **kwargs):
            nonlocal concurrent, max_seen
            if model == "plan-model":
                return PLAN_JSON_4
            concurrent += 1
            max_seen = max(max_seen, concurrent)
            await asyncio.sleep(0.05)
            concurrent -= 1
            return "ok"

        orch._call_llm = fake_call  # type: ignore[method-assign]

        # Patch get_config to return max_workers=2 so we can observe the fallback
        class FakeConfig:
            max_workers = 2

        with patch("autoconduck.config.get_config", return_value=FakeConfig()):
            # Do NOT pass max_workers — must be backward compatible
            res = await orch.plan_and_execute(
                _make_request(),
                plan_model_id="plan-model",
                worker_model_id="worker-model",
            )
        assert res.worker_ok == 4
        assert max_seen == 2, f"fallback to config max_workers=2 failed, max_seen={max_seen}"

        # Also verify explicit max_workers overrides config
        concurrent = 0
        max_seen = 0
        with patch("autoconduck.config.get_config", return_value=FakeConfig()):
            res2 = await orch.plan_and_execute(
                _make_request(),
                plan_model_id="plan-model",
                worker_model_id="worker-model",
                max_workers=1,
            )
        assert res2.worker_ok == 4
        assert max_seen == 1, f"explicit max_workers=1 should override config, got {max_seen}"

    asyncio.run(_inner())


def test_semaphore_uses_configured_value():
    """Directly assert asyncio.Semaphore is constructed with config value."""
    async def _inner():
        orch = Orchestrator()

        async def fake_call(model, messages, **kwargs):
            if model == "plan-model":
                return PLAN_JSON_4
            return "ok"

        orch._call_llm = fake_call  # type: ignore[method-assign]

        captured = {}

        orig_sem = asyncio.Semaphore

        class CapturingSemaphore(orig_sem):  # type: ignore[type-arg]
            def __init__(self, value=1):
                captured["value"] = value
                super().__init__(value)

        class FakeConfig:
            max_workers = 3

        with patch("autoconduck.config.get_config", return_value=FakeConfig()):
            with patch("asyncio.Semaphore", CapturingSemaphore):
                await orch.plan_and_execute(
                    _make_request(),
                    plan_model_id="plan-model",
                    worker_model_id="worker-model",
                )
        assert captured["value"] == 3

        # explicit arg should win
        captured.clear()
        with patch("asyncio.Semaphore", CapturingSemaphore):
            await orch.plan_and_execute(
                _make_request(),
                plan_model_id="plan-model",
                worker_model_id="worker-model",
                max_workers=2,
            )
        assert captured["value"] == 2

    asyncio.run(_inner())
