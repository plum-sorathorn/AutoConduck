import asyncio
import json
import os
import tempfile
import unittest.mock as mock
from unittest.mock import patch, AsyncMock

import autoconduck.orchestrator as orch_mod
from autoconduck.orchestrator import Orchestrator
from autoconduck.config import OrchestrationSettings

def _make_request(content="build something"):
    return {"messages": [{"role": "user", "content": content}]}

# Helper to make plan JSON with custom tasks
def _plan_json(tasks):
    return json.dumps({"tasks": tasks, "global_context": "", "summary": "test summary"})

PLAN_TWO_TASKS_SIMPLE = _plan_json([
    {"id": "t1", "goal": "do t1", "files": [], "depends_on": [], "output_contract": "c1"},
    {"id": "t2", "goal": "do t2", "files": [], "depends_on": [], "output_contract": "c2"},
])

def test_round_loop_adapts_on_failure():
    """Worker returns bad output round1 (file_exists fails), good round2 -> 2 rounds, feedback injected."""
    async def _inner():
        with tempfile.TemporaryDirectory() as tmpdir:
            orig_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                # file that will be checked - initially absent
                check_path = os.path.join(tmpdir, "expected.txt")
                # plan with one task requiring file_exists, another trivial
                plan_tasks = [
                    {"id": "t1", "goal": "create file", "files": [], "depends_on": [], "output_contract": "c1",
                     "acceptance": [{"kind": "file_exists", "path": check_path, "desc": "file must exist"}]},
                    {"id": "t2", "goal": "do t2", "files": [], "depends_on": [], "output_contract": "c2"},
                ]
                plan_json = _plan_json(plan_tasks)

                orch = Orchestrator()
                calls = []

                async def fake_call(model, messages, **kwargs):
                    calls.append((model, messages, kwargs))
                    if model == "plan-model":
                        return plan_json
                    # worker calls - messages is list[dict]
                    # Determine round by counting previous worker calls for t1
                    worker_call_count = sum(1 for m, _, _ in calls if m != "plan-model")
                    # On second worker call for t1 (round 2), create the file so verifier passes
                    # We need to detect which task this is - inspect messages content for TASK id
                    content = ""
                    for msg in messages:
                        content += str(msg.get("content", "")) + " "
                    # if this is t1 second round, create file
                    if "t1" in content and worker_call_count >= 3:
                        # second round for t1 occurs after first round's 2 workers + maybe t2 pass
                        # ensure file exists
                        with open(check_path, "w") as f:
                            f.write("hello")
                        return "x" * 50 + " good output round 2 with file"
                    if "t1" in content:
                        return "bad short"  # will be verified via file_exists -> fails (file not exist)
                    else:
                        return "y" * 50 + " good t2 output"

                orch._call_llm = fake_call  # type: ignore
                # also patch _call_llm_with_tools to same (since worker_tools=False, not used)
                orch._call_llm_with_tools = fake_call  # type: ignore

                settings = OrchestrationSettings(exploration=False, max_rounds=2, verifier="rule", worker_tools=False)

                res = await orch.plan_and_execute(
                    _make_request(),
                    plan_model_id="plan-model",
                    worker_model_id="worker-model",
                    orch_settings=settings,
                )
                assert res.rounds_used == 2, f"expected 2 rounds, got {res.rounds_used}"
                assert "t1" in res.accepted_tasks, f"t1 should be accepted {res.accepted_tasks} {res.partial_tasks}"
                assert "t2" in res.accepted_tasks
                # feedback injected: second round's messages should contain failure feedback
                # Find the call for t1 round 2 and check its messages contain feedback substring like "not found" or "file_exists"
                t1_round2_calls = []
                for model, msgs, kw in calls:
                    if model == "plan-model":
                        continue
                    txt = " ".join(str(m.get("content","")) for m in msgs)
                    if "t1" in txt and ("Feedback" in txt or "feedback" in txt or "not found" in txt):
                        t1_round2_calls.append(txt)
                assert len(t1_round2_calls) >= 1, f"feedback not injected in followup: calls={calls}"
                # also round_history non-empty
                assert "PASS" in res.round_history
            finally:
                os.chdir(orig_cwd)
    asyncio.run(_inner())

def test_escalation_ladder_used():
    async def _inner():
        with tempfile.TemporaryDirectory() as tmpdir:
            orig_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                check_path = os.path.join(tmpdir, "need.txt")
                plan_tasks = [
                    {"id": "t1", "goal": "task1", "files": [], "depends_on": [], "output_contract": "c1",
                     "acceptance": [{"kind": "file_exists", "path": check_path}]},
                    {"id": "t2", "goal": "task2", "files": [], "depends_on": [], "output_contract": "c2"},
                ]
                plan_json = _plan_json(plan_tasks)
                orch = Orchestrator()
                model_calls = []

                async def fake_call(model, messages, **kwargs):
                    model_calls.append(model)
                    if model == "plan-model":
                        return plan_json
                    # for worker: inspect content
                    content = "".join(str(m.get("content","")) for m in messages)
                    if "t1" in content:
                        # first round return bad, second round create file and return good
                        # count how many times t1 called
                        t1_count = sum(1 for m in model_calls if m != "plan-model" and model_calls.count(m) >=0)
                        # better to count worker calls
                        # we track via closure
                        # simple: if this is first t1 call, file doesn't exist
                        # if file exists, return good
                        if os.path.exists(check_path):
                            return "z" * 50 + " good"
                        # first call: don't create file
                        return "bad"
                    else:
                        return "a" * 50 + " ok t2"

                # Need to track call order more precisely
                call_log = []
                async def fake_call2(model, messages, **kwargs):
                    call_log.append((model, list(messages)))
                    if model == "plan-model":
                        return plan_json
                    content = "".join(str(m.get("content","")) for m in messages)
                    if "t1" in content:
                        # Check if this is second round (presence of Feedback) -> create file
                        if any("Feedback" in str(m.get("content","")) or "feedback" in str(m.get("content","")).lower() for m in messages):
                            with open(check_path, "w") as f:
                                f.write("x")
                            return "x" * 50 + " good t1 round2"
                        return "bad t1 round1"
                    else:
                        return "y" * 50 + " good t2"

                orch._call_llm = fake_call2  # type: ignore
                orch._call_llm_with_tools = fake_call2  # type: ignore

                settings = OrchestrationSettings(exploration=False, max_rounds=2, verifier="rule", worker_tools=False)
                res = await orch.plan_and_execute(
                    _make_request(),
                    plan_model_id="plan-model",
                    worker_model_id="m1",
                    worker_model_ladder=["m1", "m2", "m3"],
                    orch_settings=settings,
                )
                # inspect call_log worker models
                worker_models = [m for m, _ in call_log if m != "plan-model"]
                # Expect at least 3 calls: t1 round1 m1, t2 round1 m1, t1 round2 m2
                assert worker_models[0] == "m1", f"round1 should be m1 got {worker_models}"
                # Find t1 round2 model
                # t1 round2 is the last worker call if escalation worked
                # Determine which models were used for t1
                t1_models = []
                for mod, msgs in call_log:
                    if mod == "plan-model":
                        continue
                    txt = "".join(str(m.get("content","")) for m in msgs)
                    if "t1" in txt:
                        t1_models.append(mod)
                assert len(t1_models) == 2, f"expected 2 t1 calls got {t1_models}"
                assert t1_models[0] == "m1" and t1_models[1] == "m2", f"escalation failed {t1_models}"
                assert res.rounds_used == 2
            finally:
                os.chdir(orig_cwd)
    asyncio.run(_inner())

def test_early_stop_all_accepted():
    async def _inner():
        orch = Orchestrator()
        async def fake_call(model, messages, **kwargs):
            if model == "plan-model":
                return PLAN_TWO_TASKS_SIMPLE
            return "x" * 50 + " good output that passes length check"
        orch._call_llm = fake_call  # type: ignore
        orch._call_llm_with_tools = fake_call  # type: ignore
        settings = OrchestrationSettings(exploration=False, max_rounds=4, verifier="rule", worker_tools=False)
        res = await orch.plan_and_execute(_make_request(), plan_model_id="plan-model", worker_model_id="worker-model", orch_settings=settings)
        assert res.rounds_used == 1, f"expected 1 got {res.rounds_used}"
        assert len(res.accepted_tasks) == 2
        assert len(res.partial_tasks) == 0
        # no round 2 calls beyond initial 2 workers => total worker calls =2
    asyncio.run(_inner())

def test_partial_when_rounds_exhausted():
    async def _inner():
        with tempfile.TemporaryDirectory() as tmpdir:
            orig_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                missing = os.path.join(tmpdir, "missing.txt")
                plan_tasks = [
                    {"id": "t1", "goal": "fail", "files": [], "depends_on": [], "output_contract": "c1",
                     "acceptance": [{"kind": "file_exists", "path": missing}]},
                    {"id": "t2", "goal": "pass", "files": [], "depends_on": [], "output_contract": "c2"},
                ]
                plan_json = _plan_json(plan_tasks)
                orch = Orchestrator()
                async def fake_call(model, messages, **kwargs):
                    if model == "plan-model":
                        return plan_json
                    content = "".join(str(m.get("content","")) for m in messages)
                    if "t1" in content:
                        return "bad output"
                    return "y" * 50 + " good t2"
                orch._call_llm = fake_call  # type: ignore
                orch._call_llm_with_tools = fake_call  # type: ignore
                settings = OrchestrationSettings(exploration=False, max_rounds=2, verifier="rule", worker_tools=False)
                res = await orch.plan_and_execute(_make_request(), plan_model_id="plan-model", worker_model_id="w", orch_settings=settings)
                assert "t1" in res.partial_tasks, f"t1 should be partial {res.partial_tasks}"
                assert "PARTIAL" in res.round_history
                assert res.rounds_used == 2
            finally:
                os.chdir(orig_cwd)
    asyncio.run(_inner())

def test_rule_verifier_file_exists():
    async def _inner():
        with tempfile.TemporaryDirectory() as tmpdir:
            orig_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                real_file = os.path.join(tmpdir, "real.txt")
                with open(real_file, "w") as f:
                    f.write("hello")
                missing = os.path.join(tmpdir, "nope.txt")
                plan_tasks_pass = [
                    {"id": "t1", "goal": "check exists", "files": [], "depends_on": [], "output_contract": "c1",
                     "acceptance": [{"kind": "file_exists", "path": real_file}]},
                    {"id": "t2", "goal": "dummy", "files": [], "depends_on": [], "output_contract": "c2"},
                ]
                plan_tasks_fail = [
                    {"id": "t1", "goal": "check missing", "files": [], "depends_on": [], "output_contract": "c1",
                     "acceptance": [{"kind": "file_exists", "path": missing}]},
                    {"id": "t2", "goal": "dummy", "files": [], "depends_on": [], "output_contract": "c2"},
                ]
                # PASS case
                orch = Orchestrator()
                async def fake_call_pass(model, messages, **kwargs):
                    if model == "plan-model":
                        return _plan_json(plan_tasks_pass)
                    return "x" * 50 + " output"
                orch._call_llm = fake_call_pass  # type: ignore
                orch._call_llm_with_tools = fake_call_pass  # type: ignore
                settings = OrchestrationSettings(exploration=False, max_rounds=1, verifier="rule", worker_tools=False)
                res = await orch.plan_and_execute(_make_request(), plan_model_id="plan-model", worker_model_id="w", orch_settings=settings)
                assert "t1" in res.accepted_tasks, f"should PASS when file exists {res.round_history} {res.partial_tasks}"

                # FAIL case
                orch2 = Orchestrator()
                async def fake_call_fail(model, messages, **kwargs):
                    if model == "plan-model":
                        return _plan_json(plan_tasks_fail)
                    return "x" * 50 + " output"
                orch2._call_llm = fake_call_fail  # type: ignore
                orch2._call_llm_with_tools = fake_call_fail  # type: ignore
                res2 = await orch2.plan_and_execute(_make_request(), plan_model_id="plan-model", worker_model_id="w", orch_settings=settings)
                assert "t1" in res2.partial_tasks, f"should be partial when file missing {res2.accepted_tasks}"
                assert missing in res2.round_history or "not found" in res2.round_history.lower(), f"feedback should mention path {res2.round_history}"
            finally:
                os.chdir(orig_cwd)
    asyncio.run(_inner())

def test_llm_verdict_called_when_verifier_llm():
    async def _inner():
        plan_tasks = [
            {"id": "t1", "goal": "llm check", "files": [], "depends_on": [], "output_contract": "c1",
             "acceptance": [{"kind": "llm", "desc": "is good?"}]},
            {"id": "t2", "goal": "dummy", "files": [], "depends_on": [], "output_contract": "c2",
             "acceptance": [{"kind": "llm", "desc": "is good?"}]},
        ]
        # PASS verdict
        orch = Orchestrator()
        verdict_calls = []
        async def fake_call(model, messages, **kwargs):
            # plan call
            if any("task decomposer" in str(m.get("content","")).lower() for m in messages) or any(m.get("role")=="system" and "decomposer" in str(m.get("content","")).lower() for m in messages):
                return _plan_json(plan_tasks)
            # check if this is verifier call (system says You judge)
            txt = " ".join(str(m.get("content","")) for m in messages)
            if "You judge" in txt:
                verdict_calls.append(txt)
                return "PASS looks good"
            # worker call
            if model == "plan-model":
                return _plan_json(plan_tasks)
            return "x" * 50 + " worker output"
        # Need to distinguish plan vs verifier vs worker - use model id?
        # Our fake is called with model=plan-model for plan, and verifier_model for verdict
        # We'll inspect messages content for "You judge"
        orch._call_llm = fake_call  # type: ignore
        orch._call_llm_with_tools = fake_call  # type: ignore
        settings = OrchestrationSettings(exploration=False, max_rounds=1, verifier="llm", worker_tools=False)
        res = await orch.plan_and_execute(_make_request(), plan_model_id="plan-model", worker_model_id="w", orch_settings=settings, verifier_model_id="verifier-model")
        assert len(verdict_calls) >= 1, "verifier LLM should have been called"
        assert "t1" in res.accepted_tasks

        # FAIL verdict
        orch2 = Orchestrator()
        verdict_calls2 = []
        async def fake_call2(model, messages, **kwargs):
            txt = " ".join(str(m.get("content","")) for m in messages)
            if "You judge" in txt:
                verdict_calls2.append(txt)
                return "FAIL missing requirement X"
            if model == "plan-model":
                return _plan_json(plan_tasks)
            # also check system decomposer path via model check? fallback
            if any("decomposer" in str(m.get("content","")).lower() for m in messages):
                return _plan_json(plan_tasks)
            return "x" * 50 + " worker output"
        orch2._call_llm = fake_call2  # type: ignore
        orch2._call_llm_with_tools = fake_call2  # type: ignore
        # Need to handle plan detection: the first call will be plan, which our fake will treat as verifier? Fix:
        # In this test, plan call also goes through _call_llm with messages containing PLANNER_SYSTEM
        # We check for "task decomposer" to return plan
        async def fake_call2_fixed(model, messages, **kwargs):
            txt = " ".join(str(m.get("content","")) for m in messages)
            if "You judge" in txt:
                verdict_calls2.append(txt)
                return "FAIL missing requirement X"
            if "task decomposer" in txt.lower() or "return only valid json" in txt.lower():
                return _plan_json(plan_tasks)
            if model == "plan-model":
                return _plan_json(plan_tasks)
            return "x" * 50 + " worker output"
        orch2._call_llm = fake_call2_fixed  # type: ignore
        orch2._call_llm_with_tools = fake_call2_fixed  # type: ignore
        res2 = await orch2.plan_and_execute(_make_request(), plan_model_id="plan-model", worker_model_id="w", orch_settings=settings, verifier_model_id="verifier-model")
        assert "t1" in res2.partial_tasks, f"should be partial on FAIL verdict {res2.accepted_tasks} {res2.round_history}"
        assert "missing requirement" in res2.round_history.lower() or "FAIL" in res2.round_history

    asyncio.run(_inner())

def test_tool_execution_sandbox():
    async def _inner():
        with tempfile.TemporaryDirectory() as tmpdir:
            orig_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                # create a file for grep/glob
                with open(os.path.join(tmpdir, "hello.py"), "w") as f:
                    f.write("def hello():\n    return 'world'\n")
                with open(os.path.join(tmpdir, "other.txt"), "w") as f:
                    f.write("grep_target_line 123")
                # sandbox check: read_file outside cwd should deny
                res = await orch_mod._execute_tool("read_file", {"path": "../secret.txt"})
                assert "access denied" in res.lower(), f"got {res}"
                # grep should find match
                res2 = await orch_mod._execute_tool("grep", {"pattern": "grep_target", "path": ".", "max_matches": 10})
                assert "grep_target" in res2, f"grep failed {res2}"
                # glob should return paths
                res3 = await orch_mod._execute_tool("glob", {"pattern": "**/*.py"})
                assert ".py" in res3 or "hello.py" in res3, f"glob failed {res3}"
                # also unknown tool
                res4 = await orch_mod._execute_tool("unknown_tool", {})
                assert "unknown tool" in res4.lower()
            finally:
                os.chdir(orig_cwd)
    asyncio.run(_inner())

def test_compaction_includes_plan_summary_and_status():
    async def _inner():
        with tempfile.TemporaryDirectory() as tmpdir:
            orig_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                missing = os.path.join(tmpdir, "missing2.txt")
                plan_tasks = [
                    {"id": "t1", "goal": "pass task", "files": [], "depends_on": [], "output_contract": "c1"},
                    {"id": "t2", "goal": "fail task", "files": [], "depends_on": [], "output_contract": "c2",
                     "acceptance": [{"kind": "file_exists", "path": missing}]},
                ]
                plan_json = _plan_json(plan_tasks)
                orch = Orchestrator()
                async def fake_call(model, messages, **kwargs):
                    if model == "plan-model":
                        return plan_json
                    content = "".join(str(m.get("content","")) for m in messages)
                    if "t1" in content:
                        return "a" * 50 + " good t1 output with plenty of chars"
                    return "bad t2"
                orch._call_llm = fake_call  # type: ignore
                orch._call_llm_with_tools = fake_call  # type: ignore
                settings = OrchestrationSettings(exploration=False, max_rounds=1, verifier="rule", worker_tools=False)
                res = await orch.plan_and_execute(_make_request(), plan_model_id="plan-model", worker_model_id="w", orch_settings=settings)
                assert res.compacted_context is not None
                assert "t1" in res.compacted_context
                assert "t2" in res.compacted_context
                assert "PASS" in res.compacted_context or "PARTIAL" in res.compacted_context
                assert res.round_history != "", "round_history should be non-empty"
                assert "test summary" in res.compacted_context.lower() or "t1" in res.compacted_context
            finally:
                os.chdir(orig_cwd)
    asyncio.run(_inner())

# Direct invocation runner (pytest may be absent)
if __name__ == "__main__":
    tests = [
        ("test_round_loop_adapts_on_failure", test_round_loop_adapts_on_failure),
        ("test_escalation_ladder_used", test_escalation_ladder_used),
        ("test_early_stop_all_accepted", test_early_stop_all_accepted),
        ("test_partial_when_rounds_exhausted", test_partial_when_rounds_exhausted),
        ("test_rule_verifier_file_exists", test_rule_verifier_file_exists),
        ("test_llm_verdict_called_when_verifier_llm", test_llm_verdict_called_when_verifier_llm),
        ("test_tool_execution_sandbox", test_tool_execution_sandbox),
        ("test_compaction_includes_plan_summary_and_status", test_compaction_includes_plan_summary_and_status),
    ]
    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
            passed += 1
        except Exception as e:
            import traceback
            print(f"FAIL {name}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\nTotal: {passed} passed, {failed} failed out of {len(tests)}")
    import sys
    sys.exit(0 if failed == 0 else 1)
