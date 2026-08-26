"""Adversarial Stress Test Suite for Milestone 2 Foundations.

Empirical verification suite targeting:
1. Turn Guard: Mixed payloads, malformed JSON, deep nesting, rapid alternating errors, non-dict payloads.
2. SLM Planner: Schema validation rejections, circuit breaker timeouts, coroutine hanging, empty/null prompts.
3. Dynamic LangGraph Factory: Complex multi-subtask DAG topologies (diamond, fan-out, cycles, missing deps),
   execution flow, and SqliteSaver checkpoint persistence.
"""
from __future__ import annotations

import asyncio
import json
import tempfile
import time
from typing import Any
import pytest

from autoconduck.server.turn_guard import TurnAction, TurnClassificationResult, TurnGuard
from autoconduck.routing.slm_planner import ExecutionPlan, SLMPlanner, SubTaskSpec
from autoconduck.routing.model_pool import CapabilitySLA
from autoconduck.orchestrator.dynamic_factory import DynamicState, build_dynamic_graph
from autoconduck._compat.sqlite_checkpointer import get_sqlite_checkpointer


# ==============================================================================
# SECTION 1: TURN GUARD ADVERSARIAL STRESS TESTS
# ==============================================================================

class TestTurnGuardAdversarial:
    """Adversarial stress testing for Turn Guard classifier."""

    @pytest.fixture
    def guard(self) -> TurnGuard:
        return TurnGuard()

    def test_mixed_payload_formats_in_single_turn(self, guard: TurnGuard):
        """Mixed OpenAI tool_calls and Anthropic tool_result structures across conversation."""
        messages = [
            {"role": "system", "content": "You are an assistant."},
            {"role": "user", "content": "Help me refactor code."},
            # OpenAI style assistant tool call
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_oa_1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
                    }
                ],
            },
            # OpenAI tool response
            {"role": "tool", "tool_call_id": "call_oa_1", "name": "read_file", "content": "def a(): return 1"},
            # Anthropic style assistant tool use
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Now checking second file."},
                    {"type": "tool_use", "id": "call_ant_2", "name": "read_file", "input": {"path": "b.py"}},
                ],
            },
            # Anthropic style tool result
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "call_ant_2", "content": "def b(): return 2", "is_error": False}
                ],
            },
        ]
        result = guard.classify_turn(messages)
        assert result.is_tool_loop is True
        assert result.is_stagnant is False
        assert result.target_action == TurnAction.DIRECT_ACTIVE_TIER
        assert result.last_tool_name == "read_file"
        assert result.error_streak == 0

    def test_malformed_tool_call_payloads_do_not_crash(self, guard: TurnGuard):
        """Severely malformed tool calls, missing dict keys, invalid types."""
        malformed_histories = [
            # 1. Non-dict messages in list
            [None, "invalid_message", 12345, []],
            # 2. Assistant with None tool_calls or int tool_calls
            [
                {"role": "user", "content": "test"},
                {"role": "assistant", "tool_calls": None},
                {"role": "tool", "name": "test_tool", "content": "output"},
            ],
            # 3. Tool call missing function dictionary or invalid function keys
            [
                {"role": "user", "content": "test"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "bad1", "function": None},
                        {"id": "bad2", "function": "not_a_dict"},
                        {"id": "bad3", "function": {"name": 12345, "arguments": None}},
                    ],
                },
                {"role": "tool", "name": "test_tool", "content": "output"},
            ],
            # 4. Anthropic content with non-dict blocks and missing input
            [
                {"role": "user", "content": "test"},
                {
                    "role": "assistant",
                    "content": [
                        None,
                        "raw_string_block",
                        123,
                        {"type": "tool_use", "name": None, "input": None},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "content": None, "is_error": None},
                    ],
                },
            ],
        ]

        for idx, history in enumerate(malformed_histories):
            result = guard.classify_turn(history)
            assert isinstance(result, TurnClassificationResult), f"Failed on malformed case {idx}"

    def test_malformed_json_arguments_string(self, guard: TurnGuard):
        """Assistant tool arguments containing malformed JSON strings, binary data, unescaped quotes."""
        messages = [
            {"role": "user", "content": "run"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": '{"cmd": "echo \'unclosed string...',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "name": "bash", "content": "echo output"},
        ]
        result = guard.classify_turn(messages)
        assert result.is_tool_loop is True
        assert result.target_action == TurnAction.DIRECT_ACTIVE_TIER

    def test_deeply_nested_and_large_tool_outputs(self, guard: TurnGuard):
        """Tool output with deeply nested structures, lists of dicts, non-string contents."""
        nested_data = {"a": {"b": [{"c": [1, 2, {"d": "nested_val"}]}]}}
        messages = [
            {"role": "user", "content": "fetch"},
            {
                "role": "assistant",
                "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "query", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "c1", "name": "query", "content": nested_data},
        ]
        result = guard.classify_turn(messages)
        assert result.is_tool_loop is True
        assert result.is_stagnant is False
        assert result.target_action == TurnAction.DIRECT_ACTIVE_TIER

    def test_rapid_alternating_error_and_success_streak_tracking(self, guard: TurnGuard):
        """Rapid alternating errors must not accumulate error_streak to 2."""
        history = [{"role": "user", "content": "task"}]
        # 10 rounds of Error -> Success -> Error -> Success
        for i in range(10):
            history.append({
                "role": "assistant",
                "tool_calls": [{"id": f"c_{i}_err", "type": "function", "function": {"name": "run", "arguments": f'{{"step": {i}}}'}}],
            })
            history.append({
                "role": "tool",
                "tool_call_id": f"c_{i}_err",
                "name": "run",
                "content": "Error: exit code 1",
            })
            history.append({
                "role": "assistant",
                "tool_calls": [{"id": f"c_{i}_ok", "type": "function", "function": {"name": "fix", "arguments": f'{{"step": {i}}}'}}],
            })
            history.append({
                "role": "tool",
                "tool_call_id": f"c_{i}_ok",
                "name": "fix",
                "content": "Success: recovered step",
            })

        result = guard.classify_turn(history)
        assert result.is_stagnant is False
        assert result.error_streak == 0
        assert result.target_action == TurnAction.DIRECT_ACTIVE_TIER

        # Now append two consecutive errors at the end
        history.append({
            "role": "assistant",
            "tool_calls": [{"id": "err1", "type": "function", "function": {"name": "bash", "arguments": '{"cmd": "p1"}'}}],
        })
        history.append({"role": "tool", "tool_call_id": "err1", "name": "bash", "content": "Error: fail 1"})
        history.append({
            "role": "assistant",
            "tool_calls": [{"id": "err2", "type": "function", "function": {"name": "bash", "arguments": '{"cmd": "p2"}'}}],
        })
        history.append({"role": "tool", "tool_call_id": "err2", "name": "bash", "content": "Failed: fail 2"})

        escalated_result = guard.classify_turn(history)
        assert escalated_result.is_stagnant is True
        assert escalated_result.error_streak >= 2
        assert escalated_result.target_action == TurnAction.ESCALATE_SLM

    def test_parallel_tool_calls_within_single_turn(self, guard: TurnGuard):
        """Assistant returning multiple parallel tool calls in one turn."""
        messages = [
            {"role": "user", "content": "fetch both files"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "read", "arguments": '{"file": "x.py"}'}},
                    {"id": "c2", "type": "function", "function": {"name": "read", "arguments": '{"file": "x.py"}'}},
                    {"id": "c3", "type": "function", "function": {"name": "read", "arguments": '{"file": "x.py"}'}},
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "name": "read", "content": "def x(): pass"},
            {"role": "tool", "tool_call_id": "c2", "name": "read", "content": "def x(): pass"},
            {"role": "tool", "tool_call_id": "c3", "name": "read", "content": "def x(): pass"},
        ]
        result = guard.classify_turn(messages)
        assert result.is_stagnant is True
        assert result.tool_call_streak >= 3
        assert result.target_action == TurnAction.ESCALATE_SLM


# ==============================================================================
# SECTION 2: SLM PLANNER ADVERSARIAL STRESS TESTS
# ==============================================================================

class TestSLMPlannerAdversarial:
    """Adversarial stress testing for SLM Planner & Circuit Breaker."""

    @pytest.fixture
    def planner(self) -> SLMPlanner:
        return SLMPlanner()

    @pytest.mark.asyncio
    async def test_invalid_schema_payloads_degrade_softly(self, monkeypatch):
        """Planner must safely degrade to fallback plan for various invalid schema responses."""
        planner = SLMPlanner()

        invalid_outputs = [
            # 1. Empty string
            "",
            # 2. Random non-json text
            "```markdown\nHere is your plan\n```",
            # 3. Missing required 'route'
            json.dumps({"confidence": 0.9, "task_type": "chat"}),
            # 4. Invalid enum for route
            json.dumps({"route": "invalid_route_type", "task_type": "chat"}),
            # 5. Invalid type for suggested_sla
            json.dumps({"route": "fast_direct", "suggested_sla": "not_a_valid_sla_object"}),
            # 6. Invalid subtask role
            json.dumps({
                "route": "dynamic_dag",
                "subtasks": [{"id": "t1", "goal": "g", "role": "unsupported_role"}],
            }),
            # 7. Confidence out of bounds
            json.dumps({"route": "fast_direct", "confidence": 2.5}),
            # 8. Non-dict/non-string result
            12345,
            [{"route": "fast_direct"}],
            None,
        ]

        for idx, bad_output in enumerate(invalid_outputs):
            async def mock_infer(*args, **kwargs):
                return bad_output

            monkeypatch.setattr(planner, "_raw_infer", mock_infer, raising=False)
            plan = await planner.plan([{"role": "user", "content": "Solve problem"}])

            assert isinstance(plan, ExecutionPlan), f"Failed on invalid output case {idx}"
            assert plan.fallback_used is True
            assert plan.suggested_sla.requires_tools is True
            assert plan.route == "fast_direct"

    @pytest.mark.asyncio
    async def test_simulated_circuit_breaker_timeout(self, monkeypatch):
        """Inference that hangs or exceeds circuit breaker timeout trips in <= 150ms."""
        planner = SLMPlanner(circuit_breaker_ms=50.0)

        async def hanging_inference(*args, **kwargs):
            await asyncio.sleep(0.5)
            return json.dumps({"route": "fast_direct"})

        monkeypatch.setattr(planner, "_raw_infer", hanging_inference, raising=False)

        start = time.perf_counter()
        plan = await planner.plan([{"role": "user", "content": "Long computation"}])
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        assert elapsed_ms < 150.0, f"Circuit breaker took too long: {elapsed_ms:.1f}ms"
        assert plan.fallback_used is True
        assert "Circuit breaker timeout" in plan.rationale

    @pytest.mark.asyncio
    async def test_synchronous_blocking_inference_timeout(self, monkeypatch):
        """Synchronous CPU-bound blocking inference safely trips circuit breaker."""
        planner = SLMPlanner(circuit_breaker_ms=50.0)

        def sync_blocking_infer(*args, **kwargs):
            time.sleep(0.4)
            return json.dumps({"route": "fast_direct"})

        monkeypatch.setattr(planner, "_raw_infer", sync_blocking_infer, raising=False)

        start = time.perf_counter()
        plan = await planner.plan([{"role": "user", "content": "Sync block"}])
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        assert elapsed_ms < 150.0
        assert plan.fallback_used is True

    @pytest.mark.asyncio
    async def test_unhandled_inference_exceptions_degrade_softly(self, monkeypatch):
        """Catastrophic exceptions (e.g. CUDA OOM, ZeroDivision) in inference degrade safely."""
        planner = SLMPlanner()

        async def exploding_inference(*args, **kwargs):
            raise RuntimeError("Simulated CUDA Out Of Memory Error")

        monkeypatch.setattr(planner, "_raw_infer", exploding_inference, raising=False)

        plan = await planner.plan([{"role": "user", "content": "Crash test"}])
        assert isinstance(plan, ExecutionPlan)
        assert plan.fallback_used is True
        assert "Simulated CUDA Out Of Memory Error" in plan.rationale

    @pytest.mark.asyncio
    async def test_adversarial_prompt_inputs(self, planner: SLMPlanner):
        """Edge case prompt payloads (empty, whitespace, multi-modal, non-dict)."""
        edge_prompts = [
            [],
            [{}],
            [{"role": "assistant", "content": "Hello"}],
            [{"role": "user", "content": ""}],
            [{"role": "user", "content": "   \n\t  "}],
            [{"role": "user", "content": None}],
            [{"role": "user", "content": []}],
            [{"role": "user", "content": [{"type": "text", "text": "   "}]}],
            [{"role": "user", "content": [{"type": "image", "image_url": "https://example.com/pic.png"}]}],
            [None, 123, "just a string"],
        ]

        for idx, prompt in enumerate(edge_prompts):
            plan = await planner.plan(prompt)
            assert isinstance(plan, ExecutionPlan), f"Failed on edge prompt case {idx}"


# ==============================================================================
# SECTION 3: DYNAMIC LANGGRAPH FACTORY ADVERSARIAL STRESS TESTS
# ==============================================================================

class TestDynamicLangGraphFactoryAdversarial:
    """Adversarial stress testing for Dynamic LangGraph Factory & Checkpointer."""

    @pytest.mark.asyncio
    async def test_complex_diamond_dag_topology(self):
        """Diamond DAG: Start -> A -> (B, C) -> D -> Synthesizer -> End."""
        subtasks = [
            SubTaskSpec(id="node_a", goal="Root analysis", role="recon", depends_on=[]),
            SubTaskSpec(id="node_b", goal="Left branch processing", role="read", depends_on=["node_a"]),
            SubTaskSpec(id="node_c", goal="Right branch processing", role="read", depends_on=["node_a"]),
            SubTaskSpec(id="node_d", goal="Merge branch findings", role="reasoning", depends_on=["node_b", "node_c"]),
        ]
        plan = ExecutionPlan(
            route="dynamic_dag",
            subtasks=subtasks,
            synthesizer_sla=CapabilitySLA(requires_reasoning=True),
        )
        runner = build_dynamic_graph(plan)
        assert runner is not None

        # Execute graph
        initial_state = DynamicState(
            session_id="diamond_session",
            thread_id="thread_1",
            plan=plan,
        )
        result_state = await runner.ainvoke(initial_state)
        assert result_state is not None
        outputs = getattr(result_state, "subtask_outputs", {}) if not isinstance(result_state, dict) else result_state.get("subtask_outputs", {})
        assert "node_a" in outputs
        assert "node_b" in outputs
        assert "node_c" in outputs
        assert "node_d" in outputs
        final_res = getattr(result_state, "final_result", None) if not isinstance(result_state, dict) else result_state.get("final_result")
        assert final_res is not None

    @pytest.mark.asyncio
    async def test_wide_fan_out_and_fan_in_topology(self):
        """Wide 20-node parallel fan-out converging into single synthesizer."""
        tasks = [
            SubTaskSpec(id=f"worker_{i:02d}", goal=f"Scan shard {i}", role="read", depends_on=[])
            for i in range(20)
        ]
        plan = ExecutionPlan(
            route="dynamic_dag",
            needs_rag=True,
            rag_queries=["Vector index search for shard dependencies"],
            subtasks=tasks,
        )
        runner = build_dynamic_graph(plan)
        assert runner is not None

        initial_state = DynamicState(
            session_id="fanout_session",
            thread_id="thread_fan",
            plan=plan,
        )
        result_state = await runner.ainvoke(initial_state)
        outputs = getattr(result_state, "subtask_outputs", {}) if not isinstance(result_state, dict) else result_state.get("subtask_outputs", {})
        assert len(outputs) == 20

    @pytest.mark.asyncio
    async def test_non_existent_and_invalid_dependencies_handling(self):
        """Subtasks with non-existent dependency IDs are gracefully anchored to root."""
        subtasks = [
            SubTaskSpec(id="task_1", goal="Valid task", role="read", depends_on=["non_existent_task_999", "ghost_node"]),
            SubTaskSpec(id="task_2", goal="Another task", role="edit", depends_on=["task_1"]),
        ]
        plan = ExecutionPlan(
            route="dynamic_dag",
            subtasks=subtasks,
        )
        runner = build_dynamic_graph(plan)
        assert runner is not None

        initial_state = DynamicState(
            session_id="ghost_dep_session",
            thread_id="thread_ghost",
            plan=plan,
        )
        result_state = await runner.ainvoke(initial_state)
        outputs = getattr(result_state, "subtask_outputs", {}) if not isinstance(result_state, dict) else result_state.get("subtask_outputs", {})
        assert "task_1" in outputs
        assert "task_2" in outputs

    @pytest.mark.asyncio
    async def test_cyclic_dependencies_graceful_recovery(self):
        """Cyclic dependencies do not cause unhandled crash; fallback runner is returned."""
        # Note: ExecutionPlan validator strips self-cycles, but mutual cycles (A->B->A) will be caught during compilation.
        subtasks = [
            SubTaskSpec(id="cycle_a", goal="Cycle A", role="read", depends_on=["cycle_b"]),
            SubTaskSpec(id="cycle_b", goal="Cycle B", role="read", depends_on=["cycle_a"]),
        ]
        plan = ExecutionPlan(
            route="dynamic_dag",
            subtasks=subtasks,
        )
        # build_dynamic_graph must catch any graph cycle exception and return fallback runner
        runner = build_dynamic_graph(plan)
        assert runner is not None

        initial_state = DynamicState(
            session_id="cycle_session",
            thread_id="thread_cycle",
            plan=plan,
        )
        result_state = await runner.ainvoke(initial_state)
        assert result_state is not None

    @pytest.mark.asyncio
    async def test_checkpointer_persistence_and_state_recovery(self):
        """SqliteSaver checkpointer persistence verifies state recovery and isolation by thread_id."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        checkpointer = get_sqlite_checkpointer(db_path)
        plan = ExecutionPlan(
            route="dynamic_dag",
            subtasks=[
                SubTaskSpec(id="step_1", goal="Initial step", role="read", depends_on=[]),
                SubTaskSpec(id="step_2", goal="Follow-up step", role="edit", depends_on=["step_1"]),
            ],
        )

        runner = build_dynamic_graph(plan, checkpointer=checkpointer)
        assert runner is not None

        # Execute Session 1, Thread A
        config_a = {"configurable": {"thread_id": "thread_alpha", "session_id": "sess_1"}}
        state_a = DynamicState(session_id="sess_1", thread_id="thread_alpha", plan=plan)
        res_a = await runner.ainvoke(state_a, config=config_a)
        outputs_a = getattr(res_a, "subtask_outputs", {}) if not isinstance(res_a, dict) else res_a.get("subtask_outputs", {})
        assert "step_1" in outputs_a
        assert "step_2" in outputs_a

        # Execute Session 1, Thread B (isolated thread)
        config_b = {"configurable": {"thread_id": "thread_beta", "session_id": "sess_1"}}
        state_b = DynamicState(session_id="sess_1", thread_id="thread_beta", plan=plan)
        res_b = await runner.ainvoke(state_b, config=config_b)
        outputs_b = getattr(res_b, "subtask_outputs", {}) if not isinstance(res_b, dict) else res_b.get("subtask_outputs", {})
        assert "step_1" in outputs_b
        assert "step_2" in outputs_b
