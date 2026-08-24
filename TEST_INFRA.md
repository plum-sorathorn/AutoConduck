# AutoConduck 0.3.0 Test Infrastructure & Quality Assurance Specification

## 1. Overview & Architecture Philosophy

AutoConduck 0.3.0 modernizes the routing and orchestration engine into an intelligent Dynamic SLM Orchestration Engine. Testing this architecture requires an **opaque-box, requirement-driven, multi-tiered** testing methodology.

The test infrastructure is designed around four foundational principles:
1. **Opaque-Box Verification**: Tests validate observable behaviors, strict interface contracts, response schemas, and latency SLAs defined in `PROJECT.md` and `ORIGINAL_REQUEST.md`, rather than internal implementation details.
2. **Deterministic Simulation & Oracles**: Integration scenarios use thread-safe mock providers and state oracles to verify dynamic DAG compilation, SSE streaming transitions, prompt cache prefix preservation, and RAG context injection without external network flakiness.
3. **Progressive Testability & Graceful Shimming**: Tests dynamically adapt to available binary components (`llama-cpp-python`, `outlines`, `lancedb`, `langgraph-checkpoint-sqlite`) via `_compat/` fallbacks, ensuring 100% test execution across all environments.
4. **Adversarial & Stagnation Defense**: Test fixtures systematically inject corrupted tokens, syntax breakage, cycle deadlocks, tool error cascades, and context boundary overflows.

---

## 2. The 4-Tier Testing Methodology

The AutoConduck test suite is organized into 4 distinct verification tiers:

```
+-----------------------------------------------------------------------+
|  Tier 4: Real-World Application Scenarios (Full E2E Agent Flows)      |
|  - Claude Code / OpenCode / Pi multi-turn workflows, tool recovery    |
+-----------------------------------------------------------------------+
|  Tier 3: Cross-Feature Combinations (Pairwise & DAG Interactions)     |
|  - Turn Guard + SLM Escalation + SSE Streaming + Session Compaction   |
+-----------------------------------------------------------------------+
|  Tier 2: Boundary & Corner Cases (>=5 per feature)                    |
|  - Context overflows, 100ms circuit breaker timeout, corrupted syntax |
+-----------------------------------------------------------------------+
|  Tier 1: Feature Coverage (>=5 per feature)                           |
|  - Happy path contract execution for each of the 8 modernizations     |
+-----------------------------------------------------------------------+
```

---

## 3. Tier Breakdown by Feature (8 Core Subsystems)

### Feature 1: Turn Guard (`turn_guard.py`)
- **Tier 1 (Feature Coverage >=5)**:
  1. `test_turn_guard_new_user_message_routes_to_slm`: Clean user turn triggers `TurnAction.SLM_PLAN`.
  2. `test_turn_guard_active_tool_loop_bypasses_to_active_tier`: In-flight tool responses route to `TurnAction.DIRECT_ACTIVE_TIER`.
  3. `test_turn_guard_stagnation_three_identical_calls_escalates`: 3 identical consecutive tool calls trigger `TurnAction.ESCALATE_SLM`.
  4. `test_turn_guard_stagnation_two_consecutive_errors_escalates`: 2 consecutive tool errors trigger `TurnAction.ESCALATE_SLM`.
  5. `test_turn_guard_streak_counters_reset_on_progress`: Distinct successful tool calls reset error streak and stagnation counters.
- **Tier 2 (Boundary & Corner Cases >=5)**:
  1. `test_turn_guard_empty_messages_list`: Handles empty input list safely without exceptions.
  2. `test_turn_guard_malformed_tool_call_payload`: Non-dict or missing arguments in tool calls.
  3. `test_turn_guard_sub_2ms_execution_sla`: 100 consecutive classifications complete within sub-2ms average latency.
  4. `test_turn_guard_mixed_success_and_error_sequence`: Alternating success and error does not prematurely trip 2-error threshold.
  5. `test_turn_guard_nested_anthropic_and_openai_tool_formats`: Handles both Anthropic `tool_use`/`tool_result` and OpenAI `tool_calls`/`tool` structures.
- **Tier 3 (Cross-Feature Combinations)**:
  - Turn Guard bypass integrated with FastAPI streaming handler and Model Pool tier selection.
- **Tier 4 (Real-World Application Scenarios)**:
  - Aider/Claude Code multi-file edit loop: 5 continuous tool steps bypass SLM, 6th step encounters 2 errors and triggers dynamic re-planning.

---

### Feature 2: SLM Architect & Circuit Breaker (`slm_planner.py`)
- **Tier 1 (Feature Coverage >=5)**:
  1. `test_slm_planner_generates_valid_execution_plan`: Returns well-formed `ExecutionPlan` with valid JSON types.
  2. `test_slm_planner_fast_direct_route_for_simple_queries`: Simple conversational turns classify as `fast_direct`.
  3. `test_slm_planner_dynamic_dag_route_for_complex_refactoring`: Complex multi-file tasks yield `dynamic_dag` with subtasks.
  4. `test_slm_planner_rag_requirement_flag`: Queries referencing external dependencies set `needs_rag=True` and populate `rag_queries`.
  5. `test_slm_planner_model_tier_recommendations`: Suggested tiers map accurately (`cheap_fast`, `balanced`, `frontier_reasoning`).
- **Tier 2 (Boundary & Corner Cases >=5)**:
  1. `test_slm_planner_circuit_breaker_timeout_under_100ms`: Injected slow inference trips 100ms circuit breaker and returns balanced fallback.
  2. `test_slm_planner_unparseable_llm_output_fallback`: Corrupted non-JSON text gracefully falls back to `fallback_used=True`.
  3. `test_slm_planner_empty_prompt_and_system_only`: Handles minimal input without crashing.
  4. `test_slm_planner_extreme_prompt_token_length`: 32k token prompt truncated safely before SLM inference.
  5. `test_slm_planner_subtask_cyclic_dependency_sanitization`: Subtasks with cyclic `depends_on` are topological-sorted and cleaned.
- **Tier 3 (Cross-Feature Combinations)**:
  - SLM Planner output feeds directly into `DynamicLangGraphFactory` for executable DAG compilation.
- **Tier 4 (Real-World Application Scenarios)**:
  - Full repo audit prompt: SLM planner emits 3-stage plan (recon -> read -> synthesize) with sub-75ms compilation.

---

### Feature 3: Dynamic LangGraph Factory (`dynamic_factory.py`)
- **Tier 1 (Feature Coverage >=5)**:
  1. `test_dynamic_factory_compiles_linear_dag`: Compiles and executes single-path linear subtasks.
  2. `test_dynamic_factory_parallel_subtask_fanout`: Independent subtasks run concurrently via fan-out nodes.
  3. `test_dynamic_factory_conditional_rag_node_injection`: RAG node is included if and only if `plan.needs_rag` is True.
  4. `test_dynamic_factory_synthesizer_terminal_node`: Final node aggregates subtask outputs using `synthesizer_tier`.
  5. `test_dynamic_factory_sqlite_checkpointer_persistence`: Thread state is saved to `SqliteSaver` and retrievable by `thread_id`.
- **Tier 2 (Boundary & Corner Cases >=5)**:
  1. `test_dynamic_factory_empty_subtasks_plan`: Plan with zero subtasks routes directly to synthesizer or fast exit.
  2. `test_dynamic_factory_subtask_failure_isolation`: Failure in one parallel branch logs to `subtask_errors` without aborting other branches.
  3. `test_dynamic_factory_state_persistence_across_restarts`: Recreating factory with existing SQLite database recovers previous session checkpoints.
  4. `test_dynamic_factory_deep_fanout_stress`: Graph with 16 parallel subtasks compiles within 50ms and executes deterministically.
  5. `test_dynamic_factory_is_fallback_state_propagation`: Fallback execution state is preserved across all node handoffs.
- **Tier 3 (Cross-Feature Combinations)**:
  - Factory compilation executed concurrently with SSE Streamer node transitions.
- **Tier 4 (Real-World Application Scenarios)**:
  - Multi-file bugfix across `auth.py` and `config.py`: Recon -> Parallel Subagents -> Synthesizer -> State Checkpoint.

---

### Feature 4: Dynamic SSE Thinking Streamer (`sse_streamer.py`)
- **Tier 1 (Feature Coverage >=5)**:
  1. `test_sse_streamer_openai_reasoning_content_chunks`: Emits `delta.reasoning_content` with glyphs (`⏳`, `🟢`, `🔴`).
  2. `test_sse_streamer_anthropic_thinking_delta_blocks`: Emits `thinking_delta` content blocks for Anthropic clients.
  3. `test_sse_streamer_node_state_transitions`: Emits distinct events for `pending`, `running`, `completed`, and `failed`.
  4. `test_sse_streamer_smooth_markdown_transition`: Closes reasoning block and streams markdown tokens seamlessly.
  5. `test_sse_streamer_terminal_stop_event`: Sends appropriate stream termination frames (`data: [DONE]` or `message_stop`).
- **Tier 2 (Boundary & Corner Cases >=5)**:
  1. `test_sse_streamer_rapid_burst_transitions`: 50 state transitions in under 10ms stream without dropped frames.
  2. `test_sse_streamer_empty_token_stream`: Handles empty synthesizer token generator gracefully.
  3. `test_sse_streamer_client_disconnect_cancellation`: Aborts streaming generator immediately when client disconnects.
  4. `test_sse_streamer_unicode_and_ansi_escaping`: Unicode symbols and multiline detail strings escape cleanly in JSON SSE frames.
  5. `test_sse_streamer_invalid_protocol_fallback`: Unknown client protocol defaults safely to OpenAI-compatible framing.
- **Tier 3 (Cross-Feature Combinations)**:
  - SSE Streamer hooked into LangGraph node execution events during live FastAPI streaming.
- **Tier 4 (Real-World Application Scenarios)**:
  - Interactive Claude Code terminal session receiving real-time thinking status updates during complex multi-stage orchestration.

---

### Feature 5: Selective Knowledge / RAG Subsystem (`knowledge/`)
- **Tier 1 (Feature Coverage >=5)**:
  1. `test_rag_vector_store_indexing`: Ingests code files and symbols into LanceDB vector table.
  2. `test_rag_vector_store_similarity_search`: Retrieves top-k relevant snippets given natural language or identifier query.
  3. `test_rag_context_cap_enforcement_250_tokens`: Context injected into `State["verified_context"]` never exceeds 250 tokens.
  4. `test_rag_ast_extractor_symbols`: Extracts class definitions, function signatures, and docstrings accurately.
  5. `test_rag_in_memory_fallback_compatibility`: Operates transparently with `LanceDBFallbackConnection` when native LanceDB is absent.
- **Tier 2 (Boundary & Corner Cases >=5)**:
  1. `test_rag_empty_query_or_empty_index`: Empty queries return empty results without exceptions.
  2. `test_rag_binary_and_ignored_files`: Binary files and `.gitignore` matches are excluded from indexing.
  3. `test_rag_query_with_special_characters`: Queries with syntax tokens (`::`, `->`, `[T]`, quotes) execute cleanly.
  4. `test_rag_duplicate_table_creation`: Re-creating table with `exist_ok=True` or `mode="overwrite"` succeeds without locks.
  5. `test_rag_extreme_file_size_handling`: Files over 1MB are chunked and bounded safely.
- **Tier 3 (Cross-Feature Combinations)**:
  - RAG extraction invoked inside LangGraph conditional node, populating `State["verified_context"]` for subagent prompt construction.
- **Tier 4 (Real-World Application Scenarios)**:
  - User asks "How does provider authentication work?": RAG extracts auth schema (<250 tokens), Planner injects context into subagent execution.

---

### Feature 6: Session Lifecycle & Context Guard (`session_guard.py`)
- **Tier 1 (Feature Coverage >=5)**:
  1. `test_session_guard_immutable_prefix_preservation`: Verifies system prompt and initial turns remain byte-identical across turns.
  2. `test_session_guard_compaction_at_80_percent_ceiling`: Triggers compaction when context exceeds 80% of window size.
  3. `test_session_guard_code_fence_integrity`: Preserves code blocks (```python ... ```) without syntax corruption.
  4. `test_session_guard_structural_header_preservation`: Preserves markdown headers (`#`, `##`, `###`) during tool output summarization.
  5. `test_session_guard_returns_metrics_contract`: Returns `SessionGuardResult` with valid token counts and flags.
- **Tier 2 (Boundary & Corner Cases >=5)**:
  1. `test_session_guard_40_turn_continuous_session`: 40-turn simulated conversation preserves prompt cache prefix and manages memory.
  2. `test_session_guard_unclosed_code_blocks`: Handles unclosed triple backticks safely without runaway truncation.
  3. `test_session_guard_massive_single_tool_output`: 100k character command output summarized cleanly to under 1k tokens.
  4. `test_session_guard_zero_or_negative_context_window`: Defaults safely to reasonable fallback window (e.g., 8192).
  5. `test_session_guard_already_compact_context_noop`: Context under 80% ceiling is passed through untouched with `compacted=False`.
- **Tier 3 (Cross-Feature Combinations)**:
  - Session Guard applied at the ingress of `/v1/chat/completions` and `/v1/messages` before Turn Guard evaluation.
- **Tier 4 (Real-World Application Scenarios)**:
  - Continuous 50-turn agent refactoring session where tool outputs are periodically compacted while cache prefix remains intact.

---

### Feature 7: Autonomous Model Pool (`model_pool.py`)
- **Tier 1 (Feature Coverage >=5)**:
  1. `test_model_pool_tier_classification_pricing`: Auto-tiers models into `cheap_fast` (<$0.50), `balanced` ($0.50-$4.00), `frontier_reasoning` (>$4.00).
  2. `test_model_pool_context_window_filtering`: Filters out models with insufficient context window.
  3. `test_model_pool_tool_calling_support_filtering`: Requires tool calling support when `requires_tools=True`.
  4. `test_model_pool_select_for_tier_mapping`: Returns optimal model ID for requested tier.
  5. `test_model_pool_pseudo_model_resolution`: Resolves `autoconduck`, `autoconduck-budget`, `autoconduck-expensive` properly.
- **Tier 2 (Boundary & Corner Cases >=5)**:
  1. `test_model_pool_empty_catalog_fallback`: Gracefully falls back to hardcoded default models if catalog is empty.
  2. `test_model_pool_degraded_provider_exclusion`: Excludes models whose providers are marked as degraded/down.
  3. `test_model_pool_extreme_context_requirement`: Requesting 1,000,000 context tokens selects highest-context model or closest match.
  4. `test_model_pool_invalid_tier_string`: Unknown tier string defaults safely to `balanced`.
  5. `test_model_pool_zero_cost_free_models`: Free/local models ($0.00/1M) correctly classify as `cheap_fast`.
- **Tier 3 (Cross-Feature Combinations)**:
  - Model Pool tier selection consumed by Dispatcher, SLM Planner, and Dynamic Factory Synthesizer node.
- **Tier 4 (Real-World Application Scenarios)**:
  - Multi-provider environment with Anthropic, OpenAI, and Groq: dynamically routes subagents to cheap_fast and synthesizer to frontier_reasoning.

---

### Feature 8: Smoke Test & API Surface (`end_to_end_smoke.py` & Endpoints)
- **Tier 1 (Feature Coverage >=5)**:
  1. `test_smoke_v1_models_endpoint`: Validates listing of pseudo-models and configured catalog entries.
  2. `test_smoke_v1_chat_completions_openai_flow`: Tests standard and streaming chat completions.
  3. `test_smoke_v1_messages_anthropic_flow`: Tests Anthropic messages request translation and streaming response.
  4. `test_smoke_stats_audit_logging`: Validates `/stats` metrics, decision records, and cost savings calculations.
  5. `test_smoke_healthz_liveness`: Confirms `/healthz` returns 200 OK with server status.
- **Tier 2 (Boundary & Corner Cases >=5)**:
  1. `test_smoke_invalid_json_payload`: Returns 400/422 with informative error instead of 500.
  2. `test_smoke_missing_auth_header_or_empty_key`: Handles unauthenticated requests safely according to config.
  3. `test_smoke_concurrent_mixed_requests`: 20 concurrent requests across OpenAI and Anthropic endpoints execute without race conditions.
  4. `test_smoke_disconnect_during_stream`: Immediate client abort during streaming triggers cleanup without orphan tasks.
  5. `test_smoke_script_standalone_execution`: Running `python scripts/end_to_end_smoke.py` exits with code 0.
- **Tier 3 (Cross-Feature Combinations)**:
  - Full request lifecycle testing: Request -> Session Guard -> Turn Guard -> SLM Planner / Active Model -> SSE Streamer -> Stats Log.
- **Tier 4 (Real-World Application Scenarios)**:
  - Real-time agent integration test exercising Claude Code CLI and OpenCode shim talking to the local proxy server.

---

## 4. Test Suite Layout & File Index

The modernization test suite resides in `tests/`:

```
tests/
├── conftest.py                           # Shared fixtures, mock servers, and configuration overrides
├── test_turn_guard.py                    # Feature 1: Turn Guard 0ms classifier & stagnation
├── test_slm_planner.py                   # Feature 2: SLM Planner & 100ms circuit breaker
├── test_dynamic_factory.py               # Feature 3: Dynamic LangGraph Factory & SqliteSaver
├── test_sse_streamer.py                  # Feature 4: Dynamic SSE Thinking Streamer
├── test_rag_node.py                      # Feature 5: LanceDB Selective RAG Subsystem
├── test_session_guard.py                 # Feature 6: Session Guard & 80% Context Compaction
├── test_model_pool.py                    # Feature 7: Autonomous 3-tier Model Pool
├── test_pricing_and_catalog.py           # Retained & updated pricing/tier tests
├── test_server_and_apis.py               # Server routes, Anthropic shim, /stats, /healthz
├── test_orchestrator.py                  # Orchestrator tool execution and role cards
├── test_auth_and_providers.py            # Auth resolution and credential migration
├── test_cli_and_lifecycle.py             # CLI commands, daemon management, shims
├── test_agent_adapters.py                # 8 Agent adapters (Claude, OpenCode, Pi, etc.)
├── test_tui_and_onboarding.py            # Textual TUI onboarding and dashboard
├── test_e2e_modernization.py             # Cross-cutting E2E modernization simulations
└── integration/
    ├── test_launcher_process.py          # Process refcounting and supervisor lifecycle
    └── test_simulations.py               # Full multi-turn agent simulation scenarios
```

---

## 5. Execution Commands & Verification Matrix

### Run Complete Test Suite
```bash
python -m pytest -v
```

### Run Modernization Test Suites Only
```bash
python -m pytest tests/test_turn_guard.py tests/test_slm_planner.py tests/test_dynamic_factory.py tests/test_sse_streamer.py tests/test_rag_node.py tests/test_session_guard.py tests/test_model_pool.py tests/test_e2e_modernization.py -v
```

### Run Performance & Latency SLA Checks
```bash
python -m pytest tests/test_turn_guard.py -k "sub_2ms or sla" -v
python -m pytest tests/test_slm_planner.py -k "circuit_breaker or latency" -v
```

### Run End-to-End Smoke Script
```bash
python scripts/end_to_end_smoke.py
```

---

## 6. Coverage & Quality Gates

| Milestone | Gate Criteria | Verification Command |
|-----------|---------------|----------------------|
| **M1: Pre-Flight** | Version 0.3.0-dev, dependencies synced, `_compat/` shims pass | `python -m pytest` |
| **M2: Foundations** | 7 greenfield test suites pass with 100% assertions | `python -m pytest tests/test_*.py` |
| **M3: Overhaul** | Rewritten server, pricing, orchestrator, and simulation suites pass | `python -m pytest tests/` |
| **M4: Validation** | Stale tests deleted, `scripts/end_to_end_smoke.py` passes, 0 failures | `python scripts/end_to_end_smoke.py` |

AutoConduck 0.3.0 quality assurance is strictly bounded by this specification.
