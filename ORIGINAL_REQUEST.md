# Original User Request

## 2026-08-24T05:01:41Z

Refactor and modernize AutoConduck into version 0.3.0, transforming it from heuristic 10-factor scoring and static DAGs into an intelligent Dynamic SLM Orchestration Engine powered by Qwen 2.5 Coder 0.5B Instruct (Q4_K_M GGUF), dynamic LangGraph factory compilation, SSE reasoning streams, LanceDB RAG, and prompt-cache-friendly session lifecycle guards.

Working directory: `C:\Users\plum\Documents\Works\AutoConduck`
Integrity mode: `development`

## Requirements

### R1. Phase 0 — Pre-Flight & Dependency Synchronization
- Bump package version to `0.3.0-dev` across `autoconduck/__init__.py` and `pyproject.toml`.
- Synchronize `requirements.txt` and `pyproject.toml` runtime dependencies:
  - Add: `llama-cpp-python>=0.3.0`, `outlines>=0.1.0`, `lancedb>=0.10`, `langgraph-checkpoint-sqlite`.
  - Remove: `semantic-router`.
- Establish test suite baseline snapshot and create temporary compatibility shims in `autoconduck/_compat/`.

### R2. Phase 1 & 2 — Green-Field Foundations & Integration Wiring
- **Turn Guard (`autoconduck/server/turn_guard.py`)**: Synchronous 0ms classifier routing in-flight tool loops directly to the active model tier, with stagnation escalation (3+ identical calls or 2+ consecutive errors) back to the SLM.
- **SLM Architect (`autoconduck/routing/slm_planner.py`)**: Embedded Qwen 2.5 Coder 0.5B Instruct (`Q4_K_M` GGUF) with Outlines BNF logit-constrained grammar generating strictly validated `ExecutionPlan` JSON. Includes a 100ms circuit breaker degrading gracefully to the balanced tier.
- **Dynamic LangGraph Factory (`autoconduck/orchestrator/dynamic_factory.py`)**: Compiles transient `StateGraph` DAGs on the fly with parallel subtask fan-out, conditional RAG node injection, terminal Synthesizer node on `frontier_reasoning`, and `SqliteSaver` checkpointer keyed by `session_id`/`thread_id`.
- **Dynamic SSE Thinking Streamer (`autoconduck/server/sse_streamer.py`)**: Emits real-time visual DAG execution state transitions (`⏳`, `🟢`, `🔴`) as `delta.reasoning_content` (OpenAI) and `thinking_delta` (Anthropic), transitioning smoothly into markdown response content.
- **Selective Knowledge / RAG (`autoconduck/knowledge/`)**: LanceDB embedded vector index for project dependencies and API contracts, extracting concise context (max 250 tokens) injected strictly into `State["verified_context"]`.
- **Session Lifecycle & Context Guard (`autoconduck/orchestrator/session_guard.py`)**: Immutable prefix contract preserving upstream prompt-caching, with 80% context window compaction summarizing verbose tool outputs without corrupting code blocks.
- **Autonomous Model Discovery & Tiering (`autoconduck/routing/model_pool.py`)**: Dynamic auto-tiering (`cheap_fast` <$0.50/1M, `balanced` $0.50–$4.00/1M, `frontier_reasoning` >$4.00/1M) with context window and function calling filters, wrapping `pricing.select_closest()`.
- Wire Turn Guard, SLM Planner, Dynamic Factory, SSE Streamer, and Session Guard into `autoconduck/server/server_routes.py` and `autoconduck/routing/dispatcher.py`.

### R3. Phase 3 & 4 — Migration of Retained Systems & Test Suite Overhaul
- Adapt retained modules: `pricing.py` (`select_for_tier`), `stats.py` (`ExecutionPlan` logging), `config.py` (`SelectionConfig` schema update), `resolver.py`, `tui/app.py` (DAG trace visualization), `tuning.py`, `digest.py`, and `__init__.py`.
- Delete obsolete tests (`test_routing_fast_path.py`, `test_complexity_and_tuning.py`, `test_empirical_tuning.py`).
- Implement comprehensive unit and integration test suites:
  - `tests/test_turn_guard.py`
  - `tests/test_slm_planner.py`
  - `tests/test_dynamic_factory.py`
  - `tests/test_sse_streamer.py`
  - `tests/test_rag_node.py`
  - `tests/test_session_guard.py`
  - `tests/test_model_pool.py`
  - Rewrite `test_pricing_and_catalog.py`, `test_server_and_apis.py`, `test_orchestrator.py`, and `integration/test_simulations.py`.

### R4. Phase 5, 6 & 7 — Documentation, Stale Codebase Cleanup & Validation
- Update `README.md`, `AGENTS.md`, `docs/design/dynamic-model-selection.md`, `docs/design/tuning.md`, and `docs/CHANGELOG.md`.
- Create design docs: `docs/design/slm-architecture.md`, `docs/design/dynamic-dag.md`, `docs/design/session-management.md`, `docs/design/rag-subsystem.md`, and `docs/migration/0.2-to-0.3.md`.
- Delete stale files: `routing/complexity.py`, `routing/complexity_helpers.py`, `routing/evaluator.py`, `routing/semantic_router.py`, `routing/fast_graph.py`, `orchestrator/graph.py`, `orchestrator/compactor.py`, `orchestrator/complexity_helpers.py`, `orchestrator/recon.py`, `progress.py`, and `autoconduck/_compat/`.
- Rebuild knowledge graph with `graphify update .`.
- Validate all latency, streaming, context integrity, and safety acceptance criteria; ensure 100% test pass rate. Bump version to final `0.3.0`.

## Acceptance Criteria

### Performance & Latency
- [ ] In-flight tool-loop bypass overhead is $< 2\text{ ms}$.
- [ ] New user turns complete SLM planning and dynamic graph compilation in $< 75\text{ ms}$.
- [ ] SLM circuit breaker engages in $\le 100\text{ ms}$ on timeout/validation failure, falling back gracefully to the balanced tier.

### Streaming & Protocol Compatibility
- [ ] OpenAI `/v1/chat/completions` clients receive live `delta.reasoning_content` chunks showing DAG status transitions without connection stalls.
- [ ] Anthropic `/v1/messages` clients receive well-formed `thinking_delta` content blocks.
- [ ] Synthesizer completion smoothly transitions stream into primary response content without duplicate chunks or protocol errors.

### Context & Session Integrity
- [ ] Multi-turn conversations (40+ turns) maintain immutable prefix integrity to preserve prompt-cache hits on supported providers.
- [ ] Compaction at 80% context window condenses verbose tool output while preserving code fences, structural headers, and the system prompt.
- [ ] LangGraph execution state persists across server restarts via `SqliteSaver`.

### Codebase Cleanliness & Test Suite
- [ ] Zero stale imports or dangling references to deleted modules (`complexity.py`, `evaluator.py`, `semantic_router.py`, `compactor.py`, etc.).
- [ ] All unit and integration tests in `tests/` pass with zero failures.
- [ ] `scripts/end_to_end_smoke.py` passes completely against `/v1/chat/completions`, `/v1/messages`, `/v1/models`, and `/stats`.
