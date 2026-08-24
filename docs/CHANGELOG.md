# AutoConduck Changelog

## [0.3.0] - 2026-08-24

### Major Features & Architectural Overhaul
- **Embedded SLM Task Architect**: Replaced static 10-factor regex heuristics with local Qwen 2.5 Coder 0.5B Instruct (Q4_K_M GGUF), generating typed Pydantic \ExecutionPlan\ specifications with sub-100ms inference.
- **Turn Guard Subsystem (\server/turn_guard.py\)**: Sub-2ms regex classifier providing instantaneous bypass for active tool loops and stagnation detection for automated SLM re-planning.
- **Dynamic DAG LangGraph Factory (\orchestrator/dynamic_factory.py\)**: Replaced static 6-phase orchestrator with runtime compilation of tailored \StateGraph\ topologies based on plan dependency DAGs and SQLite checkpointer state persistence.
- **Knowledge Vector Store & RAG Subsystem (\knowledge/vector_store.py\)**: Embedded LanceDB vector database with fast hybrid code search and automated context distillation for complex workflows.
- **Session Lifecycle & Context Guard (\orchestrator/session_guard.py\)**: Preserves immutable prefix contract (turns 0 & 1) for maximum upstream prompt caching across 40+ turns, enforcing intelligent compaction at the 80% context window ceiling.
- **Real-Time Reasoning SSE Streamer (\server/sse_streamer.py\)**: Unified SSE streaming translating internal SLM reasoning deltas into client-compatible formats (\	hinking_delta\ for Claude Code, \delta.reasoning_content\ for OpenAI clients).
- **Autonomous 3-Tier Model Pool (outing/model_pool.py\)**: Dynamic classification into \cheap_fast\ (< .50/1M), \alanced\ (.50–.00/1M), and \rontier_reasoning\ (> .00/1M) with spend guard and degraded provider protections.

### Removed
- Deleted obsolete 0.2.x heuristics: outing/complexity.py\, outing/complexity_helpers.py\, outing/evaluator.py\, outing/semantic_router.py\, outing/fast_graph.py\.
- Deleted static orchestrator files: \orchestrator/graph.py\, \orchestrator/compactor.py\, \orchestrator/complexity_helpers.py\, \orchestrator/recon.py\.
- Deleted deprecated test suites: \	est_routing_fast_path.py\, \	est_complexity_and_tuning.py\, \	est_empirical_tuning.py\.

### Quality & Testing
- Pure async pytest test suite passing at 100% (218 passed, 2 skipped, 0 failed).
- Comprehensive adversarial stress tests for SLM circuit breaker, thread-safe SQLite checkpointing, and fan-out reducer concurrency.
