# AutoConduck Changelog

## [0.3.5] - 2026-08-25

### Model Selection Overhaul
- **Turn Guard false-positive fix (`server/turn_guard.py`)**: Removed the "complexity drift" escalation (>10 distinct files / >30 tool turns) that misclassified healthy multi-file tool loops as stagnant, causing runaway replanning and Dynamic-DAG recompilation (the grok-4.6 cost incident). Healthy tool loops now stay on DIRECT_ACTIVE_TIER; only 3+ identical calls or 2+ consecutive errors escalate.
- **Capability-vector scoring (`routing/model_pool.py`)**: Replaced the 1D `capability_score` gate with a 4-dim `capability_vector` (reasoning, tool_reliability, code_quality, latency_class) scored by `capability_fit()` (min-with-bonus), weighted per SLM `task_type` via `TASK_TYPE_WEIGHTS`. Selection is now "fit-gate then cheapest"; legacy scalar entries fall back compatibly.
- **Confidence-tightened floor (`routing/dispatcher.py:_select_planned`)**: Low SLM `plan.confidence` raises the capability floor (`min(base + 0.15*(1-conf), 0.6)`).
- **Opt-in per-selection price cap (`CapabilitySLA.max_price_usd_per_mtok`, config `selection.path_price_cap_usd_per_mtok`)**: Reframed the earlier "USD/min" ceiling into an honest per-1M-token price cap, DISABLED by default ({}). Not a spend meter; falls back to cheapest available with `fallback_reason="price_cap_emptied_pool"`.
- **Explainability (`SelectionInfo`/`RoutingDecision` → `/stats`)**: candidates_considered, candidates_excluded_by, binding_constraint, capability_fit_applied, binding_capability_dim, spend_cap_engaged, fallback_reason.

### Notes
- Version bumps now via `python scripts/bump_version.py --patch`.

## [0.3.4] - 2026-08-25

### Fixes & Reliability Enhancements
- **Configuration Resilience & In-Memory Preservation (`config/manager.py`)**: Fixed an issue where transient file locks, missing/empty reads, or background atomic file replaces could cause `get_config()` to evict loaded models and fall back to hardcoded defaults. The in-memory configuration is now preserved with fail-soft guarantees when the config file is temporarily unavailable.
- **Thread-Safe Atomic Config I/O**: Added thread locking (`threading.RLock`) around config load and save operations, collision-proof temporary filenames, and Windows-specific retry backoff for atomic replacements.
- **Automatic Backup Recovery**: `load_config()` automatically recovers active configurations from timestamped backups under `~/.autoconduck/backups/config/` on cold starts if `config.yaml` is missing or corrupted.
- **Provider-Aware Fallback Resolution (`config/resolver.py`, `routing/model_pool.py`, `routing/pricing.py`)**: Enhanced `resolve_orchestrator_model()` to discover provider credentials from environment variables (`LLMGATEWAY_API_KEY`, `ANTHROPIC_API_KEY`, etc.) and fallback presets before defaulting to `gpt-4o`.

## [0.3.2] - 2026-08-24

### Enhancements & Dynamic Model Tiering
- **Dynamic Pool-Relative Model Tiering (`routing/model_pool.py`)**: Replaced fixed static price cutoffs with pool-relative dynamic quantile partitioning. Automatically tiers models across `cheap_fast`, `balanced`, and `frontier_reasoning` based on the user's active/selected models (whether 1, 2, 3, 6, or 20+ models).
- **VCS & Git Fast-Path Intent Routing**: Explicit fast-path classification in `slm_planner.py` for git commits, diffs, status, and routine tasks, preventing runaway costs on expensive models.
- **Emoji Removal Across Logs & TUI**: Purged all emojis from logs, TUI dashboard, onboarding screens, SSE streamers, progress formatters, and CLI outputs in favor of clean bracketed ASCII markers (`[OK]`, `[WARN]`, `[ERR]`, `[..]`, `[>>]`).

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
