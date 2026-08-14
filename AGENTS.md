# AutoConduck working contract

AutoConduck is a local zero-overhead model router and task orchestrator for OpenAI-compatible coding agents. Routing and selection are specified in `docs/design/dynamic-model-selection.md` (plus `docs/design/tuning.md`).

## Commands

- Requires Python >= 3.11. Dev setup: `pip install -r requirements.txt` then `pip install -e .`. End users install via npm (`npm install -g autoconduck`); see `npm-packaging/build.py` (builds per-platform wheels; `--check` verifies without rebuilding).
- `requirements.txt` and the pyproject runtime dependencies are in sync and must stay mirrored. There is no dev-dependencies group.
- Run: `autoconduck start --headless [--port N] [--host H]`; serves the LiteLLM-backed OpenAI-compatible endpoint on 127.0.0.1:11434 by default.
- Background service: `autoconduck start --headless --daemon`; stop with `autoconduck stop`.
- Console script: `autoconduck = autoconduck.main:main` (`autoconduck/__main__.py` exists too).
- Full CLI: `install [agents...]`, `start [--headless] [--daemon] [--supervisor (hidden)] [--port N] [--host H]`, `stop [--port]`, `ensure` / `release` (internal launcher lifecycle commands, take `--port`), `stats [--json] [--days N] [--reset] [--force]`, `tune --mode`, `update [--dry-run]`, `edit`, `reset [--force]`, `uninstall [--force]`, `--version`, plus `--claude` / `--opencode` / `--pi` shortcuts.
- No arguments opens the TUI onboarding/dashboard (`autoconduck/tui/`), falling back to headless mode if the TUI import fails.
- Smoke-test a running server with `python scripts/end_to_end_smoke.py` (bounded manual test against `/v1/messages`, `/v1/chat/completions`, `/v1/models`, `/stats`).

## Non-negotiable architecture invariants

- Fast path stays under 5 ms (`tests/test_dispatcher.py` asserts < 0.005): `routing/semantic_router.py` and `routing/evaluator.py` are sync-only (zero async); routing decisions perform no I/O or LLM calls. The default fast path executes through the compiled micro-DAG in `routing/fast_graph.py` (input_sanitize → route_match → evaluate_score → tiebreaker_resolve → model_select), gated by `enable_fast_path_graph` (default on); `dispatcher.route()`'s inline branch is the legacy fallback.
- The fast path may optionally prepend deterministic file digests (see `digest.py`) — never an LLM call, bounded by `fast_path_digest_*` config.
- Confidence in the tunable ambiguous zone (default 0.60–0.75, tightened from 0.55/0.70) can invoke a cheap LLM tiebreaker — but `tiebreaker_enabled` is **false by default**, so ambiguous turns use the deterministic complexity decision unless a tiebreaker is injected or the installation opts in. If the tiebreaker is unavailable (timeout, API error, no model, or local-port recursion), the default tiebreaker returns `None` and routing falls back to a deterministic complexity decision — SLOW when complexity >= `slow_threshold` (0.75) unless the provider is degraded, in which case FAST. Reason strings are `tiebreaker: fast|slow`, `tiebreaker: fast (below-floor)`, `tiebreaker_unavailable: complexity-fallback`, and `tiebreaker_unavailable: degraded-provider`; provider health uses `pricing.is_degraded(model, window_s, error_rate)`.
- Any LangGraph, orchestrator, or schema error degrades to the fast path, never a client-facing API error.
- Model selection is dynamic closest-cost matching, not fixed tiers: `ModelEntry.tier` is advisory/display-only. Orchestrator phases use `PHASE_BANDS` (defined in `orchestrator/graph.py`) — recon [0.10,0.45], planner [0.55,0.85], subagent [0.10,0.55], executor [0.35,0.70] — with on-the-fly targets from task value, `budget_hint`, role weights, breadth damping, and compactor-summary complexity. `RoutingDecision.model` is populated on the fast path and is never None there.
- Honor `request.is_disconnected()` for instant cancellation (used in `main.py` streaming loops).
- Agent config changes are limited to `# BEGIN AUTOCONDUCK` … `# END AUTOCONDUCK`, with backups at `~/.autoconduck/backups/<agent>/<timestamp>.bak`.

## API surface

- The FastAPI app in `autoconduck/server_streaming.py`, with routes installed by `server_routes.py`, serves the LiteLLM-backed endpoint: `/v1/chat/completions` and `/v1/models` (the latter returning exactly the three pseudo-models from `PSEUDO_MODELS` (defined in `messages_models.py`, imported by `messages_api.py`): `autoconduck`, `autoconduck-budget`, `autoconduck-expensive`), plus AutoConduck-owned `/stats` and `/healthz`, and an Anthropic-compatible `/v1/messages` shim (covered by `tests/test_messages_api.py`). `main.py` remains a compatibility facade.

## Key modules

- `routing/dispatcher.py`: thin sequence of semantic router → evaluator → optional tiebreaker, delegating to `routing/fast_graph.py` when `enable_fast_path_graph` is set (default); keep its own logic minimal.
- `routing/fast_graph.py`: compiled zero-reflection micro-DAG that runs the fast path in under ~0.1 ms (`FastGraphState`, `FastGraph`, `execute_fast_graph`).
- `routing/semantic_router.py`: wrapper over the external `semantic-router` library (the "Aurelio" pillar) with fast/slow routes and confidence.
- `routing/evaluator.py`: `T_i ∈ [0,1]`, stack-trace boost `+0.25`, escalation boost `+0.30`, hysteresis clamp `≤0.50` after escalation `≥0.80`, active de-escalation to fast below `deescalation_threshold` (0.40), tool-loop fast-path suppression, and ambiguous-zone handling; slow rule is `complexity >= slow_threshold (0.75) or (route == slow_path and confidence >= boundary_high)`.
- `routing/complexity.py` and `routing/complexity_helpers.py`: single-pass regex complexity scoring over 10 config-weighted factors (length, structural, scope_breadth, code_density, abstraction_level, uncertainty_hedge, cross_domain, task_novelty, imperative_strength, multi_step) and its extraction helpers.
- `routing/pricing.py`: `select_closest()` closest-cost model matching on ln(1+cost) scaled domain (ties→cheaper, degraded exclusion, optional `band` and `max_scaled_cost` filters, `cheapest_enabled()` deterministic fallback), `target_scaled_cost()` with pseudo-model bias (±0.20) and `value_to_cost_gamma`, EMA realized-cost blend α=0.1 (≥ `ema_min_samples` = 3 samples) via `_entry_effective_value`, failover after trailing error rate >20%, spend guard (`is_over_budget`, `spend_guard_*`), file-read cost ceiling (`is_expensive_model`), `pricing_fallback.json`; `select()`/`select_model_by_tier()` are deprecated thin wrappers.
- `providers.py`: LiteLLM `openai/<model>` plus `api_base`, `/v1/models` discovery.
- `config.py`: active model configuration, `resolve_orchestrator_model`, `normalize_api_base()` (adds `/v1` for host-root gateway URLs at LiteLLM call sites) and `_repair_base_url_scheme()` (fixes `ttps://` typos and bare hosts), `qualify_model()` (`openai/<id>`), `resolve_api_key()` (auth.yaml first, then deprecated literal `api_key` — auto-migrated by `auth.migrate_from_config()` — then `api_key_env`). `SelectionConfig` carries `enable_fast_path_graph`, `enable_executor_subagents`, `executor_enable_tools` (default on), `executor_max_tool_rounds` (10), `executor_tool_time_budget_s` (180), `executor_max_read_bytes` (200_000), `executor_enable_bash` (default off), `min_orchestrator_complexity` (0.62), `max_file_read_scaled_cost` (0.55), `deescalation_threshold` (0.40), `spend_guard_*`, `phase_bands`, and `complexity_weights`.
- `auth.py`: provider credentials kept outside config.yaml in `~/.autoconduck/auth.yaml` (`providers: {name: key-or-env:NAME}`); `get_provider_key`/`set_provider_key` and `migrate_from_config()` (moves literal keys out of config.yaml with a backup).
- `resolver.py`: pure shared model-resolution/LiteLLM-dispatch logic (`resolve_model`, `call_model`, `record_decision`); consumes `RoutingDecision.model` and routes SLOW decisions to `orchestrator.run()`.
- `digest.py`: Pattern B fast-path file digests using parallel local reads, zero LLM cost, a hard time budget, and degrade-to-fast behavior.
- `launcher.py`, `launcher_procs.py`, `launcher_shims.py`: server refcounting and process discovery; shims, environment/PATH integration, `real_binary_path`, and install/uninstall helpers.
- `cli.py` and `cli_launch.py`: CLI commands and agent installation/launch/tuning helpers; `main.py` is the compatibility entrypoint.
- `orchestrator/`: LangGraph `graph.py` (owns `PHASE_BANDS`) runs recon → recon_subagent_pool → planner → Send-based `subagents.py` → `compactor.py` → executor, with a `min_orchestrator_complexity` direct-executor short-circuit; `recon.py` (`build_recon_plan`, `ReconTarget`) discovers up to 5 target files pre-planner (zero-LLM-cost when explicit paths are present); `planner.py` owns `TaskPlan` plus `_extract_file_paths`/`_read_files`/`_format_file_contents`; `roles.py` owns `RoleConfig`/`role_card` (executor role declares `edit`/`write`/`bash`); `subagents.py` (`run_subagent`) is a pure LLM text call — no tool execution anywhere in it; `compactor.py` (`compact`) merges analyst outputs; `progress.py` emits `ProgressEvent`. The executor node runs a bounded OpenAI function-calling loop (`executor_loop.py` → `tools.py`: read/grep/glob/list/edit/write + opt-in bash) gated by `executor_enable_tools` (default on); edits/writes are fail-closed against `plan.subtasks[*].scope`, and any tool-loop error degrades to text synthesis via `_call`; executor-subagent fan-out (`_run_executor_subagents` + `FileClaimRegistry`) is separately gated by `enable_executor_subagents` (default off). Shared helpers in `helpers.py`.
- `model_presets.py`, `presets_data.py`, `presets_ingest.py`, and `presets_fallback.py`: runtime catalog data and ingestion from the installed litellm registry (`litellm.model_cost` via `_ingest_litellm_costs`), while public names remain re-exported from `model_presets.py`.
- `agents/`: agent adapters (`opencode.py`, `claude_code.py`, `pi.py`).
- `messages_api.py`, `messages_models.py`, and `messages_sse.py` provide the Anthropic shim models and SSE translation; Textual `tui/` and its `tui/onboarding/` package provide onboarding/monitoring (Ctrl+C is the single quit chord per `tui/keymap.py`; Textual Ctrl+Q is disabled).

## Repository layout

- `routing/` contains the synchronous fast-path stack; `server_streaming.py`/`server_routes.py` split the FastAPI server; `cli.py`/`cli_launch.py` split CLI concerns; `presets_*` split catalog data and ingestion; and `tui/onboarding/` contains the onboarding package.

## State, environment, and scope

- State lives under `~/.autoconduck/` or `$AUTOCONDUCK_HOME`.
- Environment overrides (read in `config.py`): `AUTOCONDUCK_HOME`, `AUTOCONDUCK_PORT`, `AUTOCONDUCK_LOG_LEVEL`.
- Config also exposes `fast_path_digest_*` bounds and enablement fields for deterministic fast-path digests.
- Gitignored: `.autoconduck/`, `backups/`, `graphify-out/`, `build/`, and `*.egg-info/`.
- LiteLLM owns caching and native cost logging; `/stats` is the audit surface.
- Do not add legacy routing, state, caching, or telemetry layers. The LiteLLM-backed endpoint is the API surface; `resolver.py` is the integration point — `dispatcher.route()` (→ `fast_graph.py`) for pseudo-models, `orchestrator.run()` for SLOW decisions.

## Keeping the model catalog fresh

- The runtime catalog is derived from the installed `litellm` model registry, so upgrading LiteLLM keeps the dropdown and TUI catalog current.
- A weekly GitHub Actions workflow (`.github/workflows/refresh-catalog.yml`, cron `0 6 * * 1`) refreshes the static `docs/model_catalog.md` snapshot and opens a pull request when it changes.
- Check the snapshot locally with `python scripts/refresh_catalog.py --check`; run `python scripts/refresh_catalog.py` before npm publish.

## Gotchas

- Run `pytest` from the repository root; if the package is not installed, use `PYTHONPATH=.`. Tests are pure async unit tests with no network or services (`asyncio_mode = "auto"`, `testpaths = ["tests"]`); run a single file with `pytest tests/test_<name>.py`.
- No lint, format, or typecheck config exists — pytest is the only verification gate (the only CI workflow is the catalog refresh).
- After editing code, run `graphify update .` to keep the code graph current.
- The TUI uses Ctrl+C as its single quit chord; Textual Ctrl+Q is disabled.
