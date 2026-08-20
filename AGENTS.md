# AutoConduck working contract

AutoConduck is a local zero-overhead model router and task orchestrator for OpenAI-compatible coding agents. Routing and selection specifications are documented in `docs/design/dynamic-model-selection.md` (plus `docs/design/tuning.md`).

## Commands & Workflows

- **Python requirements**: Python >= 3.11. Dev setup: `pip install -r requirements.txt` then `pip install -e .`. End users install via npm (`npm install -g autoconduck`); see `npm-packaging/build.py` (builds per-platform wheels; `--check` verifies without rebuilding).
- **Dependency sync**: `requirements.txt` and `pyproject.toml` runtime dependencies are strictly mirrored.
- **Server execution**:
  - Headless proxy service: `autoconduck start --headless [--port N] [--host H]` (defaults to `127.0.0.1:11434`).
  - Background daemon: `autoconduck start --headless --daemon`; stop with `autoconduck stop [--port N]`.
  - TUI interactive dashboard: `autoconduck` (no arguments opens TUI onboarding/dashboard; falls back to headless if TUI import fails).
  - Console entrypoints: `autoconduck = autoconduck.main:main` and `conduck = autoconduck.main:main`.
- **Full CLI commands**:
  - `install [agents...]` — wrap agents with refcounted shims.
  - `start [--headless] [--daemon] [--supervisor (hidden)] [--port N] [--host H]` — launch the server.
  - `stop [--port N]` — stop the running daemon.
  - `ensure` / `release` — internal launcher lifecycle commands for process refcounting.
  - `stats [--json] [--days N] [--reset] [--force]` — inspect routing audit logs and cost savings.
  - `tune --mode` — optimize routing parameters.
  - `update [--dry-run]` — update runtime model catalog.
  - `edit` — open interactive model list editor.
  - `reset [--force]` / `uninstall [--force]` — restore agent configs and remove shims.
  - `--version`, `--claude`, `--opencode`, `--pi` shortcuts.
- **Smoke test**: `python scripts/end_to_end_smoke.py` (bounded test against `/v1/messages`, `/v1/chat/completions`, `/v1/models`, `/stats`).

## Non-Negotiable Architecture Invariants

- **Sub-5ms Fast Path**: `routing/semantic_router.py` and `routing/evaluator.py` are strictly synchronous (zero async, zero I/O, zero LLM calls). Default fast path executes through the streamlined zero-reflection pipeline in `routing/dispatcher.py` (input_sanitize → route_match → evaluate_score → tiebreaker_resolve → model_select) with a lightweight facade in `routing/fast_graph.py`.
- **Fast-Path File Digests**: Deterministic parallel local reads prepended bounded by `fast_path_digest_*` config — never an LLM call.
- **Ambiguous Zone & Tiebreaker**: Tunable zone (default `0.60–0.75`). LLM tiebreaker is **disabled by default** (`tiebreaker_enabled: false`), so ambiguous turns use deterministic complexity decisions. If tiebreaker fails or is unavailable, routing falls back deterministically to SLOW when complexity >= `slow_threshold` (0.75) unless provider is degraded.
- **Fail-Soft Degradation**: Any LangGraph, orchestrator, tool, or schema error degrades gracefully to the fast path—never surfacing a client-facing 500 error.
- **Dynamic Closest-Cost Model Matching**: `ModelEntry.tier` is advisory/display-only. Model selection uses logarithmic cost scaling `ln(1 + cost)` with realized EMA cost blending. Orchestrator phases use `PHASE_BANDS` in `orchestrator/graph.py` (recon `[0.10, 0.45]`, planner `[0.55, 0.85]`, subagent `[0.10, 0.55]`, executor `[0.35, 0.70]`). `RoutingDecision.model` is populated on the fast path and is never None there.
- **Stream Cancellation**: Honor `request.is_disconnected()` in streaming loops (`server/server_streaming.py`) for instantaneous client disconnection handling.
- **Agent Config Isolation**: Agent configuration edits are bounded between `# BEGIN AUTOCONDUCK` and `# END AUTOCONDUCK`, with automated backups saved to `~/.autoconduck/backups/<agent>/<timestamp>.bak`.

## API Surface

- Served by FastAPI in `autoconduck/server/server_streaming.py` with routes in `server/server_routes.py`:
  - `/v1/chat/completions`: OpenAI-compatible completions intercepting the 3 pseudo-models (`autoconduck`, `autoconduck-budget`, `autoconduck-expensive`).
  - `/v1/models`: OpenAI-compatible catalog listing returning the pseudo-models.
  - `/v1/messages`: Anthropic-compatible message translation shim (`server/messages_api.py`).
  - `/stats`: AutoConduck audit log and cost savings metrics.
  - `/healthz`: Liveness and readiness endpoint.

## Key Modules

- `routing/`:
  - `dispatcher.py`: fast-path routing pipeline and model selector.
  - `fast_graph.py`: lightweight execution facade for backwards compatibility.
  - `semantic_router.py`: wrapper over `semantic-router` (Aurelio pillar) with fast/slow route embeddings.
  - `evaluator.py`: single-pass regex complexity scoring across 10 factors + boosts + hysteresis clamping.
  - `complexity.py` & `complexity_helpers.py`: complexity extractors and factor weights.
  - `pricing.py`: `select_closest()` closest-cost matching, EMA realized-cost blending, spend guard, and degraded model exclusions.
- `auth/`:
  - `auth.py`: provider credentials in `~/.autoconduck/auth.yaml` with automatic config migration.
  - `providers.py`: LiteLLM model qualification and API discovery.
- `launcher/`:
  - `launcher.py`, `launcher_procs.py`, `launcher_shims.py`: process discovery, launcher shims, refcounting, and binary PATH injection.
- `cli/`:
  - `cli.py` & `cli_launch.py`: CLI dispatch and agent setup.
- `presets/`:
  - `model_presets.py`, `presets_data.py`, `presets_ingest.py`, `presets_fallback.py`: runtime catalog data and LiteLLM ingestion.
- `server/`:
  - `server.py`, `server_routes.py`, `server_streaming.py`: FastAPI server and streaming lifecycle.
  - `messages_api.py`, `messages_models.py`, `messages_sse.py`: Anthropic shim translation.
- `orchestrator/`:
  - `graph.py`: LangGraph workflow (`recon → recon_subagent_pool → planner → subagents → compactor → executor`).
  - `recon.py`: deterministic target file discovery pre-planner.
  - `planner.py`: `TaskPlan` generation and `verified_context` bullet distillation.
  - `roles.py`: `RoleConfig` and role cards.
  - `subagents.py`: parallel LLM text analysis.
  - `compactor.py`: zero-LLM deterministic line deduplication and token truncation.
  - `executor_loop.py` & `tools.py`: tool execution loop (read, grep, glob, list, edit, write, bash) with `FileClaimRegistry`.
- `agents/`: adapters for supported coding agents (Claude Code, OpenCode, Pi, Aider, Cursor, Continue, Kilocode, Generic OpenAI).
- `tui/`: Textual terminal UI dashboard and interactive onboarding (`tui/onboarding/`).

## Development Guidelines & Gotchas

- **Test Suite**: Pure async unit tests using `pytest` (`python -m pytest` or `PYTHONPATH=. pytest`).
- **Graph Updates**: After modifying code, run `graphify update .` to update the AST knowledge graph.
- **TUI Keymap**: `Ctrl+C` is the single quit chord (Textual's default `Ctrl+Q` is disabled).
