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

- **Sub-2ms Turn Guard**: `server/turn_guard.py` classifies active tool turns and stagnation states in <2ms with zero async/I/O/LLM overhead. Tool loop turns execute directly on active model tiers.
- **Embedded SLM Task Architect**: Uses local Qwen 2.5 Coder 0.5B Instruct (Q4_K_M GGUF) to generate structured `ExecutionPlan` specifications under 100ms.
- **Dynamic DAG Compilation**: `orchestrator/dynamic_factory.py` compiles on-the-fly LangGraph `StateGraph` workflows tailored to the exact subtask dependencies and RAG requirements.
- **Fail-Soft Degradation**: Any SLM, LangGraph, tool, or checkpointer exception degrades gracefully to direct model dispatch—never surfacing a client-facing 500 error.
- **Autonomous 3-Tier Model Matching**: Models are classified into `cheap_fast` (< $0.50/1M), `balanced` ($0.50–$4.00/1M), and `frontier_reasoning` (> $4.00/1M) tiers with logarithmic cost scaling `ln(1 + cost)` and EMA realized-cost blending.
- **Stream Cancellation**: Honor `request.is_disconnected()` in streaming loops (`server/server_streaming.py`) for instantaneous client disconnection handling.
- **Session Cache Prefix Immutability**: `orchestrator/session_guard.py` guarantees byte-identical prefix preservation (turns 0 & 1) across 40+ turns for maximum upstream prompt cache hits, with compaction at the 80% context window ceiling.
- **Agent Config Isolation**: Agent configuration edits are bounded between `# BEGIN AUTOCONDUCK` and `# END AUTOCONDUCK`, with automated backups saved to `~/.autoconduck/backups/<agent>/<timestamp>.bak`.

## API Surface

- Served by FastAPI in `autoconduck/server/server_streaming.py` with routes in `server/server_routes.py`:
  - `/v1/chat/completions`: OpenAI-compatible completions with live `delta.reasoning_content` streams.
  - `/v1/models`: OpenAI-compatible catalog listing returning the pseudo-models.
  - `/v1/messages`: Anthropic-compatible message translation shim with `thinking_delta` reasoning blocks (`server/messages_api.py`).
  - `/stats`: AutoConduck audit log, `ExecutionPlan` records, and cost savings metrics.
  - `/healthz`: Liveness and readiness endpoint.

## Key Modules

- `routing/`:
  - `dispatcher.py`: fast-path routing pipeline and model selector.
  - `slm_planner.py`: embedded Qwen 2.5 Coder 0.5B SLM task planner and `ExecutionPlan` generator.
  - `model_pool.py`: 3-tier autonomous model selector and capability filtering.
  - `pricing.py`: `select_closest()`, `select_for_tier()`, EMA realized-cost blending, spend guard, and degraded model exclusions.
- `orchestrator/`:
  - `dynamic_factory.py`: dynamic runtime LangGraph `StateGraph` compiler with Annotated fan-out state reducers.
  - `session_guard.py`: prompt-cache-friendly prefix immutability and 80% ceiling context compaction.
  - `roles.py`: `RoleConfig` and role cards.
  - `subagents.py`: parallel subtask analysts.
  - `executor_loop.py` & `tools.py`: tool execution loop (read, grep, glob, list, edit, write, bash).
- `knowledge/`:
  - `vector_store.py`: embedded LanceDB vector database for RAG code snippet retrieval.
- `server/`:
  - `server.py`, `server_routes.py`, `server_streaming.py`: FastAPI server and streaming lifecycle.
  - `turn_guard.py`: sub-2ms regex tool loop classifier.
  - `sse_streamer.py`: unified reasoning deltas streamer.
  - `messages_api.py`, `messages_models.py`, `messages_sse.py`: Anthropic shim translation.
- `auth/`:
  - `auth.py`: provider credentials in `~/.autoconduck/auth.yaml` with automatic config migration.
  - `providers.py`: LiteLLM model qualification and API discovery.
- `launcher/`:
  - `launcher.py`, `launcher_procs.py`, `launcher_shims.py`: process discovery, launcher shims, refcounting, and binary PATH injection.
- `cli/`:
  - `cli.py` & `cli_launch.py`: CLI dispatch and agent setup.
- `presets/`:
  - `model_presets.py`, `presets_data.py`, `presets_ingest.py`, `presets_fallback.py`: runtime catalog data and LiteLLM ingestion.
- `agents/`: adapters for supported coding agents (Claude Code, OpenCode, Pi, Aider, Cursor, Continue, Kilocode, Generic OpenAI).
- `tui/`: Textual terminal UI dashboard and interactive onboarding (`tui/onboarding/`).

## Development Guidelines & Gotchas

- **Test Suite**: Pure async unit tests using `pytest` (`python -m pytest` or `PYTHONPATH=. pytest`).
- **Graph Updates**: After modifying code, run `graphify update .` to update the AST knowledge graph.
- **TUI Keymap**: `Ctrl+C` is the single quit chord (Textual's default `Ctrl+Q` is disabled).
