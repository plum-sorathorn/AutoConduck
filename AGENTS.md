# AutoConduck working contract

**Generated:** 2026-08-25

AutoConduck is a local zero-overhead model router and task orchestrator for OpenAI-compatible coding agents. Routing and selection specifications are documented in `docs/design/dynamic-model-selection.md`.

## OVERVIEW

Project: **AutoConduck** (`0.3.5` in `pyproject.toml`)
Stack: Python >= 3.11 · FastAPI + uvicorn · LiteLLM · LangGraph (+ SQLite checkpointer) · Textual TUI · Pydantic v2 · LanceDB · ONNX Runtime · Outlines · httpx · tiktoken · PyYAML. End-user install is npm (`npm install -g autoconduck`); Python is the runtime.

Related context files (do not treat as source of truth over this contract): `README.md`, `PROJECT.md`, `docs/design/`.

## STRUCTURE

```text
autoconduck/          # runtime package
  routing/            # dispatcher, SLM planner, model pool, pricing
  orchestrator/       # dynamic LangGraph factory, session guard, tools
  server/             # FastAPI routes, streaming, Anthropic shim, turn guard
  knowledge/          # LanceDB RAG
  auth/               # provider credentials
  launcher/           # process discovery, shims, refcount
  cli/                # CLI dispatch
  config/             # Config model, load/save, path helpers, resolvers
  presets/            # runtime model catalog + LiteLLM ingest
  harnesses/          # Claude Code, OpenCode, Pi adapters
  tui/                # Textual dashboard + onboarding
  _compat/            # optional-dep fallbacks (llama.cpp, outlines, lancedb, onnx, sqlite)
tests/                # pytest (asyncio_mode=auto); tests/integration/
scripts/              # e2e smoke and probes
npm-packaging/        # per-platform wheel build (`build.py`)
docs/design/          # routing / architecture specs
```

Package-root utilities: `main.py` (entry), `stats.py`, `digest.py`, `jsonutil.py`, `update.py`, `model_presets.py` (re-export/compat).

## COMMANDS

| Action | Command |
| -------- | --------- |
| Install (dev) | `pip install -r requirements.txt` then `pip install -e .` |
| Install (users) | `npm install -g autoconduck` |
| Test | `python -m pytest` or `PYTHONPATH=. pytest` |
| Smoke | `python scripts/end_to_end_smoke.py` |
| Run TUI | `autoconduck` (falls back to headless if TUI import fails) |
| Run headless | `autoconduck start --headless [--port N] [--host H]` (default `127.0.0.1:11434`) |
| Daemon | `autoconduck start --headless --daemon` / `autoconduck stop [--port N]` |
| Graph | After code edits: `graphify update .` |
| NPM wheels | `python npm-packaging/build.py` (`--check` verifies without rebuild) |

Console entrypoints: `autoconduck = autoconduck.main:main` and `conduck = autoconduck.main:main`.

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
  - `update [--dry-run]` — update runtime model catalog.
  - `edit` — open interactive model list editor.
  - `reset [--force]` / `uninstall [--force]` — restore agent configs and remove shims.
  - `--version`, `--claude`, `--opencode`, `--pi` shortcuts.
- **Smoke test**: `python scripts/end_to_end_smoke.py` (bounded test against `/v1/messages`, `/v1/chat/completions`, `/v1/models`, `/stats`).

## Non-Negotiable Architecture Invariants

- **Sub-2ms Turn Guard**: `server/turn_guard.py` classifies active tool turns and stagnation states in <2ms with zero async/I/O/LLM overhead. Tool loop turns execute directly on capable models matching the required tool SLA.
- **Embedded SLM Task Architect**: Uses local Qwen 2.5 Coder 0.5B Instruct (Q4_K_M GGUF) to generate structured `ExecutionPlan` specifications under 100ms.
- **Dynamic DAG Compilation**: `orchestrator/dynamic_factory.py` compiles on-the-fly LangGraph `StateGraph` workflows tailored to the exact subtask dependencies and RAG requirements.
- **Fail-Soft Degradation**: Any SLM, LangGraph, tool, or checkpointer exception degrades gracefully to direct model dispatch—never surfacing a client-facing 500 error. Optional native deps (llama.cpp, Outlines, LanceDB, ONNX, LangGraph SQLite saver) are wrapped in `autoconduck/_compat/` so missing binaries still boot.
- **Dynamic SLA-Based Capability Matching**: `routing/model_pool.py` executes a zero-overhead "Query Optimizer" selection matrix. Instead of static tiers and pricing buckets, the SLM Architect generates strict Service Level Agreements (SLAs: `min_context`, `requires_tools`, `requires_reasoning`, `max_cost`) for every subtask. The proxy evaluates all models in the active pool and seamlessly routes to the absolute cheapest model that technically meets the multidimensional SLA requirements—guaranteeing peak cost-efficiency per-turn without arbitrary boundaries.
- **Thinking Progress Streaming**: `server/sse_streamer.py` renders SLM/orchestrator progress through `render_progress_event()`, used by both the OpenAI `server_chat.progress_stream` and Anthropic `server_messages` paths, gated by `selection.slow_stream_progress` / `AUTOCONDUCK_STREAM_PROGRESS`.
- **Stream Cancellation**: Honor `request.is_disconnected()` in streaming loops (`server/server_streaming.py`) for instantaneous client disconnection handling.
- **Session Cache Prefix Immutability**: `orchestrator/session_guard.py` guarantees byte-identical prefix preservation (turns 0 & 1) across 40+ turns for maximum upstream prompt cache hits, with compaction at the 80% context window ceiling.
- **Agent Config Isolation**: Agent configuration edits are bounded between `# BEGIN AUTOCONDUCK` and `# END AUTOCONDUCK`, with automated backups saved to `~/.autoconduck/backups/<agent>/<timestamp>.bak`.

## API Surface

- Served by FastAPI in `autoconduck/server/server_streaming.py` with routes in `server/server_routes.py`:
  - `/v1/chat/completions`: OpenAI-compatible completions with live `delta.reasoning_content` streams.
  - `/v1/models`: OpenAI-compatible catalog listing returning the pseudo-models.
  - `/v1/messages`: Anthropic-compatible message translation shim with `thinking_delta` reasoning blocks (`server/messages_api.py`).
  - `/v1/messages/count_tokens`: Anthropic-compatible token counting endpoint.
  - `/stats`: AutoConduck audit log, `ExecutionPlan` records, and cost savings metrics.
  - `/healthz`: Liveness and readiness endpoint.

## Key Modules

- `routing/`:
  - `dispatcher.py`: fast-path routing pipeline and model selector.
  - `slm_planner.py`: embedded Qwen 2.5 Coder 0.5B SLM task planner and `ExecutionPlan` generator.
  - `model_pool.py`: CapabilitySLA data model and dynamic Query Optimizer for filtering and selecting models by multidimensional SLA constraints.
  - `pricing.py`: `select_for_sla()` wrapper, spend guards, and health/degraded model exclusions.
  - `slm_downloader.py`: local SLM artifact fetch/setup.
- `orchestrator/`:
  - `dynamic_factory.py`: dynamic runtime LangGraph `StateGraph` compiler with Annotated fan-out state reducers.
  - `session_guard.py`: prompt-cache-friendly prefix immutability and 80% ceiling context compaction.
  - `roles.py`: `RoleConfig` and role cards.
  - `subagents.py`: parallel subtask analysts.
  - `executor_loop.py` & `tools.py`: tool execution loop (read, grep, glob, list, edit, write, bash).
  - `planner.py`: structured task planning prompts and validation (`SubTask`, output contracts).
  - `handoff.py`: `ExecutionHandoff` markdown/tool_calls for client harnesses.
  - `runner.py` / `skeletons.py` / `helpers.py`: graph run helpers, AST skeletons, string/model helpers.
- `knowledge/`:
  - `vector_store.py`: embedded LanceDB vector database for RAG code snippet retrieval.
- `server/`:
  - `server.py`, `server_routes.py`, `server_streaming.py`: FastAPI server and streaming lifecycle.
  - `turn_guard.py`: sub-2ms regex tool loop classifier.
  - `sse_streamer.py`: unified reasoning deltas streamer (DAG node transitions as thinking).
  - `messages_api.py`, `messages_models.py`, `messages_sse.py`: Anthropic shim translation.
- `auth/`:
  - `auth.py`: provider credentials in `~/.autoconduck/auth.yaml` with automatic config migration.
  - `providers.py`: LiteLLM model qualification and API discovery.
- `config/`:
  - `manager.py`: `load_config` / `save_config` / `get_config`.
  - `models.py`: `Config` pydantic model.
  - `paths.py`: `home_dir()`, `backups_dir()`, data roots under `~/.autoconduck`.
  - `resolver.py`: `resolve_api_key()`, `resolve_orchestrator_model()`.
- `launcher/`:
  - `launcher.py`, `launcher_procs.py`, `launcher_shims.py`: process discovery, launcher shims, refcounting, and binary PATH injection.
- `cli/`:
  - `cli.py` & `cli_launch.py`: CLI dispatch and agent setup.
- `presets/`:
  - `model_presets.py`, `presets_data.py`, `presets_ingest.py`, `presets_fallback.py`: runtime catalog data and LiteLLM ingestion.
- `harnesses/`: adapters for supported coding agent harnesses (Claude Code, OpenCode, Pi); `base.py` + `all_adapters()`.
- `tui/`: Textual terminal UI dashboard (`app.py`, `dashboard*.py`, `settings.py`) and interactive onboarding (`tui/onboarding/`).
- `_compat/`: fail-soft shims when optional native wheels are absent (`llama_fallback`, `outlines_fallback`, `lancedb_fallback`, `onnx_fallback`, `sqlite_checkpointer`).

## CODING STANDARDS

- **Language**: Python 3.11+, type hints, Pydantic v2 models for public contracts (`ExecutionPlan`, `CapabilitySLA`, `Config`, Anthropic message schemas).
- **Style**: Match neighboring modules; prefer small functions and fail-soft `try/except` over raising into the HTTP layer. Async I/O on the server/orchestrator path; Turn Guard stays sync and I/O-free.
- **Rules**: No dedicated ruff/black config in-repo; pytest `asyncio_mode = auto`, `testpaths = ["tests"]`. Keep `requirements.txt` mirrored with `pyproject.toml` runtime deps.
- **Surgical edits**: Do not “improve” adjacent formatting. Do not surface 500s for SLM/LangGraph/tool/checkpointer failures.

## WHERE TO LOOK

- **Source**: `autoconduck/`
- **Tests**: `tests/` (unit) and `tests/integration/`
- **Docs**: `docs/design/` (especially `dynamic-model-selection.md`), `README.md`, `PROJECT.md`
- **Packaging**: `pyproject.toml`, `npm-packaging/`

## Development Guidelines & Gotchas

- **Test Suite**: Pure async unit tests using `pytest` (`python -m pytest` or `PYTHONPATH=. pytest`).
- **Graph Updates**: After modifying code, run `graphify update .` to update the AST knowledge graph.
- **TUI Keymap**: `Ctrl+C` is the single quit chord (Textual's default `Ctrl+Q` is disabled). Shared conventions live in `tui/keymap.py`.
- **User data**: `~/.autoconduck/` (auth.yaml, backups, catalogs). Never write agent config outside `# BEGIN AUTOCONDUCK` / `# END AUTOCONDUCK` markers.
