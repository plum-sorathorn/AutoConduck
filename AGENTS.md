# AutoConduck — AGENTS.md

AutoConduck is a local zero-overhead model router + task orchestrator for coding agents (Claude Code, OpenCode, Pi). Python runtime; end users install via npm (`npm install -g autoconduck`). Current version: 0.3.5.

## Commands
- Dev install: `pip install -r requirements.txt` then `pip install -e .`
- Test: `python -m pytest` (pytest `asyncio_mode=auto`; tests under `tests/`, integration under `tests/integration/`)
- Single file: `python -m pytest tests/test_model_pool.py -q`
- Smoke: `python scripts/end_to_end_smoke.py`
- Run: TUI `autoconduck`; headless `autoconduck start --headless [--port] [--host]` (default `127.0.0.1:11434`); daemon `--daemon`; stop `autoconduck stop [--port]`
- Version bump: `python scripts/bump_version.py --patch` (syncs pyproject.toml, `__init__.py`, npm package.json, README/AGENTS)
- Graph: after editing code run `graphify update .`
- NPM wheels: `python npm-packaging/build.py` (`--check` to verify without rebuild)

## Non-negotiable invariants
- **Fail-soft**: any SLM/LangGraph/tool/checkpointer/config exception degrades to direct dispatch — NEVER surface a 500 to the client. Optional native deps (llama.cpp, Outlines, LanceDB, ONNX, SQLite saver) are wrapped in `autoconduck/_compat/` so missing binaries still boot.
- **Turn Guard** (`server/turn_guard.py`) MUST stay synchronous, I/O-free, regex-only, <2ms. Do not add async/LLM/tool calls there.
- **Selection is O(models), synchronous, in-memory, sub-ms** — this is the routing hot path; do not add I/O or model calls inside it.
- **Agent config edits are bounded** by `# BEGIN AUTOCONDUCK` / `# END AUTOCONDUCK` markers, with backups in `~/.autoconduck/backups/`. Never write outside those markers.
- `requirements.txt` is strictly mirrored with `pyproject.toml` runtime deps.
- Session Guard (`orchestrator/session_guard.py`) keeps the prompt prefix byte-identical across turns (upstream cache hits) and compacts at the 80% context ceiling.

## Model selection (`routing/model_pool.py`) — read before touching routing
The router is a **"fit-gate then cheapest"** selector, not a spend meter. `select_by_sla()` does:
1. Filter: enabled/undegraded/excluded → tools → reasoning → context → capability floor → cost.
2. **Capability floor** uses 4-dim `capability_vector` (reasoning, tool_reliability, code_quality, latency_class) scored by `capability_fit()` = min-over-dominant-dims (weight>0.25) + 0.1*weighted_sum, weighted per SLM `task_type` via `TASK_TYPE_WEIGHTS`. Legacy models without a vector fall back to scalar `capability_score`.
3. Sort remaining by **absolute cost ascending**, pick cheapest (unless pseudo_model contains "expensive").
4. **`max_price_usd_per_mtok` is an OPT-IN per-selection price cap** (USD per 1M tokens), NOT per-minute and NOT a time-based meter. Disabled by default (`selection.path_price_cap_usd_per_mtok` = `{}`). If it empties the pool it falls back to cheapest available with `fallback_reason="price_cap_emptied_pool"`.
5. Dispatcher `_select_planned` tightens the floor by plan confidence: `floor = min(base + 0.15*(1-confidence), 0.6)`. Low confidence → higher (more capable) floor; never overrides the price cap.
- Explainability flows through `SelectionInfo`/`RoutingDecision` into `/stats`: candidates_considered, candidates_excluded_by, binding_constraint, capability_fit_applied, binding_capability_dim, spend_cap_engaged, fallback_reason.

## Turn Guard — do not regress
Healthy tool loops (even touching many files / many turns) route to `DIRECT_ACTIVE_TIER` — the client drives its own loop, no replanning. Genuine stagnation is ONLY: 3+ IDENTICAL consecutive calls OR 2+ consecutive errors → `ESCALATE_SLM`. A prior "complexity drift" escalation (>10 files / >30 turns) was **removed** because it caused runaway replanning cost (the grok-4.6 incident). Do NOT re-add file-count / turn-count escalation for healthy loops.

## C6 empirical scoring is deferred
Do NOT build `model_scores.json` / empirical success-weighted scoring yet. It is gated on real usage data and must stay inert/off until then. Routing must remain purely static + the live per-turn SLM signal (`plan.confidence`).

## Structure / entrypoints
- `autoconduck/` runtime package; `main.py` entry; `stats.py` = **write-only** usage accounting for `/stats` (NOT consumed by routing).
- `routing/`: `dispatcher.py` (route + `_select_planned`), `slm_planner.py` (`ExecutionPlan`), `model_pool.py` (`CapabilitySLA` + selection), `pricing.py`, `slm_downloader.py`.
- `orchestrator/`: `dynamic_factory.py` (LangGraph DAG), `session_guard.py`, `executor_loop.py` + `tools.py`, `handoff.py`, `subagents.py`.
- `server/`: `server_routes.py` (routes), `server_streaming.py`, `turn_guard.py`, `messages_api.py` (Anthropic shim), `sse_streamer.py`.
- `config/`: `models.py` (`Config`/`SelectionConfig` pydantic), `manager.py`, `resolver.py`, `paths.py`.
- `knowledge/` (LanceDB RAG); `auth/`, `launcher/`, `cli/`, `presets/`, `harnesses/`, `tui/`, `_compat/`.

## Gotchas
- TUI quit chord is **Ctrl+C** (Textual default Ctrl+Q is disabled); keymap in `tui/keymap.py`.
- User data lives under `~/.autoconduck/` (auth.yaml, backups, catalogs, run/).
- Tests are pure async unit tests (`asyncio_mode=auto`); no ruff/black config in-repo — match neighboring module style and keep edits surgical (don't reformat unrelated lines).
