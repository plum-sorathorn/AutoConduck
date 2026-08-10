# AutoConduck working contract

AutoConduck is a local zero-overhead model router and task orchestrator for OpenAI-compatible coding agents. Routing and selection are specified in `docs/design/dynamic-model-selection.md` (plus `docs/design/tuning.md`).

## Commands

- Requires Python >= 3.11. Dev setup: `pip install -r requirements.txt` then `pip install -e .`. End users install via npm (`npm install -g autoconduck`); see `npm-packaging/build.py` (builds per-platform wheels; `--check` verifies without rebuilding).
- `requirements.txt` and the pyproject runtime dependencies are in sync and must stay mirrored. There is no dev-dependencies group.
- Run: `autoconduck start --headless [--port N] [--host H]`; serves the LiteLLM-backed OpenAI-compatible endpoint on 127.0.0.1:11434 by default.
- Background service: `autoconduck start --headless --daemon`; stop with `autoconduck stop`.
- Console script: `autoconduck = autoconduck.main:main` (`autoconduck/__main__.py` exists too).
- Full CLI: `install [agents...]`, `start [--headless] [--daemon] [--supervisor (hidden)] [--port N] [--host H]`, `stop [--port]`, `ensure` / `release` (internal launcher lifecycle commands, take `--port`), `stats [--json] [--days N] [--reset] [--force]`, `tune --mode`, `update [--dry-run]`, `edit`, `uninstall [--force]`, `--version`, plus `--claude` / `--opencode` / `--pi` shortcuts.
- No arguments opens the TUI onboarding/dashboard (`autoconduck/tui/`), falling back to headless mode if the TUI import fails.
- Smoke-test a running server with `python scripts/end_to_end_smoke.py` (bounded manual test against `/v1/messages`, `/v1/chat/completions`, `/v1/models`, `/stats`).

## Non-negotiable architecture invariants

- Fast path stays under 5 ms (`tests/test_dispatcher.py` asserts < 0.005): `semantic_router.py` and `evaluator.py` are sync-only (zero async); routing decisions perform no I/O or LLM calls.
- Confidence below the tunable ambiguous threshold (default 0.55–0.70) goes to a cheap LLM tiebreaker.
- Any LangGraph, orchestrator, or schema error degrades to the fast path, never a client-facing API error.
- Model selection is dynamic closest-cost matching, not fixed tiers: `ModelEntry.tier` is advisory/display-only. Orchestrator phases use `PHASE_BANDS` (defined in `orchestrator/graph.py`) — planner [0.55,0.85], subagent [0.10,0.55], executor [0.35,0.70] — with on-the-fly targets from task value, `budget_hint`, role weights, breadth damping, and compactor-summary complexity. `RoutingDecision.model` is populated on the fast path and is never None there.
- Honor `request.is_disconnected()` for instant cancellation (used in `main.py` streaming loops).
- Agent config changes are limited to `# BEGIN AUTOCONDUCK` … `# END AUTOCONDUCK`, with backups at `~/.autoconduck/backups/<agent>/<timestamp>.bak`.

## API surface

- The FastAPI app in `autoconduck/main.py` serves the LiteLLM-backed endpoint: `/v1/chat/completions` and `/v1/models` (the latter returning exactly the three pseudo-models from `PSEUDO_MODELS` in `messages_api.py`: `autoconduck`, `autoconduck-budget`, `autoconduck-expensive`), plus AutoConduck-owned `/stats` and `/healthz`, and an Anthropic-compatible `/v1/messages` shim (covered by `tests/test_messages_api.py`).

## Key modules

- `dispatcher.py`: thin sequence of semantic router → evaluator → optional tiebreaker; keep its own logic minimal.
- `semantic_router.py`: wrapper over the external `semantic-router` library (the "Aurelio" pillar) with fast/slow routes and confidence.
- `evaluator.py`: `T_i ∈ [0,1]`, stack-trace boost `+0.25`, hysteresis clamp `≤0.50` after escalation `≥0.80`, and ambiguous-zone handling.
- `pricing.py`: `select_closest()` closest-cost model matching on ln(1+cost) scaled domain (ties→cheaper, degraded exclusion, `cheapest_enabled()` deterministic fallback), `target_scaled_cost()` with pseudo-model bias (±0.20) and `value_to_cost_gamma`, EMA realized-cost blend α=0.1 (≥ `ema_min_samples` = 5 samples) via `_entry_effective_value`, failover after trailing error rate >20%, `pricing_fallback.json`; `select()`/`select_model_by_tier()` are deprecated thin wrappers.
- `providers.py`: LiteLLM `openai/<model>` plus `api_base`, `/v1/models` discovery.
- `config.py`: active model configuration, `resolve_orchestrator_model`, `normalize_api_root` (adds `/v1` for host-root gateway URLs at LiteLLM call sites), `qualify_model()` (`openai/<id>`), `resolve_api_key()` (literal keys, env names, or literal fallback values).
- `launcher.py`: launcher shims, `ensure_server`/`release_server` refcounting, PATH integration, `real_binary_path`, and `stop_server`.
- `orchestrator/`: LangGraph `graph.py` (owns `PHASE_BANDS`) → Send-based `subagents.py` → `compactor.py` → executor; `planner.py` owns `TaskPlan`.
- `model_presets.py`: runtime catalog ingested from the installed litellm registry (`litellm.model_cost` via `_ingest_litellm_costs`).
- `agents/`: agent adapters (`opencode.py`, `claude_code.py`, `cursor.py`, `continue_dev.py`, `aider.py`, `generic_openai.py`, `kilocode.py`, `pi.py`).
- `messages_api.py`, `main.py`, and Textual `tui/` (Ctrl+C is the single quit chord per `tui/keymap.py`; Textual Ctrl+Q is disabled) provide the shim, CLI, and onboarding/monitoring.

## State, environment, and scope

- State lives under `~/.autoconduck/` or `$AUTOCONDUCK_HOME`.
- Environment overrides (read in `config.py`): `AUTOCONDUCK_HOME`, `AUTOCONDUCK_PORT`, `AUTOCONDUCK_LOG_LEVEL`.
- Gitignored: `.autoconduck/`, `backups/`, `graphify-out/`, `build/`, and `*.egg-info/`.
- LiteLLM owns caching and native cost logging; `/stats` is the audit surface.
- Do not add legacy routing, state, caching, or telemetry layers. The LiteLLM-backed endpoint is the API surface; dispatcher placement relative to LangGraph follows the design's open integration verification item.

## Keeping the model catalog fresh

- The runtime catalog is derived from the installed `litellm` model registry, so upgrading LiteLLM keeps the dropdown and TUI catalog current.
- A weekly GitHub Actions workflow (`.github/workflows/refresh-catalog.yml`, cron `0 6 * * 1`) refreshes the static `docs/model_catalog.md` snapshot and opens a pull request when it changes.
- Check the snapshot locally with `python scripts/refresh_catalog.py --check`; run `python scripts/refresh_catalog.py` before npm publish.

## Gotchas

- Run `pytest` from the repository root; if the package is not installed, use `PYTHONPATH=.`. Tests are pure async unit tests with no network or services (`asyncio_mode = "auto"`, `testpaths = ["tests"]`); run a single file with `pytest tests/test_<name>.py`.
- No lint, format, or typecheck config exists — pytest is the only verification gate (the only CI workflow is the catalog refresh).
- After editing code, run `graphify update .` to keep the code graph current.
- The TUI uses Ctrl+C as its single quit chord; Textual Ctrl+Q is disabled.
