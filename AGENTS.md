# AutoConduck v2 working contract

AutoConduck is a local zero-overhead model router and task orchestrator for OpenAI-compatible coding agents. `AutoConduck-Blueprint-v2.md` is authoritative.

## Commands

- Requires Python >= 3.11. Dev setup: `pip install -r requirements.txt` then `pip install -e .`. End users install via npm (`npm install -g autoconduck`); see `npm-packaging/build.py`.
- `requirements.txt` must mirror pyproject runtime deps; it is currently stale (missing `semantic-router`, `langgraph`, and the `[proxy]` extra on `litellm`). There is no dev-dependencies group.
- Run: `autoconduck start --headless [--port N] [--host H]`; serves the LiteLLM-Proxy-backed OpenAI-compatible endpoint on 127.0.0.1:11434 by default.
- Background service: `autoconduck start --headless --daemon`; stop with `autoconduck stop`.
- Console script: `autoconduck = autoconduck.main:main` (`autoconduck/__main__.py` exists too).
- CLI: `autoconduck install [agents...]`, `start [--headless] [--daemon] [--port N] [--host H]`, `stop [--port]`, `edit`, `uninstall [--force]`, and `--version`.
- Internal launcher lifecycle commands (take `--port`): `autoconduck ensure` and `autoconduck release`.
- No arguments opens TUI onboarding/dashboard, falling back to headless mode if the TUI import fails.
- No CI, lint, or formatting config exists — `pytest` is the only verification gate. Tests are pure async unit tests with no network or services (`asyncio_mode = "auto"`, `testpaths = ["tests"]`); run a single file with `pytest tests/test_<name>.py`.

## Non-negotiable architecture invariants

- Fast path stays under 5 ms: `semantic_router.py` and `evaluator.py` are sync-only (zero async); routing decisions perform no I/O or LLM calls.
- Confidence below the tunable ambiguous threshold (default 0.55–0.70) goes to a cheap LLM tiebreaker.
- Any LangGraph, orchestrator, or schema error degrades to the fast path, never a client-facing API error.
- Honor `request.is_disconnected()` for instant cancellation.
- Agent config changes are limited to `# BEGIN AUTOCONDUCK` … `# END AUTOCONDUCK`, with backups at `~/.autoconduck/backups/<agent>/<timestamp>.bak`.
- The LiteLLM Proxy pillar serves `/v1/chat/completions` and `/v1/models`, the latter returning the three pseudo-models (`autoconduck`, `autoconduck-budget`, `autoconduck-expensive`). AutoConduck owns `/stats` and `/healthz` on top, plus an Anthropic `/v1/messages` compatibility shim.

## Key modules

- `dispatcher.py`: thin sequence of semantic router → evaluator → optional tiebreaker; keep its own logic minimal.
- `semantic_router.py`: Aurelio wrapper with fast/slow routes and confidence.
- `evaluator.py`: `T_i ∈ [0,1]`, stack-trace boost `+0.25`, hysteresis clamp `≤0.50` after escalation `≥0.80`, and ambiguous-zone handling.
- `pricing.py`: `litellm.model_cost`, EMA token correction `α=0.1`, `ln(1+cost)` scaling, and failover after a trailing error rate above 20%; falls back to `pricing_fallback.json`.
- `providers.py`: LiteLLM `openai/<model>` plus `api_base`, `/v1/models` discovery.
- `config.py`: active model configuration, `resolve_orchestrator_model`, and `normalize_api_base` (adds `/v1` for host-root gateway URLs at LiteLLM call sites). Runtime models use `qualify_model()` (`openai/<id>`); API keys use `resolve_api_key()` for literal keys, env names, or literal fallback values.
- `launcher.py`: launcher shims, `ensure_server`/`release_server` refcounting, PATH integration, `real_binary_path`, and `stop_server`.
- `orchestrator/`: LangGraph `graph.py` planner → Send-based `subagents.py` → `compactor.py` → executor; `planner.py` owns `TaskPlan`.
- `messages_api.py`: Anthropic-messages compatibility shim on the API surface (covered by `tests/test_messages_api.py`).
- `config.py`, `model_presets.py`, `agents/`, `main.py`, and Textual `tui/` provide configuration, adapters, CLI, onboarding, and monitoring; the TUI uses Ctrl+C as its single quit chord (Textual Ctrl+Q is disabled).

## State, environment, and scope

- State lives under `~/.autoconduck/` or `$AUTOCONDUCK_HOME`.
- Environment overrides: `AUTOCONDUCK_HOME`, `AUTOCONDUCK_PORT`, and `AUTOCONDUCK_LOG_LEVEL`.
- Gitignored: `.autoconduck/`, `backups/`, `graphify-out/`, `build/`, and `*.egg-info/`.
- LiteLLM owns caching and native cost logging; `/stats` is the audit surface.
- Do not add legacy routing, state, caching, or telemetry layers. The LiteLLM Proxy is the API surface; dispatcher placement relative to LangGraph follows the blueprint's open integration verification item.
