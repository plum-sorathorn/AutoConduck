# AutoConduck

AutoConduck is a local, zero-overhead model router and task orchestrator for
OpenAI-compatible coding agents. It routes each turn to the **cheapest capable
model**, keeping ordinary decisions on a sub-5 ms fast path and escalating
complex, multi-file work to parallel subagents.

Coding agents select one of three pseudo-models:

- `autoconduck` — balanced defaults
- `autoconduck-budget` — cost-optimized routing
- `autoconduck-expensive` — quality-optimized routing

## Three pillars

```text
Agent → LiteLLM Proxy (streaming/provider abstraction)
          │
          ▼
   Aurelio Semantic Router ── confidence ──┐
          │                                │
      direct model                    cheap tiebreaker
          │                                │
          └──────────────┬─────────────────┘
                         ▼
                 LangGraph async DAG
              planner → Send subagents →
                 compactor → executor
```

1. **LiteLLM Proxy** provides provider abstraction, streaming, model usage,
   and the OpenAI-compatible `/v1/chat/completions` and `/v1/models` surface.
2. **Aurelio Semantic Router** classifies fast/slow routes with embedding-based
   confidence. `dispatcher.py`, `semantic_router.py`, and `evaluator.py` add
   the routing judgment without I/O or LLM calls on the fast path.
3. **LangGraph** runs the asynchronous subagent DAG through `orchestrator/`.
   The dashboard and Textual `tui/` make routing decisions inspectable.

The orchestrator planner and subagents resolve their model from the active
configuration via `resolve_orchestrator_model` in `config.py`; they do not use
a fixed runtime model. Gateway `api_base` values may be entered with or
without `/v1`: `normalize_api_base()` retains the raw value in
`config.yaml` and appends `/v1` at each LiteLLM call site when the host root has
no path.

## Quick start

```bash
npm install -g autoconduck
autoconduck                         # TUI onboarding and dashboard
autoconduck start --headless        # proxy-only service
```

The default endpoint is `http://127.0.0.1:11434/v1`. Configure agents through
the onboarding flow, then select any AutoConduck pseudo-model.

The LiteLLM-Proxy-backed surface serves `/v1/chat/completions` (intercepting
the three pseudo-models) and `/v1/models` (returning them), while AutoConduck
owns `/stats` for routing-decision audit and cost-saved data, plus `/healthz`.

### Custom provider registration

Custom providers are stored in `~/.autoconduck/config.yaml`, or in
`$AUTOCONDUCK_HOME/config.yaml`. For example:

```yaml
custom_models:
  - provider: llmgateway
    base_url: https://api.llmgateway.io
    api_key_env: LLMGATEWAY_API_KEY
    enabled: true
model_list:
  - model_name: deepseek-v4-flash
    provider: llmgateway
    tier: balanced
```

Use the TUI Model Source screen to register this shape; keep the API key in
the environment variable named by `api_key_env`.

## Seamless agent integration

1. **Install AutoConduck:** `npm install -g autoconduck` or `pip install -e .`.
2. **Choose agents:** run `autoconduck install [agents...]`. Configs are backed
   up to `~/.autoconduck/backups/<agent>/<timestamp>.bak`; `autoconduck uninstall`
   restores them.
3. **Choose models:** use the TUI onboarding **Model Source** screen or
   `autoconduck edit` for custom endpoints/model lists or presets such as DevPass.
4. **Launch the coding agent:** it sees the three pseudo-models, while the local
   server starts with the agent and stops when the last session closes.

`autoconduck install` adds refcounted launcher shims in `~/.autoconduck/bin`
and prepends it to `PATH`. A manually started server is never killed by shim
exit. Use `autoconduck start --headless --daemon` or `autoconduck stop`; shims
use the internal `autoconduck ensure` and `autoconduck release` commands.
GUI-only agents such as Cursor and Continue cannot be wrapped; start manually
with `autoconduck start --headless` for them.

## Development

```bash
pip install -r requirements.txt
pip install -e .
pytest
```

The project is Python 3.11+; `providers.py` supports generic OpenAI-compatible
gateways and model discovery, while `pricing.py` chooses capable models.

## TUI keymap highlights

`j`/`k` navigate, `/` filters, `d` opens routing drill-down, `p` pauses or
resumes routing, `e` edits models, `[ctrl+c]` quits, and `?` shows all keys.
Textual's default Ctrl+Q is disabled; Ctrl+C is the single quit chord.
