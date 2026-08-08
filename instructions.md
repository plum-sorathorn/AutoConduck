# OpenCode integration

AutoConduck presents three selectable pseudo-models through its LiteLLM Proxy:

- `autoconduck` — balanced routing
- `autoconduck-budget` — cost-optimized routing
- `autoconduck-expensive` — quality-optimized routing

OpenCode should point its provider configuration at:

```text
http://127.0.0.1:11434/v1
```

Then select one of the three model names above. AutoConduck exposes the normal
OpenAI-compatible `POST /v1/chat/completions` and `GET /v1/models` endpoints;
LiteLLM handles provider abstraction, streaming, retries, usage, and
disconnect behavior.

## Start AutoConduck

For first-run setup, install the platform package and launch onboarding:

```bash
npm install -g autoconduck
```

The Textual `tui/` onboarding screen detects agents and model sources. To run
only the service, use:

```bash
autoconduck start --headless
```

The default listen address is `127.0.0.1:11434`; `--port N` and `--host H`
override it. Use `autoconduck edit` to reopen model selection.

## How pseudo-model routing works

Every request enters `dispatcher.py`, which sequences `semantic_router.py`,
`evaluator.py`, and an optional cheap LLM tiebreaker. Aurelio Semantic Router
classifies `fast_path` or `slow_path` and returns confidence. The evaluator
adds complexity `T_i`, stack-trace boost, hysteresis, and model-tier judgment.
Non-ambiguous decisions stay on the sub-5 ms fast path. Complex work can enter
the LangGraph DAG in `orchestrator/`: planner, parallel Send-based subagents,
compactor, and executor.

The aliases adjust the confidence bar as defined by the v2 blueprint:

| Model | Behavior |
|---|---|
| `autoconduck-budget` | Raises the confidence bar for escalation; prefers fast path and cheap models. |
| `autoconduck` | Uses evaluator thresholds as-is; balanced baseline. |
| `autoconduck-expensive` | Lowers the confidence bar; escalates at moderate confidence. |

The aliases do not identify an upstream provider model. `pricing.py` selects
the cheapest capable real model from the configured pool, using LiteLLM costs,
token correction, and degraded-routing safeguards.

## Dashboard and audit data

The Textual dashboard shows recent routing decisions, active agents, service
health, and savings. Select a decision and press `d` for drill-down. It shows:

- route (`fast_path` or `slow_path`) and confidence;
- complexity `T_i` and whether the stack-trace boost was applied;
- selected tier and pseudo-model;
- subtask count and the planner/parallel/compaction flow;
- real model used and cost.

Use `/` to filter, `j`/`k` to navigate, `p` to pause or resume routing, `e` to
edit models, `?` for all keys, and Ctrl+C to quit. Textual's default Ctrl+Q is
disabled; Ctrl+C is the single quit chord. The `?` overlay is also useful when
onboarding key bindings are not familiar.

For machine-readable routing and cost auditing, query:

```text
GET http://127.0.0.1:11434/stats
```

`/healthz` reports service health. The dashboard and `/stats` are the
AutoConduck-owned observability layer; LiteLLM remains responsible for native
provider usage and cost records.

## Automatic OpenCode configuration

The `agents/opencode.py` adapter can configure OpenCode during onboarding. It
backs up the original file before every change under:

```text
~/.autoconduck/backups/opencode/<timestamp>.bak
```

It writes only between the markers `# BEGIN AUTOCONDUCK` and
`# END AUTOCONDUCK`. Existing user configuration outside that block is never
rewritten, and the backup makes uninstall and restoration reversible.

## Server lifecycle

With launcher shims installed by `autoconduck install`, launching `opencode`
starts the server automatically and the last closing agent session releases
it. Without shims, run `autoconduck start --headless` before OpenCode and
`autoconduck stop` afterward. OpenCode still sees only the three pseudo-models
on `http://127.0.0.1:11434/v1`.

## Gateway providers

No hand-written YAML is needed for custom or gateway providers. In the TUI,
open the onboarding **Model Source** screen and choose **Custom endpoint** or a
named preset such as DevPass. Alternatively use `autoconduck edit`. The
`providers.py` module discovers models from a compatible `/v1/models` endpoint
and generates LiteLLM entries using `openai/<model>` with `api_base` and the
configured key. This supports any OpenAI-compatible gateway without a
provider-specific integration.

The `model_presets.py` and `providers.py` settings are stored under the
AutoConduck home directory. Keep provider secrets in environment variables or
the onboarding secret flow; do not commit them to agent configuration.

## Configuration checklist

1. Start AutoConduck and complete agent detection.
2. Select at least one usable model source and confirm discovery.
3. Choose a pseudo-model in OpenCode, not an upstream provider identifier.
4. Send a small request and verify the decision in the dashboard.
5. Use `d` or `/stats` when investigating route, confidence, or cost.

If the TUI cannot load, the command falls back to the headless proxy. The
OpenCode endpoint remains the same, so a headless deployment needs no separate
agent integration. Keep `AUTOCONDUCK_HOME`, `AUTOCONDUCK_PORT`, and
`AUTOCONDUCK_LOG_LEVEL` consistent with the service environment when using a
custom installation.
