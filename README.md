# AutoConduck

AutoConduck is a local, zero-overhead model router and task orchestrator for
coding agents. It routes each turn to the model whose cost is
**CLOSEST** to a task-value score computed on the fly, keeping ordinary decisions
on a sub-5 ms fast path and escalating complex, multi-file work to parallel
subagents.

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
    tier: balanced  # advisory/display-only
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

## How AutoConduck works internally

### Request lifecycle

1. Claude Code, OpenCode, or another client calls the OpenAI-compatible
   LiteLLM Proxy at `127.0.0.1:11434`, or the Anthropic `/v1/messages` shim in
   `messages_api.py`.
2. Requests for `autoconduck`, `autoconduck-budget`, or `autoconduck-expensive`
   are intercepted and handed to `dispatcher.py` through `_route_target`.
3. `dispatcher.route()` runs the synchronous, zero-I/O fast path: 
   `semantic_router.route()` → `evaluator.score()` → an optional tiebreaker.
   This path is designed to stay under 5 ms.
4. A `FAST` decision lets `pricing.select_closest()` match a model to the
   task's computed complexity value; `RoutingDecision.model` is set by the
   dispatcher. A `SLOW` decision hands the whole request to the LangGraph
   workflow in `orchestrator/graph.py`.
5. Errors anywhere degrade to the fast path, never to a client-facing API
   error.

### Semantic router (`semantic_router.py`)

The router has `fast_path` and `slow_path` routes. Fast examples include
typos, renames, “where is X defined”, docstrings, and comments. Slow examples
include “refactor the application”, “build a feature”, “implement multi-file
change”, “migrate the database”, “redesign the API”, “write integration tests
for the whole system”, and “optimize the performance of the codebase”.

The optional real embedding layer uses `semantic_router` and
`FastEmbedEncoder`, guarded by `try/except`. Without that package, keyword and
token-overlap matching over the examples is used; no match returns
`RouteMatch("fast_path", 0.0)`.

### Evaluator (`evaluator.py`)

The evaluator is pure math on the fast path:

```text
complexity = 0.15·length + 0.10·refs + 0.25·structural + 0.10·files
           + 0.15·keyword_domain + 0.15·edit_intent + 0.10·multi_step
```

Each term is clamped so the final score is at most 1. Confidence is
`max(0, complexity × 0.75)`, with a `+0.25` stack-trace boost when the prompt
contains a traceback. After an escalation (previous decision ≥ `0.80`) and
without a new stack trace, hysteresis clamps complexity to `≤ 0.50`, preventing
the system from staying hot.

The ambiguous zone is `[0.55, 0.70]`: confidence in that zone (or below its
low bound) invokes a cheap two-token LLM tiebreaker (“Reply only FAST or
SLOW”); exceptions return `fast`. The current slow rule is
`complexity >= 0.6 OR (route == slow_path AND confidence >= 0.70)`, so complex
prompts escalate even when the semantic router says fast. The budget pseudo-model multiplies confidence by `1.15`; expensive multiplies
it by `0.85`. Both instead shift the selection target: budget applies a `−0.20`
bias and expensive a `+0.20` bias.

### Pricing (`pricing.py`)

Prices resolve config → `litellm.model_cost` → `pricing_fallback.json`.
`scaled_cost` is `ln(1 + price)` relative to the pool maximum. An EMA realized-
cost blend (α=`0.1`, requiring at least 3 samples) replaces flat price for
matching. `select_closest()` picks the nearest `|scaled_cost − target|`, ties
going to the cheaper model; degraded models (over 20% errors in 300s) are
excluded, and errors or an all-degraded pool fall back to `cheapest_enabled()`.
`target_scaled_cost` maps task value using `value_to_cost_gamma` and pseudo-model
bias (budget `−0.20`, expensive `+0.20`). `select()` and
`select_model_by_tier()` are deprecated wrappers.

### LangGraph orchestrator (`orchestrator/`)

The graph is `planner → subagent_pool → (compactor | end) → executor → END`.
Its `after_plan` edge retries or falls back; if the plan is `None` twice, the
graph returns `None` and the request uses the fast path.

`build_task_plan()` uses a few-shot, strict JSON prompt to produce a `TaskPlan`
of `SubTask{id, goal, scope, output_contract, constraints, depends_on,
verified_context, read_budget}`. The planner reads the in-scope files itself and
distills their contents into up to eight short `verified_context` bullets (each
under 15 words). Those bullets are injected into every subagent prompt, so
subagents do not re-read the files.

Independent dependency waves run concurrently with `asyncio.gather`; the Send
envelope is retained for map/reduce API compatibility. A subagent whose task
depends on sibling tasks receives those siblings' outputs concatenated verbatim
as `CONTEXT FROM SIBLING TASKS`; there is no summarization or rewriting at this
boundary. Subagent outputs are then ordered by dependency count and merged.

The compactor is deterministic and LLM-free, not a synthesis model call. It
drops lines whose `file:line` references appeared in an earlier report, drops
exact duplicate lines, and truncates the result at about 950 words (about 1k
tokens) while preserving complete lines. The result is stored in
`state.compacted`. Phase bands are planner `[0.55,0.85]`, subagents
`[0.10,0.55]`, and executor `[0.35,0.70]`, each with on-the-fly targets from
task value, subtask prompt/role/plan breadth and planner `budget_hint`, or
compactor-summary complexity and integration count. The compactor costs
nothing.

The executor prompt is assembled exactly as:

```text
Original request:
<original user messages, passed through untouched>

Analyst summary:
<state.compacted>
```

The original request is never rewritten by compaction. The expensive executor
performs the final synthesis from the raw request and the deduplicated,
truncated analyst lines; this is the only LLM-driven synthesis step in the
pipeline.

In short, between phases the pipeline only (a) distills file contents into
`verified_context` at the planner, (b) joins sibling outputs verbatim, and (c)
deterministically deduplicates and truncates in the compactor. There is no
inter-phase LLM summarization layer; final synthesis is the executor's job.

Every planner and subagent call is a plain `litellm` completion to the resolved
gateway model, not a subprocess spawn of Claude Code. The `agents/` adapters
only configure external CLIs through `launcher.py` shims.

The subagent prompt template is:

```text
ROLE: You are a read-only file analyst. You do not propose fixes or write code.
TASK: {goal}
FILES IN SCOPE (only these): {scope}
REQUIRED OUTPUT FORMAT: {output_contract}
DO NOT: {constraints}
CONTEXT FROM SIBLING TASKS: {upstream_summaries}
VERIFIED CONTEXT (do not re-investigate): {verified_context bullets}
TOOL BUDGET: You may make at most {read_budget} additional file reads/tool calls beyond what's given above. Work with what you have first.
VERIFY BEFORE RETURNING: {verify hooks}
```

### Configuration and observability

Configuration is `config.yaml` under `~/.autoconduck` or `$AUTOCONDUCK_HOME`:

| Tunable | Default |
| --- | ---: |
| `ambiguous_low` | 0.55 |
| `ambiguous_high` | 0.70 |
| `escalation_threshold` | 0.80 |
| `hysteresis_floor` | 0.50 |
| `stack_trace_boost` | 0.25 |
| `ema_alpha` | 0.1 |
| `degraded_error_rate` | 0.20 |
| `degraded_window_s` | 300 |
| `pseudo_model` | `autoconduck` |
| `phase_bands` | planner, subagent, executor ranges |
| `complexity_weights` | seven evaluator factor weights |
| `value_to_cost_gamma` | 1.0 |
| `pseudo_bias_budget` / `pseudo_bias_expensive` / `pseudo_bias_enabled` | -0.20 / 0.20 / true |
| `ema_min_samples` | 3 |
| `expose_value_in_stats` | true |
| `model_list` | id, prices, enabled; `tier` is advisory/display-only |

`/stats` is the routing audit log (path, model, latency); `/healthz` is the
liveness endpoint. To see generated prompts, run:

```bash
AUTOCONDUCK_LOG_LEVEL=DEBUG autoconduck start --headless
```

This prints `SUBAGENT PROMPT [...]` and `PLANNER PROMPT` blocks, plus an INFO
`route=… model=… ms=…` line for each request. Other environment overrides are
`AUTOCONDUCK_HOME` and `AUTOCONDUCK_PORT`.

To trigger the slow path, make the request structurally complex: mention at
least three files, multiple symbols, or use phrasing such as “refactor X
across these files”, “implement multi-file change”, or “write integration
tests for the whole system”.

### Module map

`dispatcher.py` (sequence) · `semantic_router.py` · `evaluator.py` ·
`pricing.py` · `config.py` (`resolve_orchestrator_model`, `qualify_model`,
`resolve_api_key`, `normalize_api_base`) · `providers.py` · `launcher.py`
(shims and ensure/release refcounting) · `messages_api.py` (Anthropic shim) ·
`orchestrator/{graph,planner,subagents,compactor}.py` · `agents/` (external
CLI adapters) · `tui/` (Textual dashboard).

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
