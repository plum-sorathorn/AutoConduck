# AutoConduck — Unified Master Blueprint (v2)

## 1. Objective & Overview

AutoConduck is a local, zero-overhead model router and task orchestrator for open-source AI coding agents (Claude Code, OpenCode, Aider, Continue, Cursor, Kilo Code, and any OpenAI-compatible agent).

It presents itself to coding agents as three selectable pseudo-models:

- `autoconduck` (balanced)
- `autoconduck-budget` (cost-optimized)
- `autoconduck-expensive` (quality-optimized)

AutoConduck transparently routes every request turn to the cheapest real model capable of handling it, while automatically escalating complex, multi-file tasks to an asynchronous parallel subagent DAG.

**Design philosophy for v2:** don't rebuild commodity infrastructure. AutoConduck is a *thin, opinionated layer* on top of three mature open-source libraries. The only code that is genuinely AutoConduck's own is the routing judgment (confidence scoring, hysteresis, ambiguous-zone handling) and the coding-specific DAG planning prompts. Everything else — provider abstraction, streaming, cost tracking, parallel execution, graph state — is delegated.

**Provider coverage note:** AutoConduck does not require a model source to be a "named provider" in LiteLLM's registry. LiteLLM's generic `openai/<model>` + `api_base` mechanism lets any OpenAI-compatible endpoint — including flat-rate gateways like DevPass (LLM Gateway) — be added as a routing target through config alone. See Section 6.5 for how this is surfaced in the TUI.

---

## 2. Foundation Libraries ("The Three Pillars")

```
┌─────────────────────────────────────────────────────────────┐
│  Pillar 1: LiteLLM Proxy                                    │
│  Provider abstraction, SSE streaming, disconnect handling,  │
│  cost/usage data, OpenAI-compatible endpoint, pre-call hooks│
└──────────────────────────────┬──────────────────────────────┘
                                │ pre-call hook
                                ▼
┌─────────────────────────────────────────────────────────────┐
│  Pillar 2: Semantic Router (Aurelio)                         │
│  Embedding-based route classification + confidence score.   │
│  Replaces hand-written regex/keyword tiers.                 │
└───────────────┬───────────────────────────────┬─────────────┘
                │ high confidence                │ low confidence
                ▼                                ▼
     ┌─────────────────────┐          ┌───────────────────────┐
     │  Route directly      │          │  Cheap LLM tiebreaker │
     │  (Fast or Slow)      │          │  (ambiguous zone only)│
     └──────────┬───────────┘          └───────────┬───────────┘
                │                                    │
   (Fast-Path)  ▼                                    ▼ (routes to Fast or Slow)
┌──────────────────────────────┐  ┌───────────────────────────┐
│ LiteLLM model selection      │  │ Pillar 3: LangGraph        │
│ direct single-model call,    │  │ Multi-agent DAG: planner → │
│ streamed back to client      │  │ parallel subagents (Send)  │
│                               │  │ → compactor → executor     │
└──────────────────────────────┘  └───────────────────────────┘
```

None of AutoConduck's own code re-implements SSE streaming, provider auth, parallel task execution, or graph state — those are Pillar responsibilities. AutoConduck's own code is the **dispatcher** and the **scoring/hysteresis layer**, described below.

---

## 3. Distribution & Packaging Architecture

- **Engine Core:** Python 3.11+, built around `litellm[proxy]`, `semantic-router`, `langgraph`, `pydantic`, `textual`.
- **Standalone Compiled Binary:** Per OS/architecture (`darwin-arm64`, `linux-x64`, `win32-x64`) via PyInstaller or Nuitka. Bundle size will be larger than the original single-file design (~120–180 MB estimated) because of the added dependency weight from LangGraph/Semantic Router — this must be validated with an early prototype build, since it's the main risk this pillar-based design introduces to the "zero host dependency" goal.
- **npm / bun Distribution Shim:** Thin npm package (`autoconduck`) using `optionalDependencies` for platform binaries.
- **User Experience:**

  ```bash
  npm install -g autoconduck  # or: bun add -g autoconduck
  ```

  No local Python, pip, or virtualenv required by the end user.

---

## 4. User Interaction & Auto-Configuration Engine

### First Run (`autoconduck`)

Launches the keyboard-only TUI (see Section 9) directly into onboarding:

- **Agent Detection:** Scans filesystem/PATH for supported coding agents.
- **Model Ingestion:** User selects model sources (API keys or provider account presets). Live pricing pulled from LiteLLM's model cost registry.
- **Atomic Config Patching:** Before touching any agent config, back up the original to `~/.autoconduck/backups/<agent>/<timestamp>.bak`. Patches are written inside strictly delimited AutoConduck blocks so re-running setup never clobbers user-owned config.
- **Live Dashboard:** Proxy launches and TUI transitions into the live monitoring view.

### CLI Subcommands

- `autoconduck` — opens live dashboard if configured; a hotkey re-enters setup.
- `autoconduck start --headless` — background proxy only, no TUI (CI/systemd/launchd).
- `autoconduck edit` — re-opens model selection without re-patching agent configs.
- `autoconduck uninstall` — restores original agent configs from backups, stops daemon.

---

## 5. End-to-End System Data Flow

```
Client (Coding Agent)
   │ POST /v1/chat/completions {model: "autoconduck*"}
   ▼
LiteLLM Proxy ──▶ pre-call hook: dispatcher.py
   │
   ▼
dispatcher.py (AutoConduck's own code — thin)
   ├─ 1. semantic_router.py  ──▶ route match + confidence score (<5ms typical)
   ├─ 2. evaluator.py        ──▶ complexity score T_i, stack-trace boost, hysteresis clamp
   └─ 3. Ambiguous zone?     ──▶ cheap LLM tiebreaker (FAST/SLOW, 2-token response)
   │
   ├── [FAST-PATH] ──▶ LiteLLM model selection (pricing.py picks cheapest capable model)
   │                       │
   │                       ▼
   │                   Streamed back to client via LiteLLM Proxy
   │
   └── [SLOW-PATH] ──▶ LangGraph graph.invoke(state)
                           ├─ Node: planner        (TaskPlan Pydantic schema)
                           ├─ Node: subagent_pool   (LangGraph Send API, parallel fan-out)
                           ├─ Node: compactor       (structured merge, <1k tokens)
                           └─ Node: executor        (final model call via LiteLLM)
   │
   ▼
Streamed SSE chunks back to client (via LiteLLM Proxy in both paths)
```

---

## 6. Detailed Component Specifications

### 6.1 `dispatcher.py` — Thin Entry Point (replaces old `gatekeeper.py`)

This is intentionally the smallest file in the system. Its only job is sequencing calls to the three pillars and applying AutoConduck's own scoring logic on top of their outputs. It does not implement classification itself.

```python
def route(messages: list[Message], history: ConversationState) -> RoutingDecision:
    match = semantic_router.route(messages[-1])          # Pillar 2
    score = evaluator.score(messages, history, match)     # AutoConduck logic
    if score.confidence_band == "ambiguous":
        decision = tiebreaker_llm_call(messages[-1])       # cheap forced-choice
    else:
        decision = score.path                              # FAST or SLOW
    return decision
```

### 6.2 `semantic_router.py` — Route Classification (Pillar 2 wrapper)

- Defines two top-level routes, `fast_path` and `slow_path`, each seeded with curated example utterances (the direct successor to the old regex list — e.g. `fast_path` seeded with phrasings like "fix this typo," "rename this function," "where is X defined"; `slow_path` seeded with "refactor the application," "review the backend," "build a feature").
- Returns a route label **and** a similarity/confidence score — the confidence score is what feeds the ambiguous-zone decision, not just the label.
- Route examples require ongoing curation as real usage data comes in via `/stats` — this is an explicit maintenance cost, not a "set once" system.

### 6.3 `evaluator.py` — Complexity Scoring, Boosts & Hysteresis (AutoConduck's own logic)

This layer sits *after* Semantic Router and is what actually differentiates AutoConduck from a plain semantic router deployment:

- **Delta scoring:** combines Semantic Router's confidence with a scalar complexity estimate `T_i ∈ [0.0, 1.0]`, evaluated on `messages[-1]` only (prevents historical bloat inflating a single turn's score).
- **Stack-trace boost:** `+0.25` bounded boost if the latest message contains stack traces, compiler errors, or unhandled exceptions.
- **Hysteresis clamp:** if turn `t-1` escalated to a Reasoning-tier model (`T ≥ 0.80`), turn `t` is clamped to `T ≤ 0.50` unless a new stack trace is detected — prevents tier-thrashing across a conversation.
- **Ambiguous zone:** if Semantic Router's confidence score falls below a defined threshold (tunable, starting point `0.55–0.70` similarity), the decision is handed to the cheap LLM tiebreaker rather than trusted directly.

### 6.4 `pricing.py` — Cost Matrix (thin wrapper over LiteLLM)

- Reads live model costs directly from `litellm.model_cost`, with local JSON fallback.
- **EMA token correction:** estimates tokens via `tiktoken`, corrects predictions using an Exponential Moving Average (`α = 0.1`) against actual usage LiteLLM returns per completion.
- **Logarithmic scaling:** `Scaled Cost (C_m') = ln(1 + Cost_m)`, normalized across candidate models.
- **Degraded routing failover:** if a model's error rate exceeds 20% over a trailing 5-minute window, bypass it for the next-cheapest candidate. (LiteLLM's own fallback/retry config can handle much of this natively — evaluate before writing custom failover logic.)

### 6.5 `providers.py` — Custom / Gateway Provider Support

Not every model source is a "named provider" in LiteLLM's registry (e.g. DevPass by LLM Gateway, or any future flat-rate/subscription gateway). This does **not** require writing a custom LiteLLM provider plugin. LiteLLM already has a generic mechanism for exactly this: any endpoint that implements the OpenAI chat-completions API can be called by prefixing the model string with `openai/` and supplying an `api_base` + `api_key`:

```yaml
model_list:
  - model_name: devpass-claude-fable-5
    litellm_params:
      model: openai/claude-fable-5          # openai/ prefix = "call this as an OpenAI-compatible endpoint"
      api_base: https://api.llmgateway.io/v1
      api_key: os.environ/DEVPASS_API_KEY
```

This means AutoConduck's job is **not** to build provider-specific integrations for every gateway — it's to make this existing LiteLLM mechanism trivially configurable from the TUI, so the user never has to hand-write YAML. `providers.py` implements:

- **A "Custom Endpoint" option** in the onboarding Model Source screen (Section 9), alongside named presets (Anthropic, OpenAI, Google). Selecting it prompts for: display name, base URL, API key/env var, and optionally a static model list.
- **Model auto-discovery:** most OpenAI-compatible gateways (DevPass included) expose `GET /v1/models`. On adding a custom endpoint, `providers.py` calls that route to auto-populate the model picker rather than requiring the user to type exact model IDs by hand.
- **Generated LiteLLM config entries:** each selected model from a custom endpoint is written into the `model_list` using the `openai/<model>` + `api_base` pattern above — this is pure config generation, not new routing code.
- **A small built-in preset for DevPass specifically** (base URL, known env var names, doc link) so it shows up as a one-click option rather than requiring the user to go through the fully generic "Custom Endpoint" flow — worth doing given how common gateway subscriptions like this are, and cheap to maintain since it's just a config stub, not integration code.

**Cost-tracking implication (`pricing.py`):** DevPass and similar flat-rate gateways bill a fixed subscription, not metered per-token cost — so `litellm.model_cost` won't have accurate per-token pricing for models routed through it out of the box. `pricing.py`'s existing local JSON fallback (Section 6.4) is exactly the mechanism to handle this: define an entry that reflects effective cost (e.g., subscription price ÷ included allowance, or a flat $0 with a separate "quota remaining" indicator) rather than pretending it's metered at provider rates. The dashboard's cost-saved figure should visually distinguish metered spend from subscription-allowance spend — showing "$12.40 saved" when the underlying model was actually "free" against an already-paid subscription is misleading otherwise.

### 6.6 `orchestrator/` — LangGraph DAG (replaces old `orchestrator.py`)

- **Planner node:** requests a structured `TaskPlan` from a fast-tier model. Schema unchanged from v1:

  ```python
  class SubTask(BaseModel):
      id: str
      goal: str                    # single imperative sentence
      scope: list[str]             # resolved file paths — never vague descriptions
      output_contract: str         # required shape of the answer
      constraints: list[str]       # explicit "do NOT" list
      depends_on: list[str] = []
  ```

  Few-shot the planner with 1–2 worked examples (e.g. a sample "refactor auth flow" request mapped to a realistic DAG) — planners default to vague scope fields without this.

- **Subagent pool node:** uses LangGraph's `Send` API for the parallel map-reduce fan-out — this replaces hand-rolled `asyncio.gather` + `Semaphore(4)`, and gets checkpointing for free. Each subagent prompt is built from a fixed template:

  ```
  ROLE: You are a read-only file analyst. You do not propose fixes or write code.
  TASK: {goal}
  FILES IN SCOPE (only these): {scope}
  REQUIRED OUTPUT FORMAT: {output_contract}
  DO NOT: {constraints}
  CONTEXT FROM SIBLING TASKS: {upstream_summaries}
  ```

  File content is injected directly into the prompt where feasible, rather than asking the subagent to fetch files itself — keeps the step deterministic.

- **Compactor node:** merges structured (schema-constrained, not free-text) subagent outputs into a single prioritized, deduplicated summary under 1k tokens, preserving file:line references.

- **Executor node:** receives the compacted contract and produces the final implementation/response.

- **Reliability fallback:** if `TaskPlan` schema validation fails, retry once. Second failure abandons Slow-Path and falls back to Fast-Path on the original request — LangGraph's conditional edges handle this branching natively.

### 6.7 LiteLLM Proxy — Streaming & Endpoints (Pillar 1)

- `POST /v1/chat/completions` — intercepts pseudo-models, invokes `dispatcher.py` via pre-call hook, forwards upstream, streams SSE back.
- `GET /v1/models` — returns `autoconduck`, `autoconduck-budget`, `autoconduck-expensive`.
- Disconnect cancellation, retries, and cost logging are native LiteLLM Proxy features — do not reimplement.
- `/stats` & `/healthz` — custom endpoints layered on top for routing-decision audit logs and cache hit ratios (this part stays AutoConduck's own, since LiteLLM doesn't know about your routing tiers).

> **Open verification item:** confirm LiteLLM Proxy's pre-call hook can hand off an entire request to a different execution path (LangGraph) rather than only modifying/selecting the target model for a single completion call. If the hook model doesn't support that cleanly, a minimal dispatcher may need to sit *in front of* both LiteLLM and LangGraph rather than inside a LiteLLM hook.

---

## 7. Pseudo-Model Threshold Adjustments

Since Semantic Router returns a confidence score rather than a raw continuous scalar, the pseudo-model multipliers now apply to the **confidence threshold required to accept the direct route** (i.e., how easily a request escalates to Slow-Path or a higher model tier), not to `T_i` directly:

| Pseudo-Model Alias | Effect | Routing Behavior |
|---|---|---|
| `autoconduck-budget` | Raises the confidence bar needed to escalate | Stays on Fast-Path / cheap models unless Semantic Router + evaluator are both highly confident escalation is warranted |
| `autoconduck` (default) | Uses evaluator's thresholds as-is | Balanced baseline |
| `autoconduck-expensive` | Lowers the confidence bar needed to escalate | Escalates to Slow-Path / top-tier models even at moderate confidence |

This mapping needs empirical tuning once real telemetry exists — the exact multiplier values are a starting point, not a final answer.

---

## 8. File Structure

```
autoconduck/
├── main.py                     # CLI entrypoint & subcommands
├── dispatcher.py                # Thin entry point (Section 6.1) — AutoConduck's own logic
├── semantic_router.py           # Pillar 2 wrapper: route definitions, example curation
├── evaluator.py                 # Complexity scoring, boosts, hysteresis — AutoConduck's own
├── pricing.py                   # Thin wrapper over litellm.model_cost + EMA corrector
├── providers.py                 # Custom/gateway endpoint config generation, model discovery
├── orchestrator/
│   ├── graph.py                 # LangGraph graph definition (Pillar 3)
│   ├── planner.py                # TaskPlan generation + few-shot examples
│   ├── subagents.py              # Send-based fan-out, prompt templates
│   └── compactor.py              # Structured merge logic
├── config.py                    # YAML/env/flag config loader
├── tui/                          # See Section 9
│   ├── app.py
│   ├── onboarding.py
│   ├── dashboard.py
│   └── keymap.py
├── model_presets.py             # Account-level provider model discovery
├── agents/                       # Agent auto-patching registry
│   ├── base.py
│   ├── claude_code.py
│   ├── opencode.py
│   ├── aider.py
│   ├── continue_dev.py
│   ├── kilocode.py
│   ├── cursor.py
│   └── generic_openai.py
└── requirements.txt

npm-packaging/
├── autoconduck/
│   ├── package.json
│   └── bin/autoconduck.js
├── autoconduck-darwin-arm64/
├── autoconduck-darwin-x64/
├── autoconduck-linux-x64/
├── autoconduck-linux-arm64/
├── autoconduck-win32-x64/
└── build.py
```

---

## 9. TUI Specification — Keyboard-Only, Minimalist

### 9.1 Design principles

Looking at what already works well in this space (OpenCode's TUI and Claude Code's CLI are the two closest reference points), the strongest shared traits are worth deliberately carrying over:

- **No mouse dependency, ever.** Every action reachable by a single keypress or a short chord. This isn't just an aesthetic choice — it's what makes a TUI feel fast rather than like a GUI trapped in a terminal.
- **Restraint over decoration.** Box-drawing characters are used structurally (to separate panels), not decoratively. No gradients, no ASCII-art logos beyond a small wordmark on the splash screen, minimal color — a small, consistent palette (2–3 accent colors max: one for cost/savings, one for warnings, one for active/selected state) rather than a rainbow of syntax-highlighting-style color.
- **Information density without clutter.** Claude Code's CLI favors a mostly single-column, conversational flow with sparse chrome; OpenCode leans more panel-based (sidebar + main view) because it's managing more simultaneous state (multiple sessions, file trees). AutoConduck's dashboard is closer to OpenCode's problem shape (multiple simultaneous routing decisions, live stats) but should resist adding panels that aren't earning their space — every panel should answer a question the user actually has in the first five seconds of looking at the screen.
- **Progressive disclosure.** Onboarding shows one decision at a time, not a giant settings form. The live dashboard defaults to a compact view; detail (full routing rationale for a specific request) is a drill-down, not shown by default.
- **Status is ambient, not modal.** Cost savings, active routing tier, and connection health should be visible in a persistent status line, not requiring a keypress to check — the equivalent of a tmux status bar, not a popup.

### 9.2 Screens

**Splash / Onboarding (first run)**

```
┌─ AutoConduck ────────────────────────────────────────────────┐
│                                                                │
│  Detected agents:                                             │
│  › Claude Code        ~/.claude/settings.json                 │
│    OpenCode           ~/.config/opencode/config.json          │
│    Aider              not found                               │
│                                                                │
│  [enter] toggle   [j/k] move   [a] select all   [→] continue  │
└────────────────────────────────────────────────────────────────┘
```

- `j`/`k` (or arrow keys) to move selection — vim-style navigation is the expected default for this audience (both OpenCode and Claude Code CLI assume vim-literate users).
- `space`/`enter` to toggle a checkbox-style selection.
- `→` or `tab` to advance to the next onboarding step (model source selection), `←` to go back — no separate "back button" widget, just a keybind, shown in the persistent footer.
- Every screen's available keys are listed in a single-line footer, never a separate help screen the user has to remember to open.

**Model Source Selection (step 2)**

```
┌─ Model Sources ──────────────────────────────────────────────┐
│                                                                │
│  › Anthropic (API key detected: sk-ant-***…4f2)               │
│    OpenAI    (not configured)                                 │
│    Google    (not configured)                                 │
│    DevPass (LLM Gateway)   (not configured)                   │
│    Custom endpoint…                                            │
│                                                                │
│  Selected models for routing pool:                            │
│    ✓ claude-fable-5           $$   reasoning                  │
│    ✓ claude-haiku-4-5         $    fast                       │
│      gpt-5-mini                $    fast                      │
│                                                                │
│  [enter] toggle   [j/k] move   [→] continue   [q] quit         │
└────────────────────────────────────────────────────────────────┘
```

- Pricing tier shown as a compact `$`/`$$`/`$$$` glyph rather than exact numbers at this stage — reduces visual noise; exact live pricing is available on drill-down (`d`) or in the dashboard later.
- Named gateway presets (like DevPass) and the fully generic **Custom endpoint…** option both live in this same list — selecting either drops into a short sub-flow (base URL confirmation, API key entry, then live model auto-discovery via `GET /v1/models` where supported) rather than a separate settings page. Models pulled from a flat-rate gateway are tagged distinctly (e.g. a `~` glyph instead of `$`/`$$`) in the pool list, signaling "subscription allowance" rather than "metered cost" — this distinction carries through to the dashboard's cost-saved figure (Section 6.5).

**Live Dashboard (default view after setup)**

```
┌─ AutoConduck ── proxy: ● running :4141 ── saved: $12.40 today ┐
│                                                                 │
│  recent routing decisions                                      │
│  12:04:01  fast   claude-haiku-4-5     "fix typo in..."   0.9  │
│  12:03:44  slow   claude-fable-5       "review backend"   3.2s │
│  12:03:40  fast   gpt-5-mini            "rename function"  0.4  │
│                                                                 │
│  active agents: claude-code · opencode                         │
│                                                                 │
│  [d] drill into selected  [/] filter  [p] pause routing         │
│  [e] edit models          [?] all keys  [q] quit                │
└──────────────────────────────────────────────────────────────────┘
```

- Persistent header status line (proxy state, cost saved) never scrolls away — this is the "ambient status" principle; it's the single most-referenced piece of information so it should never require navigation to see.
- The routing log is the main content, deliberately close to a scrollback/log aesthetic (like `tmux` or `journalctl -f`) rather than a boxed table — this reads faster at a glance and scales naturally as more decisions stream in.
- `/` opens a lightweight filter input inline (not a separate modal/screen) — type to filter by agent, model, or path fast/slow, `esc` clears it.
- `d` drills into the currently selected log line to show full routing rationale (which tier, confidence score, why) — this is where detail *earns* its own screen, rather than cluttering the default view.

**Drill-down detail (on `d`)**

```
┌─ Routing Decision ── 12:03:44 ────────────────────────────────┐
│                                                                 │
│  prompt:      "review the backend and make improvements"       │
│  route:       slow_path      confidence: 0.81                  │
│  complexity:  T = 0.74  (stack-trace boost: no)                │
│  tier:        expensive (autoconduck-expensive)                │
│  subtasks:    4 (parallel)  →  compact  →  execute              │
│  model used:  claude-fable-5                                    │
│  cost:        $0.043                                            │
│                                                                  │
│  [esc] back   [c] copy full plan JSON   [q] quit                │
└─────────────────────────────────────────────────────────────────┘
```

### 9.3 Global keymap conventions

A single consistent keymap across all screens, following the vim/tmux conventions this audience already knows, rather than inventing new ones:

| Key | Action |
|---|---|
| `j` / `k` or `↓` / `↑` | move selection |
| `enter` / `space` | select / toggle |
| `esc` | back / close drill-down / cancel filter |
| `/` | inline filter (dashboard) |
| `?` | show full keybind reference (overlay, dismiss with `esc`) |
| `q` | quit current screen / quit app from top-level |
| `p` | pause/resume routing (dashboard only — lets the proxy pass everything to a single fixed model, useful for debugging) |
| `e` | jump to model edit screen from anywhere |

- `?` is the one exception to "no separate help screen" — it's an on-demand overlay, dismissed instantly, not a default-visible panel, so it doesn't cost screen real estate for users who already know the keymap.
- No key does something destructive without confirmation — e.g. `uninstall`-adjacent actions inside the TUI should require a typed confirmation (`y` + `enter`, not a single keypress) since they touch other tools' config files.

### 9.4 Why this shape over alternatives

- A pure single-column chat-style view (closer to Claude Code CLI) would undersell AutoConduck's actual value proposition — the routing decisions and cost savings *are* the product, so they need to be the persistent visual centerpiece, not buried behind a conversational transcript.
- A heavier multi-pane view (closer to some `k9s`/`lazygit`-style tools) was considered but rejected for the default screen — AutoConduck's dashboard doesn't yet have enough simultaneous independent state (no file tree, no multi-session management) to justify permanent panel splits. Panels are reserved for the drill-down, where there's genuinely more to show.
- Filtering inline (`/`) rather than as a separate modal keeps the log-like reading flow uninterrupted — closer to how `less`/`journalctl` handle search, which this audience already has muscle memory for.

---

## 10. Hard Requirements & Technical Constraints

- **Zero Host Dependencies:** `npm install -g autoconduck` results in a fully operational command without requiring Python/pip on the host.
- **Sub-5ms Fast-Path Decision Latency:** Semantic Router + evaluator overhead must stay under 5ms for the non-ambiguous case; validate this holds once LangGraph/Semantic Router are actually in the dependency tree, not just in isolation.
- **Zero Hard Failures:** Planner schema failures or LangGraph node errors must never surface as a client-facing API error — always degrade to Fast-Path.
- **Safety & Reversibility:** Every agent config modification creates a verbatim backup in `~/.autoconduck/backups/`. `autoconduck uninstall` fully restores original state.
- **Streaming Disconnect Policy:** Rely on LiteLLM Proxy's native disconnect handling rather than reimplementing cancellation.
- **No Placeholder Code:** Every module, route definition, scoring formula, and prompt template must be fully implemented and production-ready.
- **Packaging validation is now a first-order risk, not an afterthought:** because v2 adds LangGraph and Semantic Router to the dependency graph, prototype the compiled-binary build *early* (Section 3) to confirm bundle size and startup time stay acceptable before committing further engineering time to feature work.
