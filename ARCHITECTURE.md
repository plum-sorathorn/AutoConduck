# AutoConduck — Detailed Architecture Plan

> **Source:** `autoconduck-blueprint.md` (Unified Master Blueprint v1)  
> **Status:** Planning / Pre-implementation  
> **Date:** 2026-08-08  
> **Audience:** Implementers, reviewers, packaging maintainers

---

## Table of Contents

1. [Purpose & Scope](#1-purpose--scope)
2. [Design Principles & Hard Constraints](#2-design-principles--hard-constraints)
3. [High-Level System Context](#3-high-level-system-context)
4. [Technology Stack](#4-technology-stack)
5. [Repository & Runtime Layout](#5-repository--runtime-layout)
6. [Component Map — Responsibilities & Boundaries](#6-component-map--responsibilities--boundaries)
7. [Control & Data Flow Schema (Authoritative)](#7-control--data-flow-schema-authoritative)
8. [Module Specifications](#8-module-specifications)
9. [Inter-Module Interaction Matrix](#9-inter-module-interaction-matrix)
10. [Configuration & State Model](#10-configuration--state-model)
11. [Agent Adapter Subsystem](#11-agent-adapter-subsystem)
12. [Packaging & Distribution Architecture](#12-packaging--distribution-architecture)
13. [Cross-Cutting Concerns](#13-cross-cutting-concerns)
14. [Observability, Telemetry & UX Surfaces](#14-observability-telemetry--ux-surfaces)
15. [Error Handling & Degradation Strategy](#15-error-handling--degradation-strategy)
16. [Performance Budgets](#16-performance-budgets)
17. [Security & Reversibility](#17-security--reversibility)
18. [Implementation Phases & Dependency Graph](#18-implementation-phases--dependency-graph)
19. [Testing & Verification Plan](#19-testing--verification-plan)
20. [Risks & Open Decisions](#20-risks--open-decisions)

---

## 1. Purpose & Scope

AutoConduck is a **local, zero-overhead model router + task orchestrator** for OpenAI-compatible AI coding agents.

**Core promise to the user:**

- Install via `npm install -g autoconduck` — no Python/pip required.
- In every agent's model picker, three new pseudo-models appear: `autoconduck` (balanced), `autoconduck-budget` (cost-optimized), `autoconduck-expensive` (quality-optimized).
- Every turn is routed to the cheapest *capable* real model. Complex, multi-file tasks are automatically fanned out into an async **subagent DAG**; simple turns stay on a sub-5 ms **fast path**.

This document translates the blueprint into an **implementation-ready plan**: component boundaries, data contracts, sequencing, failure modes, and build/release mechanics. It is the single reference for building the system without placeholder code.

**Out of scope (v1):** multi-user server mode, cloud-hosted proxy, fine-tuned routing model, persistent vector memory.

---

## 2. Design Principles & Hard Constraints

| ID | Constraint | Enforcement |
|----|------------|-------------|
| C1 | **Zero host dependencies** — `npm install -g` yields a working `autoconduck` binary on win32/darwin/linux x64/arm64 | Binary bundles Python runtime + LiteLLM via PyInstaller/Nuitka; npm package only contains platform binary + JS shim |
| C2 | **Sub-5 ms fast-path overhead** (p50) | `gatekeeper.py` + `evaluator.py` stay synchronous, regex + arithmetic only; no LLM call on fast path; `tiktoken` cached |
| C3 | **Zero hard failures** to the client | Any orchestrator/DAG/schema error degrades to fast-path execution; proxy never surfaces internal 500 to agent — it forwards or falls back |
| C4 | **Safety & reversibility** | Every config patch: `~/.autoconduck/backups/<agent>/<timestamp>.bak` verbatim copy; patches are block-delimited (`# BEGIN AUTOCONDUCK` … `# END AUTOCONDUCK`); `uninstall` restores; re-running setup updates *only* owned blocks |
| C5 | **Instant disconnect cancellation** | Proxy monitors `request.is_disconnected()` and cancels upstream `httpx` / LiteLLM task immediately to prevent billed waste |
| C6 | **No placeholder code** | Every regex, hysteresis rule, EMA update, semaphore limit, and cost formula must be implemented and tested; CI enforces 100% module coverage on core paths |
| C7 | **OpenAI-compatible surface** | Agents see standard `/v1/chat/completions` (streaming + non-streaming) and `/v1/models`; request/response shapes are unmodified except `model` field mutation |

---

## 3. High-Level System Context

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Machine                              │
│                                                                  │
│  ┌──────────────┐   ┌──────────────────┐   ┌──────────────────┐  │
│  │ Coding Agent │   │  AutoConduck     │   │ Provider APIs    │  │
│  │ (Claude Code,│──▶│  Proxy (FastAPI) │──▶│ (Anthropic,      │  │
│  │  OpenCode,   │◀──│  :11434 or       │◀──│  OpenAI, Google, │  │
│  │  Aider, etc) │   │  :4000           │   │  local LLMs)     │  │
│  └──────────────┘   └──────────────────┘   └──────────────────┘  │
│                         │                                        │
│              ┌──────────┼──────────┐                              │
│              ▼          ▼          ▼                              │
│         TUI (textual)  Config   Backups (~/.autoconduck/)        │
└─────────────────────────────────────────────────────────────────┘
```

**Three deployment postures:**

1. **Interactive (default):** `autoconduck` → TUI onboarding → launches proxy + live dashboard.
2. **Headless:** `autoconduck start --headless` → background proxy daemon (systemd/launchd/CI).
3. **Reconfiguration:** `autoconduck edit` / `autoconduck uninstall`.

Agent → Proxy communication is **always localhost HTTP**. No external network exposure.

---

## 4. Technology Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Language | **Python 3.11+** | Required by LiteLLM + textual; performance vs. 3.10 |
| Web | **FastAPI + uvicorn** | Streaming SSE, async, `request.is_disconnected()` |
| HTTP | **httpx (async)** | LiteLLM-compatible, cancellation-aware |
| LLM Abstraction | **LiteLLM (async core only)** | Unified provider registry + `model_cost` map |
| Tokenization | **tiktoken** | Fast, offline, cached encodings; EMA correction layer on top |
| Validation | **pydantic v2** | `TaskPlan` DAG schema, config schema, request validation |
| TUI | **textual** | Keyboard-only, cross-platform, rich live dashboard |
| Packaging | **PyInstaller or Nuitka** | Single-file binary per OS/arch |
| Distribution | **npm `optionalDependencies`** | Familiar install; platform binary resolution via `bin/autoconduck.js` |
| Telemetry | In-process ring buffer + `/stats` JSON | No external DB; persistence optional (SQLite or JSONL log) |

**Stripping for binary size:** LiteLLM provider modules not selected by user at build time are excluded (tree-shake via PyInstaller hooks). Target 40–70 MB.

---

## 5. Repository & Runtime Layout

### 5.1 Source Layout (implements §7 of blueprint)

```
autoconduck/                        # Python package — compiled to binary
├── __init__.py
├── main.py                         # CLI entrypoint (argparse/click)
│   # subcommands: (no-args) | start --headless | edit | uninstall | --help | --version
├── proxy.py                        # FastAPI app: /v1/*, /stats, /healthz, SSE, disconnect
├── gatekeeper.py                   # 3-tier classifier (sync, <3ms budget)
├── evaluator.py                    # Delta-token scorer + stack-trace boost + hysteresis
├── pricing.py                      # EMA estimator + log cost matrix + failover registry
├── orchestrator.py                 # Pydantic DAG planner + semaphore pool + compactor
├── config.py                       # Loader: YAML + env + flags, schema, validation, backups path
├── tui.py                          # Textual app: onboarding screens + live dashboard
├── model_presets.py                # Provider discovery + pricing ingestion
├── state.py         (new)          # Runtime state: EMA params, error-rate windows, cache
├── cache.py         (new)          # Disk cache for identical prompts (optional)
├── telemetry.py     (new)          # Ring buffer, audit log, /stats aggregation
└── agents/
    ├── __init__.py
    ├── base.py                     # BaseAdapter ABC
    ├── claude_code.py
    ├── opencode.py
    ├── aider.py
    ├── continue_dev.py
    ├── kilocode.py
    ├── cursor.py
    └── generic_openai.py

npm-packaging/
├── autoconduck/                    # npm wrapper 'autoconduck'
│   ├── package.json
│   └── bin/autoconduck.js          # platform resolver → spawns binary
├── autoconduck-darwin-arm64/
│   ├── package.json
│   └── bin/autoconduck             # actual binary (PyInstaller output)
├── autoconduck-darwin-x64/  (idem)
├── autoconduck-linux-x64/   (idem)
├── autoconduck-linux-arm64/ (idem)
├── autoconduck-win32-x64/   (idem)
└── build.py                        # Cross-compile matrix, version stamping, shasum

~/.autoconduck/                     # Runtime home (user data, not in repo)
├── config.yaml
├── backups/<agent>/<timestamp>.bak
├── cache/                          # optional disk cache
└── logs/routing.jsonl
```

### 5.2 Runtime Processes

```
autoconduck (binary)
├─ main.py  → parse argv
│   ├─ no-args & no config      → tui.py (onboarding) → writes config.yaml → spawns proxy
│   ├─ no-args & has config     → tui.py (dashboard)  → spawns proxy
│   ├─ start --headless         → proxy.py (uvicorn)  → background
│   ├─ edit                     → tui.py (model selection only)
│   └─ uninstall                → agents/*.revert() → remove config.yaml → exit
└─ proxy.py (uvicorn on 127.0.0.1:<port>)
    ├─ gatekeeper → evaluator → pricing  (fast path)
    └─ gatekeeper → orchestrator → pricing (slow path)
```

Port: default `11434` (configurable); `config.yaml` stores chosen port. TUI offers port-conflict detection.

---

## 6. Component Map — Responsibilities & Boundaries

### 6.1 `config.py` — Configuration Loader

| Concern | Detail |
|---------|--------|
| Inputs | `~/.autoconduck/config.yaml`, env (`AUTOCONDUCK_*`), CLI flags (flags > env > file) |
| Schema | `pydantic` model: `port`, `pseudo_models`, `models: list[ModelEntry]`, `pricing_source`, `cache_enabled`, `log_level`, `backups_retention` |
| `ModelEntry` | `id`, `provider`, `api_key_ref` (env var name, never raw key in YAML), `tier` (budget/balanced/expensive/reasoning), `price_in/out` (from registry), `enabled` |
| Responsibilities | Validate, expand `model_cost` fallback, resolve port, manage `~/.autoconduck` creation, expose `get_config()` singleton (cached, reload on `edit`) |
| No-go | No LLM calls, no filesystem scan — pure config |

### 6.2 `model_presets.py` — Provider & Pricing Discovery

| Concern | Detail |
|---------|--------|
| Inputs | `litellm.model_cost` dict, `litellm.get_supported_openai_params` style registry, local `pricing_fallback.json` |
| Interaction | Called during TUI onboarding and on `edit`; optionally background refresh every 24h |
| Output | Normalized `list[ModelEntry]` with `price_in/out` per 1K tokens, `context_window`, `supports_streaming` |
| Account presets | "Anthropic account" → claude models; "OpenAI account" → gpt-*; "Google account" → gemini-*; "Custom API" → user pastes base_url + key + model list |
| Pricing fallback | If `litellm.model_cost` missing entry, use bundled `pricing_fallback.json` (checked in) |

### 6.3 `gatekeeper.py` — 3-Tier Dual-Path Classifier (<3 ms)

**Pure synchronous function:**

```python
def classify(request: ChatRequest, history: list[Message]) -> Decision:
    # returns Decision(path="fast"|"slow"|"ambiguous", reason=str, tier_hint=..., elapsed_ms=float)
```

- **Tier 1 — Fast-Path Regex Override:** `len(prompt) < 120 and REGEX_FAST.match(prompt)` where `REGEX_FAST = re.compile(r"^(fix|format|typo|rename|docstring|check syntax|where is|grep)\b", re.I)` → `fast`
- **Tier 2 — Slow-Path Override:** `len(attachments) > 3 or any(kw in prompt.lower() for kw in SLOW_KEYWORDS)` where `SLOW_KEYWORDS = ["refactor application","build feature","architecture","backtesting","migrate","rewrite entire"]` → `slow`
- **Tier 3 — Ambiguous Zone:** else compute `T_i = evaluator.score(messages[-1])`; if `T_i < 0.40` → fast, `T_i > 0.55` → slow, else `0.40 ≤ T_i ≤ 0.55` → **single cheap forced-choice LLM call** (`FAST`/`SLOW` + reasoning string, 2-token output via configured cheapest model, 800 ms timeout, fallback to `fast` on timeout/error).

No state, no async, no pricing.

### 6.4 `evaluator.py` — Delta Token Complexity Scorer

```python
def score(last_message: Message, prev_state: TurnState) -> float:
    # returns T_i in [0.0, 1.0]
```

- **Delta-only:** scores `messages[-1]` only; never cumulative history length.
- **Signals (weighted sum → sigmoid):** token count (tiktoken `cl100k_base`), code-block ratio, file-reference count, imperative complexity keywords, question depth, stack-trace presence.
- **Stack Trace Boost:** if `STACK_RE = re.compile(r"(Traceback|at\s+\w+\.\w+\(|UnhandledPromiseRejection|Error:|Exception:)", re.I)` matches → `T_i = min(1.0, T_i + 0.25)`
- **Hysteresis Cooldown:** if `prev_state.used_reasoning_tier == True` (prev `T >= 0.80`) then `T_i = min(T_i, 0.50)` unless new stack trace in current turn. `TurnState` kept in `state.py` ring buffer keyed by session/conversation id (from request metadata or hash of system prompt; best-effort per-client).
- Returns float clamped `[0.0, 1.0]`.

### 6.5 `pricing.py` — Logarithmic Cost Matrix & EMA Corrector

```python
class PricingEngine:
    def estimate_tokens(self, messages: list[Message], intent: str) -> tuple[int,int]: ...
    def scaled_costs(self, t_in: int, t_out: int, models: list[ModelEntry]) -> list[ScoredModel]: ...
    def select(self, T_i_prime: float, models: list[ModelEntry], t_in:int,t_out:int) -> ModelEntry: ...
    def record_usage(self, model_id: str, actual_in:int, actual_out:int): ...
    def is_degraded(self, model_id: str) -> bool: ...
```

- **Token est:** `T_in = tiktoken_len(messages)` (exact). `T_out = intent_table[intent]` (e.g., `fix: 400, refactor: 2500, architecture: 3500`, tunable; stored in config).
- **EMA correction:** `pred_out_ema = α*actual_out + (1-α)*pred_out` per intent category, `α=0.1`. Persisted in `state.py` (JSON, loaded at startup, flushed on `record_usage`).
- **Log cost:** `Cost_m = price_in * T_in + price_out * T_out` (prices per token, normalized from per-1K). `C_m' = ln(1 + Cost_m)`.
- **Tier selection:** map `T_i'` to tier buckets: `T' < 0.33 → budget`, `0.33–0.75 → balanced`, `>0.75 → reasoning/expensive`. Within tier, pick cheapest `C_m'` not degraded. If all degraded, pick cheapest overall.
- **Degraded failover:** sliding 5-min window per model: `error_rate = errors / total`; if `>0.20` and `total >= 5` → bypass for next selection; log to telemetry.
- **Pseudo-model bias:** `T_i' = transform(T_i, pseudo_model)` applied *before* `select()` (see §8.8).

### 6.6 `orchestrator.py` — Parallel Subagent DAG Engine

```python
class Orchestrator:
    async def plan_and_execute(self, request: ChatRequest, t_in:int,t_out:int) -> OrchestratorResult: ...

# Pydantic schemas
class TaskPlan(BaseModel):
    tasks: list[SubTask]  # 2-6 items
class SubTask(BaseModel):
    id: str; goal: str; files: list[str]; depends_on: list[str]; output_contract: str

class OrchestratorResult(BaseModel):
    compacted_context: str  # <1k tokens
    plan: TaskPlan | None
    degraded_to_fast: bool
    reason: str
```

- **Planning LLM call:** uses **fast-tier** model (cheapest balanced) with `response_format=json_schema` for `TaskPlan`. System prompt instructs DAG with `depends_on` and file isolation.
- **Retry-Once → Fallback:** on `ValidationError` or empty `tasks`, retry exactly once with repair prompt. Second failure → `degraded_to_fast=True`, return `None` to caller; proxy routes original request via fast path.
- **Worker pool:** `asyncio.gather(*workers, return_exceptions=True)` capped by `asyncio.Semaphore(4)`; each worker is an async LiteLLM call with isolated `messages` slice + `files` context + 30 s timeout. Workers do NOT call orchestrator recursively.
- **Compaction:** summarize worker outputs into `<1k token` structured contract (bullet list + file diffs summary) via cheap model or deterministic template if worker count ≤2. This compacted context is **prepended** to the original request when finally calling the execution model (selected by pricing).
- **No hard failure:** exceptions per worker are captured, logged, and excluded from compaction; at least 1 worker success → proceed; 0 successes → degrade to fast path.

### 6.7 `proxy.py` — Streaming Reverse Proxy & Endpoints

FastAPI app. Single responsibility: **intercept, route, mutate, stream, cancel, telemetry**.

**Endpoints:**

| Method | Path | Behavior |
|--------|------|----------|
| `POST` | `/v1/chat/completions` | Core routing (see §7) |
| `GET` | `/v1/models` | Return 3 pseudo-models (OpenAI shape) for GUI discovery |
| `GET` | `/stats` | Routing audit log, latency histogram, cost savings, cache hit ratio, degraded models |
| `GET` | `/healthz` | `{status:"ok", version, uptime, port}` |
| `GET` | `/` | Redirect to `/stats` or TUI dashboard link |

**`POST /v1/chat/completions` pipeline (async):**

1. Validate JSON → `ChatRequest` (pydantic). Reject if missing `model` or `messages` with OpenAI-compatible error shape.
2. If `model not in {"autoconduck","autoconduck-budget","autoconduck-expensive"}` → **passthrough** (forward untouched via LiteLLM; still emits telemetry as `passthrough`).
3. Else:
   - Disk cache check (key = hash(model+messages[-1]+pseudo_model)) — if hit and `cache_enabled`, return cached SSE replay (optional, off by default).
   - `request.is_disconnected()` poll task starts (cancels upstream on disconnect).
   - `gatekeeper.classify()` → path.
   - If `ambiguous` → one cheap forced-choice LLM call (see §6.3).
   - If `slow` → `orchestrator.plan_and_execute()` (async, up to ~4 s including worker parallelism; streaming to client is delayed until compaction done — acceptable tradeoff; alternatively emit progress SSE comments if agent tolerates).
   - Else `fast` → `evaluator.score()` → `T_i'` via pseudo-model transform → `pricing.select()`.
   - Mutate payload: `model = selected_real_model_id`, inject `compacted_context` as system message if orchestrator succeeded.
   - Forward via `litellm.acompletion(..., stream=...)` or direct `httpx` if provider not in LiteLLM.
   - Stream SSE chunks back (`StreamingResponse` with `media_type: text/event-stream`), forwarding verbatim; record `usage` from final chunk for EMA.
   - Telemetry: push `RoutingEvent` to ring buffer.
4. On `request.is_disconnected() == True` at any await point → cancel upstream task (`task.cancel()`), close `httpx` response, stop streaming, log `cancelled`.

**Streaming details:** must set `X-Accel-Buffering: no`, flush each `data: ...\n\n`, forward `[DONE]`. Support both `stream:true` and `stream:false`.

### 6.8 Pseudo-Model Threshold Transform (§6 of blueprint)

Applied in proxy *after* evaluator, *before* pricing:

```python
def transform(T: float, pseudo: str) -> float:
    if pseudo == "autoconduck-budget":    return T * 0.6
    if pseudo == "autoconduck-expensive": return min(1.0, T * 1.4 + 0.1)
    return T  # "autoconduck"
```

This single scalar biases tier selection without duplicating routing logic.

### 6.9 Supporting Modules

| Module | Role |
|--------|------|
| `state.py` | Holds `TurnState` history (per session hash), `EMAState` per intent, `ErrorWindow` per model (deque of timestamped success/fail, 5-min TTL), disk-cache index. Pruned lazily on access. Thread-safe via `asyncio.Lock`. |
| `cache.py` | Optional disk cache: `hash(model+last_message) → SSE transcript`. LRU with 100 MB cap; disabled by default (agents already cache). Useful for replay tests. |
| `telemetry.py` | Ring buffer (last 500 events), JSONL append to `~/.autoconduck/logs/routing.jsonl`, aggregation for `/stats`. Event shape: `{ts, pseudo_model, real_model, path, T_i, T_i_prime, cost_est, latency_ms, degraded, cache_hit, error}`. |
| `tui.py` | Two modes: **Onboarding** (agent detection → model ingestion → config patch → launch proxy + dashboard) and **Dashboard** (live table of routing events, latency sparkline, savings, hotkeys: `e` edit, `q` quit, `r` re-patch). Uses `textual` reactive + `proxy` telemetry polling (`/stats` via httpx or direct in-process queue when co-located). |
| `agents/base.py` | `class BaseAdapter(ABC)`: `detect()→bool`, `config_paths()→list[Path]`, `backup(path)`, `patch(config, models)`, `revert()`, `validate()` |

---

## 7. Control & Data Flow Schema (Authoritative)

> **Authority:** This section is the single source of truth for *who owns what*, *where decisions are made*, *what data moves*, and *which system/external calls fire*. Module specs (§6, §8), the interaction matrix (§9), and error tables (§15) **must not contradict** this schema. On conflict, prefer §7 and update the other sections.

### 7.0 Ownership Map (RACI-style)

| Artifact / Concern | **Owner** (sole writer) | **Readers** | **Must not touch** |
|--------------------|-------------------------|-------------|--------------------|
| CLI argv / process lifecycle | `main.py` | — | `proxy`, adapters |
| `~/.autoconduck/config.yaml` | `config.py` (load/save/validate); TUI writes only via `config` API | `proxy`, `pricing`, `tui`, adapters (read-only `Config`) | Direct YAML I/O outside `config.py` |
| `~/.autoconduck/state.json` (EMA, sessions, error windows) | `state.py` | `evaluator`, `pricing`, `proxy` | `gatekeeper` (stateless) |
| `TurnState` / hysteresis | `state.py` | `evaluator` (read), `proxy` (write after turn) | `orchestrator` |
| Routing path decision (`fast`/`slow`) | `gatekeeper.py` | `proxy` (acts on `Decision`) | `pricing`, adapters |
| Complexity score `T_i` | `evaluator.py` | `gatekeeper` (tier 3), `proxy` (fast path), transform | `orchestrator` planning |
| Pseudo transform `T_i → T_i'` | `proxy.py` (calls `pricing.transform` or local fn) | `pricing.select` | Gatekeeper (pre-transform) |
| Real model selection | `pricing.py` | `proxy`, `orchestrator` (final exec model) | `gatekeeper`, `evaluator` |
| `TaskPlan` DAG + worker pool + compaction | `orchestrator.py` | `proxy` (consumes `OrchestratorResult`) | Recursive orchestrator calls from workers |
| Request mutation (`model`, system prepend) | `proxy.py` only | Upstream LiteLLM/httpx | Adapters at request time |
| SSE stream / disconnect cancel | `proxy.py` | Agent client | Orchestrator (no direct client I/O) |
| Telemetry ring + JSONL | `telemetry.py` | TUI, `/stats` | Core path must not block on disk |
| Disk prompt cache | `cache.py` | `proxy` | Default off |
| Agent config files on disk | `agents/*` via `BaseAdapter` | TUI onboarding/uninstall | `proxy` (never patches agents) |
| Provider HTTP (LLM I/O) | LiteLLM / httpx **invoked only by** `proxy` (final) and `orchestrator` (plan + workers) + gatekeeper ambiguous call (via injectable client owned by proxy) | — | `evaluator`, `pricing.select`, `config` |
| npm binary spawn | `bin/autoconduck.js` | OS process | Python package |

**Hard ownership rules:**

1. **Only `proxy.py` speaks HTTP to the coding agent.** Orchestrator never streams to the client.
2. **Only `pricing.py` chooses a real model id** (including degraded failover). Proxy/orchestrator pass `T_i'` + estimates; they do not hardcode model ids except passthrough of non-pseudo models.
3. **Only `gatekeeper.py` emits path `Decision`.** Proxy never re-classifies after a decision except on orchestrator `degraded_to_fast` (forced re-entry to fast path on original request).
4. **Only adapters mutate third-party agent configs.** Uninstall/edit flows go `main` → `tui`/`uninstall` → `agents/*`.

---

### 7.1 Process & Control Topology

```
┌─ OS / npm ──────────────────────────────────────────────────────────────┐
│  autoconduck.js  ──spawn──▶  autoconduck binary (main.py)               │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
              [no-args]         [start --headless]   [edit|uninstall]
                    │                 │                 │
                    ▼                 │                 ▼
              tui.py (textual)        │           tui / agents.revert
                    │                 │
                    │  spawns / embeds│
                    └────────┬────────┘
                             ▼
                    proxy.py (uvicorn 127.0.0.1:port)
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
   gatekeeper.py        orchestrator.py      pricing.py
         │                   │                   │
         └──── evaluator.py ─┘                   │
                             │                   │
                             ▼                   ▼
                      LiteLLM / httpx  ◀─────────┘
                             │
                             ▼
                      Provider APIs (external)
```

**Control domains:**

| Domain | Controller | Blocks until | Cancels via |
|--------|------------|--------------|-------------|
| Process start/stop | `main.py` | TUI exit or signal | `SIGINT/SIGTERM` → uvicorn graceful |
| Per-request pipeline | `proxy` handler task | Response complete or disconnect | `request.is_disconnected` → `task.cancel()` |
| Slow-path DAG | `orchestrator.plan_and_execute` | Compaction or degrade | Parent cancel propagates to workers |
| Ambiguous classify LLM | `gatekeeper` (sync call site in proxy async) | ≤800 ms | Timeout → default FAST |

---

### 7.2 Master Request Pipeline (Control Flow)

Authoritative step order for `POST /v1/chat/completions`. Step IDs are stable for logs/tests (`RoutingEvent.step` / debug traces).

```
CF-00  ENTRY
         │  Owner: proxy
         │  In:  raw HTTP body
         │  Out: ChatRequest | 4xx OpenAI-shaped error
         ▼
CF-01  VALIDATE
         │  Owner: proxy (pydantic ChatRequest)
         │  Fail → return error; stop
         ▼
CF-02  MODEL GATE  ◀── DECISION D1
         │  Owner: proxy
         │  if model ∉ PSEUDO_MODELS → CF-P (passthrough)
         │  else → CF-03
         ▼
CF-03  CACHE LOOKUP  ◀── DECISION D2
         │  Owner: cache.py (called by proxy)
         │  hit & cache_enabled → CF-99 stream cached SSE; stop
         │  miss → CF-04
         ▼
CF-04  DISCONNECT MONITOR START
         │  Owner: proxy (background task, poll 50 ms)
         │  Side effect: can jump to CF-CANCEL from any await
         ▼
CF-05  CLASSIFY  ◀── DECISION D3 (path)
         │  Owner: gatekeeper.classify
         │  In:  ChatRequest, history/TurnState key
         │  Out: Decision{path, reason, tier_hint?, T_i?, elapsed_ms}
         │        path ∈ {fast, slow, ambiguous}
         │
         ├─ path=fast ──────────────────────────────▶ CF-20
         ├─ path=slow ──────────────────────────────▶ CF-30
         └─ path=ambiguous ──▶ CF-06
                                │
CF-06  FORCED-CHOICE LLM  ◀── DECISION D4
         │  Owner: proxy invokes cheap model (gatekeeper prompt contract)
         │  System call: LiteLLM/httpx, max_tokens=10, timeout 800 ms
         │  Parse FAST|SLOW; fail/timeout → FAST
         │  → CF-20 or CF-30
         ▼
════════ FAST PATH ════════════════════════════════
CF-20  SCORE
         │  Owner: evaluator.score
         │  In:  messages[-1], TurnState
         │  Out: T_i ∈ [0,1]  (clamp 0.5 on exception)
         ▼
CF-21  TRANSFORM  ◀── DECISION D5 (bias only; not path)
         │  Owner: proxy
         │  T_i' = transform(T_i, pseudo_model)
         ▼
CF-22  ESTIMATE TOKENS
         │  Owner: pricing.estimate_tokens
         │  Out: (t_in, t_out)  [t_in exact tiktoken; t_out intent+EMA]
         ▼
CF-23  SELECT MODEL  ◀── DECISION D6
         │  Owner: pricing.select(T_i', models, t_in, t_out)
         │  Out: ModelEntry (real id); never fails (cheapest fallback)
         ▼
CF-24  MUTATE PAYLOAD
         │  Owner: proxy
         │  model ← real id; optional system prepend if slow-path context
         ▼
CF-25  FORWARD STREAM  ── SYSTEM CALL S1
         │  Owner: proxy → litellm.acompletion | httpx
         │  Stream SSE to client
         ▼
CF-26  POST-TURN
         │  Owner: proxy → pricing.record_usage, state, telemetry[, cache]
         │  Out: RoutingEvent
         ▼
CF-99  EXIT (response complete | cancelled | cached)

════════ SLOW PATH ════════════════════════════════
CF-30  PLAN_AND_EXECUTE  ── SYSTEM CALLS S2,S3
         │  Owner: orchestrator
         │  S2: planning LLM (fast-tier, JSON schema TaskPlan)
         │      ValidationError/empty → retry once (S2b) → else DEGRADE
         │  S3: ≤4 workers (Semaphore), 30 s each, gather return_exceptions
         │  Compact successes → compacted_context (<1k tokens)
         │  0 successes or double plan fail → degraded_to_fast=True
         │
         ├─ degraded_to_fast ──▶ CF-20 on ORIGINAL request (no compaction)
         └─ success ───────────▶ CF-31
CF-31  DERIVE T_i' FOR EXEC
         │  Owner: proxy (+ evaluator/transform as on fast path;
         │         may use max worker complexity hint if present)
         ▼
CF-32  SELECT EXEC MODEL  ◀── D6 (same owner: pricing)
         ▼
CF-33  MUTATE + PREPEND compacted_context as system message
         ▼
CF-25 … CF-26 … CF-99  (same forward/post-turn)

════════ PASSTHROUGH ══════════════════════════════
CF-P   FORWARD UNTOUCHED  ── S1
         │  Owner: proxy; telemetry path=passthrough
         ▼
CF-99

════════ CANCEL ═══════════════════════════════════
CF-CANCEL  (from any await if disconnected)
         │  Owner: proxy
         │  upstream_task.cancel(); aclose httpx; log cancelled; stop billable work
```

---

### 7.3 Decision Points Catalog

| ID | Name | Owner | Inputs | Outputs | Failure default |
|----|------|-------|--------|---------|-----------------|
| **D1** | Pseudo vs passthrough | `proxy` | `request.model` | branch CF-P or CF-03 | N/A (deterministic) |
| **D2** | Cache hit | `cache` + `proxy` | hash(model+last+pseudo), `cache_enabled` | hit → CF-99 / miss → CF-04 | miss |
| **D3** | Path classify | `gatekeeper` | prompt, attachments, optional `T_i` | `fast` \| `slow` \| `ambiguous` | N/A (pure); tier3 uses evaluator |
| **D3.1** | Tier1 fast regex | `gatekeeper` | `len(prompt)<120` ∧ `REGEX_FAST` | force `fast` | — |
| **D3.2** | Tier2 slow keywords / files | `gatekeeper` | attachments>3 ∨ `SLOW_KEYWORDS` | force `slow` | — |
| **D3.3** | Tier3 score bands | `gatekeeper`+`evaluator` | `T_i` vs 0.40 / 0.55 | fast / slow / ambiguous | — |
| **D4** | Ambiguous LLM | `proxy`+gatekeeper contract | truncated last message | FAST or SLOW | **FAST** |
| **D5** | Pseudo transform | `proxy` | `T_i`, pseudo id | `T_i'` | identity if unknown pseudo |
| **D6** | Tier + cheapest model | `pricing` | `T_i'`, costs, degraded flags | `ModelEntry` | cheapest non-empty enabled |
| **D7** | Plan valid? | `orchestrator` | pydantic `TaskPlan` | accept / retry / degrade | degrade after 2 fails |
| **D8** | Worker keep? | `orchestrator` | exception/timeout per worker | include in compact or drop | drop worker |
| **D9** | All workers dead? | `orchestrator` | success count | proceed vs `degraded_to_fast` | degrade |
| **D10** | Model degraded? | `pricing` | 5-min error window | bypass model | if all bypassed, ignore flags |
| **D11** | Disconnect? | `proxy` | `request.is_disconnected` | continue vs CF-CANCEL | cancel wins |

**Band constants (D3.3):** `T_i < 0.40` → fast; `T_i > 0.55` → slow; else ambiguous → D4.

**Tier buckets (D6):** `T_i' < 0.33` → budget; `0.33–0.75` → balanced; `> 0.75` → reasoning/expensive; within tier min `C_m' = ln(1+Cost_m)` among non-degraded.

**Transform (D5):**

```
autoconduck-budget:    T' = T * 0.6
autoconduck:           T' = T
autoconduck-expensive: T' = min(1.0, T * 1.4 + 0.1)
```

---

### 7.4 Data Flow Schema (Artifacts in Motion)

#### 7.4.1 Artifact dictionary

| Artifact | Produced by | Consumed by | Lifetime | Notes |
|----------|-------------|-------------|----------|-------|
| `ChatRequest` | proxy validate | gatekeeper, orchestrator, pricing est., LiteLLM | request | OpenAI-compatible; `model` mutated only at CF-24/33 |
| `Decision` | gatekeeper | proxy | request | `{path, reason, tier_hint?, T_i?, elapsed_ms}` |
| `T_i` | evaluator | gatekeeper (tier3), proxy transform | request | delta-only on `messages[-1]` |
| `T_i'` | proxy transform | pricing.select | request | never persisted |
| `(t_in, t_out)` | pricing.estimate_tokens | pricing.select, telemetry | request | `t_out` EMA-corrected by intent |
| `ModelEntry` | pricing.select / config | proxy mutate, orchestrator workers | request | real provider model id |
| `TaskPlan` / `SubTask[]` | orchestrator planner LLM | worker spawner | request | 2–6 tasks; `depends_on`, `files`, `output_contract` |
| `worker_outputs[]` | orchestrator workers | compactor | request | failures excluded |
| `compacted_context` | orchestrator | proxy mutate (system prepend) | request | **&lt;1k tokens**; discarded after forward |
| `OrchestratorResult` | orchestrator | proxy | request | `{compacted_context, plan, degraded_to_fast, reason}` |
| `TurnState` | state (read); proxy (write) | evaluator hysteresis | session TTL 30m / LRU 200 | keyed by session hash or `x-session-id` |
| `EMAState` | pricing.record_usage → state | estimate_tokens | process + `state.json` | per intent, α=0.1 |
| `ErrorWindow` | pricing.record_usage | is_degraded / select | 5-min sliding | per `model_id` |
| `RoutingEvent` | proxy → telemetry | TUI, `/stats`, JSONL | ring 500 + disk | see shape below |
| `Config` | config.py | all modules at start | process (reload on edit) | keys only as env *names* |
| Cached SSE blob | cache.py | proxy CF-03 | disk LRU 100MB | optional |

#### 7.4.2 `RoutingEvent` (telemetry data contract)

```text
{
  ts, pseudo_model, real_model, path,           # path: fast|slow|ambiguous|passthrough|cancelled
  T_i, T_i_prime, cost_est, latency_ms,
  degraded, cache_hit, error, reason,           # reason from Decision / orchestrator
  steps_ms: { classify, score, plan, workers, select, ttfb }
}
```

#### 7.4.3 Data-flow diagram (per pseudo-model turn)

```
                    ┌──────────── config.yaml / state.json ────────────┐
                    ▼                                                  │
Agent ──POST──▶ proxy ──▶ cache? ──▶ gatekeeper ──▶ evaluator ──▶ T_i  │
                  │              │         │              ▲            │
                  │              │         │         TurnState ────────┤
                  │              │         ▼                           │
                  │              │    Decision(path)                   │
                  │              │         │                           │
                  │         ┌────┴────┐   │                           │
                  │         ▼         ▼   ▼                           │
                  │      [slow]    [fast/after D4]                    │
                  │         │         │                               │
                  │         ▼         │                               │
                  │   orchestrator    │                               │
                  │    │    │         │                               │
                  │   S2   S3         │                               │
                  │    │    │         │                               │
                  │    └──compact─────│                               │
                  │         │         │                               │
                  │         ▼         ▼                               │
                  │      T_i' = transform(T_i, pseudo)                │
                  │         │                                         │
                  │         ▼                                         │
                  │      pricing.select ──▶ ModelEntry                │
                  │         │                    ▲                    │
                  │         │              ErrorWindow/EMA ───────────┤
                  │         ▼                                         │
                  │   mutated ChatRequest (+ optional system context) │
                  │         │                                         │
                  │        S1 litellm/httpx ──▶ Provider              │
                  │         │                                         │
                  │         ▼                                         │
                  │   SSE ──▶ Agent                                   │
                  │         │                                         │
                  │         ▼                                         │
                  └── record_usage / TurnState / RoutingEvent ────────┘
```

---

### 7.5 System & External Calls Registry

| ID | Call | Invoker | Callee | Sync | Timeout | On failure |
|----|------|---------|--------|------|---------|------------|
| **S0** | (none) — pure CPU | gatekeeper tier1/2, evaluator, pricing.select | — | sync | N/A (budget &lt;5 ms total fast path) | clamp / cheapest |
| **S1** | Final completion (stream/non-stream) | `proxy` | LiteLLM `acompletion` or httpx | async | provider-defined; cancel on disconnect | surface provider error; `record_usage(error)` |
| **S2** | Plan `TaskPlan` JSON | `orchestrator` | LiteLLM (fast-tier model) | async | ~1.5 s soft | retry once (S2b) then degrade |
| **S2b** | Plan repair | `orchestrator` | LiteLLM | async | same | `degraded_to_fast` |
| **S3** | Worker subtask ×N (N≤4) | `orchestrator` | LiteLLM | async parallel | 30 s each | exclude worker; if 0 ok → degrade |
| **S4** | Compaction summarize (optional LLM) | `orchestrator` | LiteLLM cheap **or** deterministic template if N≤2 | async/sync | short | template fallback |
| **S5** | Ambiguous FAST/SLOW | `proxy` (gatekeeper contract) | LiteLLM cheapest | async w/ timeout | **800 ms** | default FAST |
| **S6** | Pricing registry refresh | `model_presets` | LiteLLM `model_cost` / network | async | onboarding/edit; 24h bg | `pricing_fallback.json` |
| **S7** | Agent detect/patch/revert | `tui` / `main uninstall` | filesystem via `agents/*` | sync | N/A | per-adapter error; continue others |
| **S8** | Telemetry JSONL append | `telemetry` | filesystem | sync debounced | N/A | drop event; never fail request |
| **S9** | state.json flush | `state` | filesystem | sync debounced (every 10 records / SIGTERM) | N/A | keep memory; next start defaults |
| **S10** | Disk cache read/write | `cache` | filesystem | sync | N/A | treat as miss |
| **S11** | `/stats` `/healthz` | external or TUI | `proxy` | async HTTP | N/A | — |
| **S12** | npm shim spawn | Node `autoconduck.js` | OS exec binary | sync | N/A | actionable missing-binary error |

**Call authorization matrix (who may invoke LLM I/O):**

| Module | May call LiteLLM/httpx? |
|--------|-------------------------|
| `proxy` | **Yes** — S1, S5 |
| `orchestrator` | **Yes** — S2, S2b, S3, S4 |
| `gatekeeper` | **No direct** — S5 owned by proxy using gatekeeper prompts |
| `evaluator`, `pricing` (select path), `config`, `agents` | **No** |

---

### 7.6 Sequence: Fast-Path (typical — ~p90 traffic)

```
Agent                          Proxy (FastAPI)                Gatekeeper  Evaluator  Pricing   LiteLLM/Provider
 │ POST /v1/chat/completions    │                               │          │         │         │
 ├─────────────────────────────▶│                               │          │         │         │
 │  {model:"autoconduck",       │  CF-01 parse & validate       │          │         │         │
 │   messages:[..., last]}      │  CF-03 cache check (miss)     │          │         │         │
 │                              │  CF-04 spawn disconnect mon.  │          │         │         │
 │                              │──────── CF-05 classify ──────▶│          │         │         │
 │                              │  D3: Tier1? Tier2? T_i?       │──score──▶│         │         │
 │                              │  decision: FAST (T=0.28)      │◀─T_i=0.28┘         │         │
 │                              │  CF-21 T'=transform(0.28)     │          │         │         │
 │                              │  CF-22/23 ────────────────────│──────────│──select▶│         │
 │                              │  T'=0.28 → tier=budget        │          │◀─model: gpt-4o-mini
 │                              │  CF-24 mutate {model:gpt-4o-mini}
 │                              │  CF-25 S1 forward stream ───────────────────────────────────▶│
 │                              │◀──────────────── SSE chunks ──────────────────────────────┤
 │◀──────────────── SSE ────────┤  CF-26 record usage → EMA     │          │         │         │
 │                              │         push RoutingEvent     │          │         │         │
```

**Latency budget:** parse 0.2 ms + gatekeeper 1–3 ms + pricing 0.3 ms + forward = **&lt;5 ms overhead** before first upstream byte. **No S2–S5 on pure fast path.**

---

### 7.7 Sequence: Slow-Path (complex task)

```
Agent                    Proxy               Gatekeeper  Orchestrator              Pricing   Provider
 │ POST ...               │                    │          │                        │         │
 ├───────────────────────▶│                    │          │                        │         │
 │                        │──CF-05 classify───▶│          │                        │         │
 │                        │  D3.2 Tier2 hit → SLOW        │                        │         │
 │                        │                    │          │                        │         │
 │                        │──────── CF-30 ────────────────▶│                        │         │
 │                        │                    │          │ S2 plan (TaskPlan)     │         │
 │                        │                    │          │ S2b retry once on fail │         │
 │                        │                    │          │ S3 ≤4 workers (sem,30s)│         │
 │                        │                    │          │ S4 compact (&lt;1k tok)   │         │
 │                        │◀──── OrchestratorResult ───────┤                        │         │
 │                        │  D7–D9 ok → compacted_context  │                        │         │
 │                        │  CF-31/32 T' → D6 ─────────────│──select───────────────▶│         │
 │                        │  CF-33 mutate + prepend system │                        │         │
 │                        │  CF-25 S1 ─────────────────────────────────────────────────────▶│
 │◀────── SSE (execution) ┤◀──────────────── SSE ────────────────────────────────────────┤
 │                        │  CF-26 post-turn               │                        │         │
```

**Key invariant:** slow-path planning failure (2 schema failures) or zero worker successes → immediate fast-path fallback (**CF-20…**) on **original request**; no error surfaces to agent (C3).

---

### 7.8 Sequence: Ambiguous Band + Degrade + Cancel

```
Ambiguous (0.40 ≤ T_i ≤ 0.55):
  proxy ──S5──▶ cheap LLM ──▶ FAST|SLOW ──▶ CF-20 or CF-30
  timeout/error ──▶ FAST (D4 default)

Orchestrator degrade:
  CF-30 ──▶ degraded_to_fast=True ──▶ CF-20 (original messages, no compact)
  RoutingEvent.path records slow_attempt + degraded flag

Disconnect:
  monitor ──D11──▶ CF-CANCEL ── cancel S1/S2/S3 tasks ──▶ log cancelled ──▶ no further SSE
```

---

### 7.9 Lifecycle Flows (Non-Request Control)

| Flow | Control steps | Data written | System calls |
|------|---------------|--------------|--------------|
| **First run (no config)** | main → tui onboarding → model_presets → agents.patch → config.save → start proxy → dashboard | `config.yaml`, backups, agent configs | S6, S7, bind port |
| **Returning interactive** | main → config.load → tui dashboard → embed/spawn proxy | state/telemetry only | S11 poll |
| **Headless start** | main → config.load → uvicorn proxy | state/telemetry | bind; optional S9 |
| **edit** | main → tui model screens → config.save (no agent re-patch unless user requests `r`) | `config.yaml` | S6 optional |
| **uninstall** | main → agents.revert → remove config (keep backups policy) | restore agent files; delete config | S7 |
| **SIGTERM** | main/uvicorn → flush state + telemetry → exit | `state.json`, log flush | S8, S9 |

---

### 7.10 Concurrency, Re-entrancy & Ordering Invariants

1. **Single-writer mutate:** only the request handler task mutates the outbound payload for that request.
2. **No recursive orchestration:** workers must not call `plan_and_execute`.
3. **Cancel beats billable work:** D11 aborts in-flight S1–S5; orchestrator workers receive cancellation.
4. **Fast path remains sync-CPU until S1:** CF-05/20/21/22/23 must not `await` except D4 (S5) when ambiguous.
5. **Degrade is re-entry, not error:** `degraded_to_fast` jumps to CF-20 without raising to the agent.
6. **Passthrough skips D3–D6:** non-pseudo models never enter gatekeeper/orchestrator.
7. **Transform once per final select:** D5 applied immediately before D6 for the execution model (including slow-path final); not applied to S2/S3/S5 internal calls (those use explicit tier/cheap picks).

---

### 7.11 Pseudo-Model Transform Application Point

```
T_i (evaluator) ──▶ transform(pseudo_model) ──▶ T_i' ──▶ pricing.select tier ──▶ real model
                                    │
                     budget: ×0.6  ─┤  (cheap bias)
                     balanced: ×1.0─┤
                     expensive: min(1.0, ×1.4+0.1) (quality bias)
```

Applied once per turn for **execution** model choice (CF-21 / CF-31), after path decision, before pricing. Orchestrator **planning/workers** use fixed tier policy (fast/cheap), not `T_i'`.

---

### 7.12 Cross-References

| Topic | Detail section |
|-------|----------------|
| Module APIs & regex | §6, §8, Appendix B |
| Caller→callee matrix | §9 (must match §7.2 / §7.5) |
| Config & state shapes | §10 |
| Adapter patch ownership | §11 |
| Degradation table | §15 (defaults aligned with D4, D7–D10) |
| Latency budgets | §16 |

---

## 8. Module Specifications

### 8.1 `main.py` — CLI Entrypoint

```
autoconduck [--version] [--help]
autoconduck                          # interactive (onboard or dashboard)
autoconduck start [--headless] [--port PORT]
autoconduck edit
autoconduit uninstall [--force]
```

- Uses `argparse` (stdlib, no extra dep in binary).
- `--headless` suppresses TUI, runs `uvicorn.run(proxy.app, host="127.0.0.1", port=...)` foreground; caller (systemd/launchd) manages daemonization.
- Signal handling: `SIGINT/SIGTERM` → graceful uvicorn shutdown.
- Reads `config.py` on startup; if missing and no-args → onboarding.
- `edit` reuses TUI model-selection screen but skips `agents/*.patch`.

### 8.2 `proxy.py` — Detailed Contract

**Request shape (OpenAI-compatible subset):**

```python
class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    tools: list | None = None
    tool_choice: str | None = None
    # plus extra passthrough fields: **kwargs
```

**Response:** verbatim upstream SSE or JSON; proxy does not re-chunk.

**`/v1/models` response:**

```json
{
  "object": "list",
  "data": [
    {"id":"autoconduck","object":"model","created":0,"owned_by":"autoconduck"},
    {"id":"autoconduck-budget","object":"model","created":0,"owned_by":"autoconduck"},
    {"id":"autoconduck-expensive","object":"model","created":0,"owned_by":"autoconduck"}
  ]
}
```

**`/stats` response:**

```json
{
  "uptime_seconds": 1234,
  "total_requests": 87,
  "fast_path_ratio": 0.78,
  "cache_hit_ratio": 0.05,
  "avg_overhead_ms": 2.1,
  "cost_savings_estimate": {"without_router": 1.23, "with_router": 0.41},
  "recent_events": [ ... last 50 RoutingEvent ... ],
  "degraded_models": ["claude-3.5-sonnet: 0.33 error rate (5m)"],
  "pricing_ema": {"fix": {"pred_out": 380, "alpha":0.1}}
}
```

### 8.3 `gatekeeper.py` — Internal Detail

```python
REGEX_FAST = re.compile(r"^(fix|format|typo|rename|docstring|check syntax|where is|grep)\b", re.I)
SLOW_KEYWORDS = ["refactor application","build feature","architecture","backtesting",
                 "migrate","rewrite entire","monorepo","codebase-wide"]
AMBIGUOUS_LOW, AMBIGUOUS_HIGH = 0.40, 0.55
```

**Ambiguous LLM call (only when `0.40 ≤ T_i ≤ 0.55`):**

- Model: cheapest enabled (from pricing registry).
- Prompt: `Decide FAST or SLOW for: "<last_message truncated 400 chars>"\nReply with one word: FAST or SLOW, then a one-line reason after " | ".`
- `max_tokens=10`, `temperature=0` Log reason string; on parse failure/timeout → default `FAST`.

**Testability:** gatekeeper accepts injected `score_fn` for unit tests (no LLM needed).

### 8.4 `evaluator.py` — Signals & Scoring

Pseudocode for `T_i` baseline before boost/hysteresis:

```
features = {
  token_len:    len(tiktoken.encode(last_message.content)) / 4000,  # normalized 0-1
  code_ratio:   code_chars / total_chars,
  file_refs:    count(r"[\w/.-]+\.(py|ts|js|go|rs|md)") / 5,
  imperative:   0.2 if any(["implement","create","analyze"] in msg) else 0,
  question_depth: 0.15 * count("?") capped 0.3
}
raw = weighted_sum(features, weights={token_len:0.35, code_ratio:0.25, file_refs:0.2, ...})
T = sigmoid((raw - 0.5) * 6)  # sharp around 0.5, range 0..1
```

Weights tuned empirically; unit tests pin thresholds for known prompts.

### 8.5 `pricing.py` — Degraded Window Implementation

```python
@dataclass
class ErrorWindow:
    events: deque[tuple[float, bool]]  # (timestamp, is_error)
    def error_rate(self, now: float) -> float: ...
    def prune(self, now: float): ...  # drop >5 min old
```

Pruning on each `record_usage` and `is_degraded` check. No background thread.

### 8.6 `orchestrator.py` — Planning Prompt & Contracts

System prompt skeleton for planner:

```
You are a task decomposer. Break the user request into 2-6 parallel subtasks.
Each subtask must have isolated file context and an output_contract (what it must return).
Return JSON matching TaskPlan schema. Prefer file-disjoint subtasks.
```

Worker prompt: `SubTask.goal + file context + output_contract`.

### 8.7 `tui.py` — Screens & Navigation

| Screen | Content | Keys |
|--------|---------|------|
| Welcome | Logo, detect summary | `Enter` continue |
| Agent Detection | Table: agent, found?, config path, patch status | `Space` toggle, `a` select all |
| Model Ingestion | Provider cards, API key inputs (masked), live pricing fetch | `Tab` navigate, `Enter` fetch pricing |
| Model Selection | Table with price/quality tier, checkboxes | `Space` toggle, `Enter` confirm |
| Patching | Progress log, backup paths | auto |
| Dashboard | Live routing table, latency, savings, degraded warnings | `e` edit, `q` quit, `p` pause, `c` clear |

TUI runs proxy in background thread when in dashboard mode; communicates via `telemetry.py` queue (not HTTP polling when co-located for zero overhead).

### 8.8 `config.py` — Schema Example

```yaml
version: 1
port: 11434
cache_enabled: false
log_level: info
models:
  - id: gpt-4o-mini
    provider: openai
    api_key_env: OPENAI_API_KEY
    tier: budget
    price_in: 0.00015   # per 1K
    price_out: 0.0006
  - id: claude-3-5-sonnet-20241022
    provider: anthropic
    api_key_env: ANTHROPIC_API_KEY
    tier: reasoning
    price_in: 0.003
    price_out: 0.015
pseudo_models:
  autoconduck: {transform: "x"}
  autoconduck-budget: {transform: "x*0.6"}
  autoconduck-expensive: {transform: "min(1,x*1.4+0.1)"}
```

API keys **never** written to YAML; only env var names. TUI sets env for current session and instructs user to persist in shell profile.

---

## 9. Inter-Module Interaction Matrix

| Caller → Callee | Trigger | Data Passed | Return | Sync/Async | Failure |
|-----------------|---------|-------------|--------|------------|---------|
| `main` → `config` | startup, `edit`, `uninstall` | path | `Config` object | sync | missing file → onboarding |
| `main` → `tui` | no-args | `Config|None` | patched config | sync (textual loop) | user quit → exit |
| `main` → `proxy` | after config ready | `Config` | running server | async | port in use → prompt or auto-increment |
| `proxy` → `gatekeeper` | every pseudo-model request | `ChatRequest` | `Decision` | sync | never fails (pure fn) |
| `gatekeeper` → `evaluator` | tier 3 ambiguous | `messages[-1]`, `TurnState` | `T_i` float | sync | never fails |
| `proxy` → `evaluator` | fast-path | same | `T_i` | sync | clamp to 0.5 on error |
| `proxy` → `pricing` | both paths | `T_i'`, `t_in/t_out`, `models` | `ModelEntry` | sync | fallback cheapest |
| `proxy` → `orchestrator` | slow-path | `ChatRequest` | `OrchestratorResult` | async | degrade to fast |
| `orchestrator` → `pricing` | after compaction | `T_i'` | `ModelEntry` | sync | fallback |
| `orchestrator` → LiteLLM | planning + workers | `messages`, `model` | `completion` | async | retry once → degrade |
| `proxy` → LiteLLM | final execution | mutated `ChatRequest` | SSE stream | async | propagate as 502 with retry hint |
| `proxy` → `state` | after each turn | `model_id`, `usage` | — | sync | log only |
| `proxy` → `telemetry` | after each turn | `RoutingEvent` | — | sync | never fails |
| `proxy` → `cache` | before routing | `hash` | `cached SSE|None` | sync | miss → proceed |
| `tui` → `agents/*` | onboarding | `Config` | backup paths | sync | per-adapter error shown, continue |
| `tui` → `model_presets` | ingestion | provider keys | `list[ModelEntry]` | async (httpx) | fallback pricing file |

**Critical invariant:** `gatekeeper` and `evaluator` have **no async dependency** and no I/O; they are pure and testable. `pricing` is sync except `record_usage` side-effect. Only `proxy` and `orchestrator` are async.

---

## 10. Configuration & State Model

### 10.1 Persistent Config (`~/.autoconduck/config.yaml`)

Validated by `pydantic` on load; version field for future migrations.

### 10.2 Ephemeral State (`state.py` + `~/.autoconduck/state.json`)

```json
{
  "ema": {"fix": 380, "refactor": 2400, "default": 800},
  "sessions": {"a1b2c3": {"last_T": 0.82, "last_was_reasoning": true, "ts": 171782...}},
  "error_windows": {"gpt-4o-mini": [{"ts":..., "ok":false}, ...]}
}
```

- Flushed on `SIGTERM` and every 10 `record_usage` calls (debounced write).
- `sessions` keyed by `hash(system_prompt + first_user_message)` or explicit `x-session-id` header if agent sends it; LRU capped 200 entries, TTL 30 min.
- `state.json` is optional; if missing/corrupt, re-initialized with defaults (no crash).

### 10.3 Backups (`~/.autoconduck/backups/<agent>/<ISO8601>.bak`)

- Verbatim file copies, not diffs.
- `uninstall` restores most recent backup per agent; if no backup, removes only AutoConduck blocks.
- Retention: keep last 5 per agent; older pruned on new patch.

---

## 11. Agent Adapter Subsystem

### 11.1 Adapter Contract (`agents/base.py`)

```python
class BaseAdapter(ABC):
    id: str                          # "claude_code", "opencode", ...
    display_name: str
    @abstractmethod
    def detect(self) -> bool: ...    # PATH + filesystem scan
    @abstractmethod
    def config_paths(self) -> list[Path]: ...
    def backup(self, path: Path) -> Path: ...
    def patch(self, config: Config) -> None: ...
    def revert(self) -> None: ...
    def validate(self) -> bool: ...  # post-patch sanity check
```

**Patch mechanics:** each adapter knows its file format (JSON, YAML, TOML, env). It inserts/updates an AutoConduck-delimited block:

```
# BEGIN AUTOCONDUCK — do not edit between these markers
[autoconduck config]
# END AUTOCONDUCK
```

For JSON (e.g., `settings.json`), it merges keys under `"autoconduck"` namespace and tracks them for revert.

### 11.2 Per-Agent Notes

| Agent | Config File(s) | Patch Strategy |
|-------|----------------|----------------|
| Claude Code | `~/.config/claude/settings.json` or `~/.claude.json` | Merge `env` + `model` overrides; add `autoconduck` endpoint |
| OpenCode | `opencode.json` / `.opencode/config.json` | Insert `providers` entry pointing to `http://127.0.0.1:<port>/v1` |
| Aider | `.aider.conf.yml` / env | Write `openai_api_base` |
| Continue | `~/.continue/config.json` | Add `models` array with 3 entries + `apiBase` |
| Cursor | `~/.cursor/settings.json` | Model override; docs-checked |
| Kilo Code | `kilo-config.json` | Generic OpenAI-compatible block |
| Generic OpenAI | Env-based (`OPENAI_API_BASE`) | Instructional — TUI shows export command |

Each adapter is behind a feature flag; unsupported agents fall back to `generic_openai`.

### 11.3 Detection Algorithm

1. `shutil.which("<agent-binary>")` check.
2. Config file existence check.
3. Mark as `detected` only if (1) OR (2). TUI shows status with icon.

---

## 12. Packaging & Distribution Architecture

### 12.1 Binary Build (`npm-packaging/build.py`)

```
Input:  autoconduck/ + requirements.txt
Tool:   PyInstaller (primary) or Nuitka (alt for smaller binary on linux)
Steps:
  1. Create venv, pip install -r requirements.txt
  2. PyInstaller spec: onefile, console=True, hidden-imports=[litellm, tiktoken, textual]
  3. Exclude unused LiteLLM providers (strip to user-selected or all if generic build)
  4. Bundle pricing_fallback.json, tiktoken encodings
  5. Output: dist/autoconduck-<platform>-<arch>[/.exe]
  6. shasum + version stamp (from git tag)
Matrix: darwin-arm64, darwin-x64, linux-x64, linux-arm64, win32-x64
```

**Size control:** `litellm` is largest dep (~25 MB). Use PyInstaller `--exclude-module` for providers not needed; lazy import where possible.

### 12.2 npm Shim (`npm-packaging/autoconduck/`)

`package.json`:

```json
{
  "name": "autoconduck",
  "version": "0.1.0",
  "bin": { "autoconduck": "bin/autoconduck.js" },
  "optionalDependencies": {
    "autoconduck-darwin-arm64": "0.1.0",
    "autoconduck-darwin-x64": "0.1.0",
    "autoconduck-linux-x64": "0.1.0",
    "autoconduck-linux-arm64": "0.1.0",
    "autoconduck-win32-x64": "0.1.0"
  }
}
```

`bin/autoconduck.js` (Node):

```js
// Resolve platform package, locate binary, spawn with inherited stdio
const plat = `${process.platform}-${process.arch}`; // map darwin/arm64 etc.
const bin = require.resolve(`autoconduck-${plat}/bin/autoconduck`);
spawn(bin, process.argv.slice(2), {stdio:'inherit'});
```

Each `autoconduck-<plat>` package contains only the binary + minimal `package.json`.

### 12.3 Release Flow

1. `git tag vX.Y.Z` → CI (GitHub Actions) matrix builds binaries.
2. `build.py` verifies each binary boots (`--help` + `/healthz` smoke).
3. `npm publish` for wrapper + 5 platform packages (via `npm publish --access public` per package).
4. User: `npm install -g autoconduck` → npm resolves `optionalDependencies` for current platform → `autoconduck` command available.

---

## 13. Cross-Cutting Concerns

### 13.1 Concurrency Model

- FastAPI async everywhere in `proxy`/`orchestrator`.
- `evaluator`/`gatekeeper`/`pricing.select` are sync but called from async context (no `await` needed; they are CPU-bound <1 ms).
- Orchestrator workers: `asyncio.Semaphore(4)` shared; `asyncio.gather(..., return_exceptions=True)`.
- `state.py` guarded by `asyncio.Lock` for EMA/error-window mutations; reads are lock-free (copy-on-read).

### 13.2 Token Estimation

- `tiktoken` encoding `cl100k_base` cached globally; `len(encoding.encode(text))` per message.
- `pricing_fallback.json` also stores per-model tokenizer hint (claude → `claude` tokenizer via `tiktoken` approximation; acceptable error <10% — corrected by EMA).

### 13.3 Streaming & Cancellation

- Upstream request is an `asyncio.Task`; disconnect monitor is a second task polling `request.is_disconnected()` every 50 ms.
- On disconnect: `upstream_task.cancel()` + `await upstream_response.aclose()` (httpx) + break streaming loop.
- Client receives no further chunks; server logs `cancelled`.

---

## 14. Observability, Telemetry & UX Surfaces

### 14.1 Live Dashboard (TUI)

- Polls `telemetry.py` ring buffer (in-process) or `/stats` HTTP when proxy is separate process.
- Widgets: routing decision table (last 50), cost savings gauge, latency histogram (p50/p95), degraded model banner.
- Hotkeys: `e` → model selection, `r` → re-patch agents, `q` → quit (leaves proxy running if `--headless` was used; otherwise stops).

### 14.2 `/stats` & `/healthz`

- `/stats` is the headless equivalent of the dashboard; JSON for scripting/CI.
- `/healthz` used by TUI to detect proxy liveness and by `autoconduck start --headless` health checks.

### 14.3 Logging

- Structured JSONL to `~/.autoconduck/logs/routing.jsonl` (one line per `RoutingEvent`).
- Console logs via `uvicorn` with `log_level` from config.
- No PII beyond prompt length/file count; prompt content not logged by default (opt-in debug flag).

---

## 15. Error Handling & Degradation Strategy

| Failure Point | Behavior | Client Sees |
|---------------|----------|-------------|
| `gatekeeper` ambiguous LLM timeout/error | Default `FAST` | Normal routing (fast path) |
| `evaluator` exception | Clamp `T_i = 0.5` → pricing picks balanced tier | Normal |
| `pricing` all models degraded | Pick cheapest available (ignore degraded flag) | Normal |
| `litellm.model_cost` fetch fail | Use `pricing_fallback.json` | Normal |
| Orchestrator `TaskPlan` validation fail (1st) | Retry once with repair prompt | No visible delay beyond ~1 s |
| Orchestrator validation fail (2nd) | `degraded_to_fast=True` → route original request via fast path | Normal response via fast model |
| Orchestrator worker timeout (30 s) | Exclude worker, compact remaining | Normal (partial compaction) |
| Orchestrator all workers fail | Degrade to fast path | Normal |
| Upstream provider 429/5xx | `pricing.record_usage(is_error=True)` → on next turn that model is bypassed if error_rate>20%; proxy returns upstream error verbatim (with retry hint header) | Upstream error (expected) |
| Client disconnect | Cancel upstream immediately | Connection closed |
| Config corrupt | TUI shows error, offers reset; `proxy` refuses to start with clear message | N/A |
| Port in use | Auto-increment probe (`port+1` … `port+10`) or prompt in TUI | N/A |

**Principle:** never throw unhandled exception to agent for any internal routing/orchestration failure. The only errors surfaced are *provider* errors (which the agent already handles).

---

## 16. Performance Budgets

| Path | Budget | Measured At |
|------|--------|-------------|
| Gatekeeper (regex + delta score) | <3 ms p50, <5 ms p99 | `gatekeeper.classify` wall time |
| Pricing select + EMA | <1 ms | `pricing.select` |
| Total fast-path overhead (pre-forward) | **<5 ms p50** | proxy timestamp before `acompletion` |
| Ambiguous LLM call | 300–800 ms (cheap model) | only for 0.40–0.55 band (~10–15% of traffic est.) |
| Orchestrator planning | 800–1500 ms | slow path only |
| Orchestrator workers (parallel) | 2–8 s (4× parallel, 30 s cap each) | slow path only |
| SSE first-byte latency (fast path) | overhead + provider TTFB | end-to-end |

Fast path is the hot path; slow path is intentionally slower but yields higher quality via parallelism.

---

## 17. Security & Reversibility

- **API keys:** stored only as env var references; TUI masks input (`*`); keys passed to LiteLLM via `os.environ` at runtime, never written to disk. Optional `keyring` integration for OS keychain (future).
- **Backups:** verbatim copy before any write; permissions preserved; `uninstall` restores and verifies byte-equality with backup.
- **Patch isolation:** all modifications inside delimited blocks; re-running setup diffs only those blocks (idempotent).
- **No remote code execution:** orchestrator workers are LLM calls only; no filesystem mutation by AutoConduck itself.
- **Localhost only:** proxy binds `127.0.0.1` by default; no `0.0.0.0` unless explicit `--host` flag.

---

## 18. Implementation Phases & Dependency Graph

```
Phase 0 ──► Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4 ──► Phase 5 ──► Phase 6
 Setup      Core        Routing     Orchestr.   Proxy       TUI         Packaging
                        Logic                 + Streaming              + Release
```

| Phase | Deliverables | Dependencies | Verification |
|-------|--------------|--------------|--------------|
| **0. Scaffolding** | `autoconduck/` package layout, `requirements.txt`, `config.py` (schema + load/save), `state.py`, `telemetry.py` stubs, `pricing_fallback.json` | None | `pytest tests/test_config.py` |
| **1. Pricing & Evaluator** | `evaluator.py` (delta scorer, stack boost, hysteresis), `pricing.py` (EMA, log scaler, degraded window, `transform()`), `model_presets.py` | Phase 0 | Unit tests: known prompts → expected `T_i`; EMA convergence; log scaling monotonicity |
| **2. Gatekeeper** | `gatekeeper.py` (3 tiers, ambiguous zone), ambiguous LLM call stub (mockable) | Phase 1 | Unit tests: regex fast-path, slow keywords, ambiguous band boundaries; mock LLM |
| **3. Orchestrator** | `orchestrator.py` (TaskPlan schema, planner, retry, semaphore pool, compaction) | Phase 1+2 | Unit + integration: schema validation, retry→fallback, semaphore limit, compaction <1k tokens |
| **4. Proxy** | `proxy.py` (FastAPI app, all endpoints, SSE streaming, disconnect cancellation, mutation, cache) | Phase 0–3 | Integration: `httpx` mock upstream, streaming tests, disconnect cancellation test, `/v1/models` discovery |
| **5. Agent Adapters + TUI** | `agents/*` (all 7 adapters), `tui.py` (onboarding + dashboard) | Phase 0,4 | Manual TUI walkthrough + adapter unit tests (temp config files, backup/restore assertions) |
| **6. Packaging & Release** | `npm-packaging/build.py`, PyInstaller spec, npm wrapper, CI matrix | Phase 5 | Smoke: binary boots, `autoconduck --help`, proxy `/healthz`, `npm pack` dry-run |

**Critical path:** 1 → 2 → 4 (core value). Orchestrator (3) and TUI (5) can parallelize after 1.

**Estimated sequencing for single implementer:** 0 (0.5d) → 1 (1d) → 2 (0.5d) → 3 (1.5d) → 4 (1.5d) → 5 (2d) → 6 (1d) = ~8 days.

---

## 19. Testing & Verification Plan

| Area | Test Type | Tool | Key Cases |
|------|-----------|------|-----------|
| `config.py` | unit | pytest | missing file, env override, invalid YAML, preserver of `api_key_env` |
| `evaluator.py` | unit | pytest | short fix → low T, architecture prompt → high T, stack trace boost +0.25 cap, hysteresis clamp |
| `pricing.py` | unit | pytest | EMA update α=0.1 convergence, log cost ordering, degraded bypass, transform bias |
| `gatekeeper.py` | unit | pytest | Tier1 regex, Tier2 keywords, ambiguous band, mock LLM FAST/SLOW, timeout → FAST |
| `orchestrator.py` | unit + async | pytest-asyncio | valid plan, retry-once, double-fail→fallback, semaphore 4, 30s timeout, compaction token limit |
| `proxy.py` | integration | pytest-asyncio + httpx MockTransport | passthrough non-pseudo model, fast/slow routing, SSE streaming, disconnect cancel, cache hit, `/v1/models` shape |
| `agents/*` | unit (tmpdir) | pytest | detect, backup creation, patch idempotence, revert byte-equality |
| `tui.py` | manual + snapshot | textual pilot | onboarding flow, dashboard live update, hotkeys |
| Packaging | smoke (CI) | bash | binary size 40–70MB, `--version`, `/healthz` after spawn, npm shim resolves correct platform |
| E2E | manual | real agents | Claude Code + OpenCode picking `autoconduck` and completing a fix + a refactor task |

**Coverage target:** >85% on `gatekeeper`, `evaluator`, `pricing`, `orchestrator`, `proxy` (core routing). Adapters and TUI lower is acceptable with manual verification.

---

## 20. Risks & Open Decisions

| # | Risk / Decision | Impact | Mitigation / Recommendation |
|---|-----------------|--------|-----------------------------|
| 1 | **PyInstaller vs Nuitka** — size vs build time vs compatibility | Binary size, startup latency | Spike both on `linux-x64` early in Phase 6; choose per platform if needed (Nuitka often smaller on linux) |
| 2 | **`tiktoken` in binary** bloats size + needs encoding files | ~5–10 MB | Bundle only `cl100k_base`; lazy load; verify PyInstaller collects `tiktoken_ext` |
| 3 | **LiteLLM async streaming cancellation** — `acompletion` may not respect `task.cancel()` | Wasted billing on disconnect | Wrap with `httpx` direct fallback that is cancellation-aware; test with mock delayed upstream |
| 4 | **Slow-path streaming UX** — agent waits for compaction before any SSE byte | Perceived latency 3–8 s | Option: emit SSE comment `": autoconduck orchestrating...\n\n"` keepalive if agent tolerates; document tradeoff |
| 5 | **Session tracking for hysteresis** — no explicit session ID from agents | Hysteresis may be per-process not per-conversation | Key by `hash(system_prompt + first_user_msg)` + LRU; acceptable best-effort; log when ambiguous |
| 6 | **Agent config formats drift** (Cursor, Kilo) | Patch breaks on agent update | `validate()` post-patch + `generic_openai` fallback; version-pin adapter tests |
| 7 | **npm `optionalDependencies` on exotic arch** (linux-arm64) | Install succeeds but binary missing | Shim detects missing binary and prints actionable error + manual download link |
| 8 | **Ambiguous LLM call adds cost/latency** | 10–15% of turns pay extra | Monitor ratio; if too high, widen bands or make bands configurable in `config.yaml` |
| 9 | **EMA cold start** — first estimates inaccurate | Early cost estimates off | Seed `intent_table` from blueprint examples; EMA corrects within ~10 turns per intent |
| 10 | **Port conflicts** (11434 used by Ollama) | Proxy fails to start | Default to `11434` but auto-probe `11435+`; TUI shows chosen port and patches agents accordingly |

---

## Appendix A — File-Level Responsibility Checklist

Every file must be fully implemented (no `TODO`/`pass` stubs) before Phase gate:

- [ ] `main.py` — all subcommands, signal handling, port probing
- [ ] `config.py` — schema, load/save, env/flag precedence, validation
- [ ] `state.py` — TurnState, EMAState, ErrorWindow, persistence
- [ ] `telemetry.py` — ring buffer, JSONL, `/stats` aggregation
- [ ] `cache.py` — hash, LRU, cap (if enabled)
- [ ] `model_presets.py` — ingestion, fallback, normalization
- [ ] `evaluator.py` — delta scorer, stack boost, hysteresis, tiktoken
- [ ] `pricing.py` — EMA, log scaler, tier select, degraded failover, transform
- [ ] `gatekeeper.py` — 3 tiers, ambiguous LLM call, mockable
- [ ] `orchestrator.py` — TaskPlan, planner, retry, semaphore pool (4), 30 s timeout, compaction <1k
- [ ] `proxy.py` — FastAPI, `/v1/*`, `/stats`, `/healthz`, SSE, disconnect, mutation, cache check
- [ ] `tui.py` — onboarding screens, dashboard, hotkeys, proxy background thread
- [ ] `agents/base.py` — ABC, backup/patch/revert with delimited blocks
- [ ] `agents/*.py` — 7 adapters fully implemented per table in §11.2
- [ ] `npm-packaging/build.py` — matrix, version stamp, shasum, smoke
- [ ] `npm-packaging/autoconduck/bin/autoconduck.js` — platform resolver
- [ ] `pricing_fallback.json` — checked-in pricing snapshot

---

## Appendix B — Key Regex & Constants (Single Source of Truth)

```python
# gatekeeper.py
REGEX_FAST = re.compile(r"^(fix|format|typo|rename|docstring|check syntax|where is|grep)\b", re.I)
SLOW_KEYWORDS = ["refactor application","build feature","architecture","backtesting",
                 "migrate","rewrite entire","monorepo","codebase-wide"]
AMBIGUOUS_LOW, AMBIGUOUS_HIGH = 0.40, 0.55
FAST_PROMPT_MAX_LEN = 120

# evaluator.py
STACK_RE = re.compile(r"(Traceback|at\s+\w+\.\w+\(|UnhandledPromiseRejection|Error:|Exception:)", re.I)
HYSTERESIS_THRESHOLD = 0.80
HYSTERESIS_CLAMP = 0.50

# pricing.py
EMA_ALPHA = 0.1
DEGRADED_ERROR_RATE = 0.20
DEGRADED_WINDOW_SECONDS = 300
DEGRADED_MIN_SAMPLES = 5

# orchestrator.py
MAX_WORKERS = 4
WORKER_TIMEOUT_S = 30
COMPACTION_TOKEN_LIMIT = 1000
PLAN_RETRIES = 1  # retry once → fallback

# proxy.py
DEFAULT_PORT = 11434
PSEUDO_MODELS = {"autoconduck","autoconduck-budget","autoconduck-expensive"}
DISCONNECT_POLL_MS = 50
```

---

*End of Architecture Plan — ready for phased implementation per §18.*
