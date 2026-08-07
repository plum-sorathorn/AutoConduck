# AutoConduck — Unified Control & Data Flow Schema

> **Status:** Canonical (resolves conflicts in `ARCHITECTURE.md`)  
> **Date:** 2026-08-08  
> **Audience:** Implementers  
> **Authority:** This document wins on ownership, decision points, and data contracts. `ARCHITECTURE.md` remains the broader plan; where they disagree, **this file is source of truth**.

---

## 0. Problem Framing

AutoConduck is a localhost OpenAI-compatible proxy that:

1. Classifies each turn (fast / slow / ambiguous),
2. Selects a real model under a cost/quality bias,
3. Optionally fans complex work into a subagent DAG,
4. Streams the provider response back unchanged.

Implementation is pre-code (`ARCHITECTURE.md` only). The plan has **ownership collisions** that would cause double-scoring, async-in-sync modules, and unclear slow-path model selection. This schema freezes those decisions before Phase 1+.

---

## 1. Ownership Conflicts — Resolved

| # | Conflict | Evidence in ARCHITECTURE.md | Canonical Owner | Rule |
|---|----------|----------------------------|-----------------|------|
| O1 | **Ambiguous FAST/SLOW LLM call** | §6.3 puts it in gatekeeper; §6.7 puts it in proxy; §6.3 also says gatekeeper is “no async” | **`proxy.py`** | Gatekeeper is **pure sync**. On band hit it returns `path="ambiguous"` + `T_i`. Proxy owns the async force-choice call. |
| O2 | **Who calls `evaluator.score`** | Gatekeeper (tier 3) and proxy (fast path) both call it | **`gatekeeper` only (for path)**; **`proxy` once more for pricing input if path did not already produce `T_i`** | Gatekeeper always returns `T_i` when it scored (tier 3). Tier 1/2 may leave `T_i=None`. Proxy ensures a single `T_i` exists before transform (score once if missing). **Never score twice for the same turn.** |
| O3 | **`transform(T, pseudo)` location** | §6.5 “inside pricing before select”; §6.8 “in proxy after evaluator” | **`pricing.transform` pure fn**; **proxy is the only caller** | Pricing owns the math; proxy owns *when* it runs (once per turn, after path settled, before `select`). Orchestrator must **not** call `transform`. |
| O4 | **Slow-path final model selection** | §7.2 implies proxy; §9 says orchestrator→pricing | **`proxy.py`** | Orchestrator returns `OrchestratorResult` only (context + plan + flags). Proxy computes `T_i` → `T_i'` → `pricing.select` → mutates payload → final LiteLLM call. |
| O5 | **“T' from max worker T”** | §7.2 vague | **Dropped** | Workers do not produce complexity scores. Slow path uses the **same turn-level `T_i`** (from evaluator on last user message) as fast path, then `transform`. Optional future: bump `T_i` by fixed `+0.1` when `degraded_to_fast=False` (config flag); default **off**. |
| O6 | **EMA / error-window writes** | pricing vs state vs proxy | **`pricing.record_usage` / `record_error`** write through **`state.py`**; **proxy** is sole post-turn caller | Orchestrator workers report success/fail to proxy via result fields; proxy records degraded signals for worker models **and** final execution model. |
| O7 | **LiteLLM call sites** | proxy + orchestrator both call LiteLLM | **Split by purpose** | Orchestrator: plan + workers only. Proxy: ambiguous force-choice + final execution + passthrough. No recursive orchestrator. |
| O8 | **Cache key / hit serving** | proxy + cache | **`cache.py`** implements store/load; **`proxy`** decides hit/miss and whether cache is enabled | Cache never routes. |
| O9 | **Telemetry emission** | proxy-centric | **`proxy`** (and orchestrator only for internal plan/worker spans nested in the same `RoutingEvent`) | One `RoutingEvent` per client request. |
| O10 | **Config reload** | main/tui/proxy | **`config.get_config()`** singleton; **TUI `edit`** or SIGHUP-equivalent triggers reload; proxy reads config at request start (cheap ref) | No hot-reload mid-stream. |

### Non-negotiable invariants

1. **C2:** Fast path pre-forward overhead p50 &lt; 5 ms → gatekeeper + evaluator + pricing.select stay CPU-only, no I/O.
2. **C3:** Internal failures degrade to fast path / cheapest model; never 500 for routing bugs.
3. **C7:** Client-visible wire format is OpenAI-compatible; only `model` (and optional injected system context) change.
4. **Single scorer pass:** at most one `evaluator.score` per request.
5. **Single transform pass:** at most one `transform` per request.
6. **Single final completion:** exactly one streaming/non-streaming completion returned to the client (workers are internal).

---

## 2. Component Responsibility Matrix (RACI-lite)

| Component | Owns | May call | Must not |
|-----------|------|----------|----------|
| **`main.py`** | Process lifecycle, CLI, signal handling, port probe, spawn proxy/TUI | config, tui, uvicorn(proxy) | Routing, LiteLLM |
| **`config.py`** | Schema, load/save, precedence (flags &gt; env &gt; file), `~/.autoconduck` paths | pydantic, fs | LLM, scoring |
| **`model_presets.py`** | Provider discovery, normalize `ModelEntry`, pricing fallback ingest | litellm.model_cost, httpx (onboarding) | Request path |
| **`gatekeeper.py`** | Path decision: `fast` \| `slow` \| `ambiguous` | evaluator (sync) | async, pricing, LiteLLM, state writes |
| **`evaluator.py`** | `T_i ∈ [0,1]` from last message + TurnState | tiktoken, state **read** (TurnState) | path decision, pricing, async |
| **`pricing.py`** | token est, log-cost matrix, tier buckets, `select`, `transform`, EMA update API, degraded check | state R/W (EMA, ErrorWindow), config models | path decision, HTTP, orchestration |
| **`orchestrator.py`** | TaskPlan, plan LLM, retry-once, worker pool, compaction | LiteLLM (plan/workers), pricing **read-only helpers only if needed for cheap plan model id via injected `plan_model_id`** | `transform`, final client completion, gatekeeper |
| **`proxy.py`** | HTTP surface, pipeline orchestration, mutation, stream, cancel, passthrough | all core modules, LiteLLM final/ambiguous | Agent file patching |
| **`state.py`** | TurnState ring, EMA map, ErrorWindows, session keys, persistence | fs (debounced) | Business decisions |
| **`cache.py`** | Disk LRU of SSE/JSON transcripts | fs | Routing |
| **`telemetry.py`** | Ring buffer, JSONL, `/stats` aggregate | fs append | Routing |
| **`tui.py`** | Onboarding, dashboard, agent patch UX | agents, model_presets, config, telemetry | Inline scoring |
| **`agents/*`** | detect/backup/patch/revert | fs | Proxy runtime |

---

## 3. Control-Flow Schema (Canonical Pipeline)

### 3.1 Process-level control

```
main.parse_argv
  ├─ uninstall     → agents.revert* → exit
  ├─ edit          → tui.model_select → config.save → exit
  ├─ start --headless → config.load → proxy.serve (foreground)
  └─ (default)
        ├─ no config → tui.onboard → config.save → proxy.serve + tui.dashboard
        └─ has config → proxy.serve + tui.dashboard
```

### 3.2 Request-level control — `POST /v1/chat/completions`

```
                    ┌──────────────────────────────────────┐
                    │  D0 VALIDATE ChatRequest (pydantic)  │
                    └──────────────────┬───────────────────┘
                                       │
                    ┌──────────────────▼───────────────────┐
                    │  D1 model ∈ PSEUDO_MODELS ?          │
                    └──────────┬─────────────────┬─────────┘
                          no   │                 │ yes
                               ▼                 ▼
                     PASSTHROUGH           D2 cache hit?
                     LiteLLM as-is              │
                     + telemetry           yes ─┴─ no
                     return                     │
                                                ▼
                                     start disconnect monitor
                                                │
                    ┌───────────────────────────▼───────────────────────────┐
                    │  D3 gatekeeper.classify(req, turn_state) → Decision   │
                    │     path ∈ {fast, slow, ambiguous}, T_i?: float|None  │
                    └───────────┬─────────────────┬─────────────────┬───────┘
                           fast │            slow │      ambiguous  │
                                │                 │                 ▼
                                │                 │         D4 force_choice LLM
                                │                 │         (proxy, cheap model,
                                │                 │          800ms, default FAST)
                                │                 │                 │
                                │                 │         path := FAST|SLOW
                                ▼                 ▼                 │
                          ┌─────┴──── join path ──┴─────────────────┘
                          │
                    ┌─────▼─────────────────────────────────────────┐
                    │  D5 ENSURE T_i                                  │
                    │  if Decision.T_i is None:                      │
                    │      T_i = evaluator.score(last, turn_state)   │
                    │  on error: T_i = 0.5                           │
                    └─────────────────────┬─────────────────────────┘
                                          │
                    ┌─────────────────────▼─────────────────────────┐
                    │  D6 path == slow ?                            │
                    └──────────┬──────────────────────┬─────────────┘
                          yes  │                      │ no (fast)
                               ▼                      │
                    orchestrator.plan_and_execute     │
                      ├─ plan ok → compacted_context  │
                      └─ fail  → degraded_to_fast     │
                               │                      │
                               └──────────┬───────────┘
                                          ▼
                    ┌─────────────────────────────────────────────┐
                    │  D7 T_i' = pricing.transform(T_i, pseudo)   │
                    │  (t_in,t_out) = pricing.estimate_tokens(...)│
                    │  model = pricing.select(T_i', models, ...)  │
                    └─────────────────────┬───────────────────────┘
                                          ▼
                    mutate payload: model := real_id
                    if compacted_context: prepend system msg
                                          │
                    ┌─────────────────────▼───────────────────────┐
                    │  D8 forward LiteLLM/httpx (stream|json)     │
                    │  disconnect → cancel upstream               │
                    └─────────────────────┬───────────────────────┘
                                          ▼
                    record_usage / record_error → state
                    update TurnState (last_T, used_reasoning)
                    telemetry.push(RoutingEvent)
                    optional cache.store
                    return to client
```

### 3.3 Decision points (normative)

| ID | Location | Inputs | Outputs | Failure default |
|----|----------|--------|---------|-----------------|
| **D0** | proxy | raw JSON | `ChatRequest` or 400 OpenAI error | 400 |
| **D1** | proxy | `model` | passthrough vs route | — |
| **D2** | proxy | cache key, `cache_enabled` | hit body or miss | miss |
| **D3** | gatekeeper | messages, attachments meta, TurnState | `Decision` | path=fast, T_i=0.5 |
| **D3.T1** | gatekeeper | len&lt;120 ∧ REGEX_FAST | path=fast, T_i=None | — |
| **D3.T2** | gatekeeper | attachments&gt;3 ∨ SLOW_KEYWORDS | path=slow, T_i=None | — |
| **D3.T3** | gatekeeper | else score | if T&lt;0.40 fast; &gt;0.55 slow; else ambiguous; **always set T_i** | T_i=0.5, fast |
| **D4** | proxy | last msg truncated, cheapest model | path FAST\|SLOW | FAST |
| **D5** | proxy | Decision.T_i | definitive T_i | 0.5 |
| **D6** | proxy | path | run orchestrator or skip | skip / degrade fast |
| **D7** | proxy+pricing | T_i, pseudo, models | real `ModelEntry` | cheapest non-empty |
| **D8** | proxy | mutated req | SSE/JSON to client | upstream error verbatim |

### 3.4 Gatekeeper internal (sync only)

```
classify(request, history_meta, turn_state) -> Decision:
  prompt = last_user_text(request.messages)
  if len(prompt) < FAST_PROMPT_MAX_LEN and REGEX_FAST.match(prompt):
      return Decision(path="fast", reason="tier1_regex", T_i=None, elapsed_ms=…)
  if attachment_count(request) > 3 or any(k in prompt.lower() for k in SLOW_KEYWORDS):
      return Decision(path="slow", reason="tier2_override", T_i=None, elapsed_ms=…)
  T_i = evaluator.score(request.messages[-1], turn_state)  # sole score in gatekeeper
  if T_i < AMBIGUOUS_LOW:  return Decision("fast", "tier3_low", T_i, …)
  if T_i > AMBIGUOUS_HIGH: return Decision("slow", "tier3_high", T_i, …)
  return Decision("ambiguous", "tier3_band", T_i, …)
```

**Note:** Tier1/Tier2 intentionally skip scoring for latency; proxy D5 scores once before pricing.

---

## 4. Data-Flow Schema & Contracts

### 4.1 Wire contracts (client ↔ proxy)

#### `ChatRequest` (inbound)

```python
class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list  # OpenAI multimodal passthrough
    name: str | None = None
    tool_calls: list | None = None
    tool_call_id: str | None = None

class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    tools: list | None = None
    tool_choice: str | dict | None = None
    model_config = ConfigDict(extra="allow")  # passthrough unknown fields
```

**Headers (optional):**

| Header | Use |
|--------|-----|
| `x-session-id` | Prefer as session key for TurnState |
| `authorization` | Forwarded upstream if present; local keys usually from env |

#### Outbound to client

- Streaming: verbatim upstream SSE (`data: …\n\n`, final `[DONE]`).
- Non-streaming: verbatim upstream JSON.
- Errors: OpenAI-shaped `{ "error": { "message", "type", "code" } }`.
- Proxy may add response header `x-autoconduck-model: <real_id>` (optional, documented) — **body model field stays whatever upstream returns**.

#### `GET /v1/models`

Fixed three pseudo-models (see ARCHITECTURE §8.2).

#### `RoutingEvent` (telemetry / stats)

```python
class RoutingEvent(BaseModel):
    ts: float
    request_id: str
    pseudo_model: str | None       # None if passthrough
    real_model: str
    path: Literal["fast", "slow", "ambiguous_resolved_fast",
                  "ambiguous_resolved_slow", "passthrough", "cache_hit"]
    gate_reason: str | None
    T_i: float | None
    T_i_prime: float | None
    degraded_to_fast: bool = False
    cache_hit: bool = False
    cost_est: float | None
    latency_overhead_ms: float
    latency_total_ms: float | None
    cancelled: bool = False
    error: str | None = None
    worker_ok: int | None = None
    worker_fail: int | None = None
```

### 4.2 Internal contracts

#### `Decision` (gatekeeper → proxy)

```python
class Decision(BaseModel):
    path: Literal["fast", "slow", "ambiguous"]
    reason: str
    T_i: float | None          # set on tier3; None on tier1/2
    elapsed_ms: float
```

#### `TurnState` (state → evaluator)

```python
class TurnState(BaseModel):
    session_key: str
    last_T: float | None = None
    used_reasoning_tier: bool = False   # last_T >= 0.80 after transform? see §4.4
    ts: float
```

Session key resolution order:

1. `x-session-id` header if present,
2. else `sha256(system_content + first_user_content)[:16]`,
3. else `"anon"`.

LRU 200, TTL 30 min.

#### Evaluator

```python
def score(last_message: Message, prev_state: TurnState | None) -> float:
    """Pure-ish: tiktoken + regex; reads prev_state only. Returns T_i in [0,1]."""
```

Signals (weights fixed in code, tunable later via config): token_len, code_ratio, file_refs, imperative, question_depth → weighted sum → sigmoid; stack boost +0.25 cap 1.0; hysteresis clamp to 0.50 if `used_reasoning_tier` unless stack present.

#### Pricing

```python
def transform(T: float, pseudo: str) -> float:
    if pseudo == "autoconduck-budget":    return T * 0.6
    if pseudo == "autoconduck-expensive": return min(1.0, T * 1.4 + 0.1)
    return T

def estimate_tokens(messages, intent: str) -> tuple[int, int]: ...
def select(T_i_prime: float, models: list[ModelEntry], t_in: int, t_out: int) -> ModelEntry: ...
def record_usage(model_id: str, actual_in: int, actual_out: int, intent: str) -> None: ...
def record_error(model_id: str) -> None: ...
def is_degraded(model_id: str) -> bool: ...
```

Tier buckets on `T_i'`:

| T_i' | Tier pool |
|------|-----------|
| &lt; 0.33 | budget |
| 0.33–0.75 | balanced |
| &gt; 0.75 | reasoning/expensive |

Within pool: min `C_m' = ln(1 + cost)` among non-degraded; if empty, cheapest overall.

#### Orchestrator

```python
class SubTask(BaseModel):
    id: str
    goal: str
    files: list[str] = []
    depends_on: list[str] = []
    output_contract: str

class TaskPlan(BaseModel):
    tasks: list[SubTask]  # min 2, max 6

class OrchestratorResult(BaseModel):
    compacted_context: str | None  # None if degraded
    plan: TaskPlan | None
    degraded_to_fast: bool
    reason: str
    worker_ok: int = 0
    worker_fail: int = 0
    plan_model_id: str | None = None
    worker_model_ids: list[str] = []
```

```python
async def plan_and_execute(
    request: ChatRequest,
    *,
    plan_model_id: str,      # injected by proxy (cheapest balanced)
    worker_model_id: str,    # injected by proxy (balanced default)
) -> OrchestratorResult: ...
```

**Proxy injects plan/worker model ids** via `pricing.select` with fixed synthetic T' (e.g. plan T'=0.40, workers T'=0.55) **before** calling orchestrator — keeps model policy in pricing, not hard-coded in orchestrator.

Compaction: &lt; `COMPACTION_TOKEN_LIMIT` (1000) tokens; prepended by proxy as:

```python
{"role": "system", "content": f"[AutoConduck context]\n{compacted_context}"}
```

inserted after existing system messages (or at index 0 if none).

#### Cache

```python
def make_key(pseudo: str, last_message: Message) -> str: ...  # sha256
def get(key: str) -> bytes | None: ...
def put(key: str, body: bytes) -> None: ...
```

### 4.3 Data movement diagram (logical)

```
ChatRequest
    │
    ├─► gatekeeper ──Decision(+T_i?)──► proxy
    │         │
    │         └─► evaluator ──T_i──► (embedded in Decision when tier3)
    │
    ├─► evaluator (D5 if T_i missing) ──T_i──► proxy
    │
    ├─► state:TurnState ──read──► evaluator
    │         ▲
    │         └── write (post-turn) ── proxy
    │
    ├─► [slow] orchestrator
    │         ├─ LiteLLM plan  ──TaskPlan──► workers
    │         ├─ LiteLLM ×N    ──texts──► compactor ──compacted_context──► proxy
    │         └─ OrchestratorResult ───────────────────────────────► proxy
    │
    ├─► pricing.transform(T_i) ──T_i'──► pricing.select ──ModelEntry──► proxy
    │         ▲
    │         └── models[] from config; EMA/degraded from state
    │
    ├─► LiteLLM final ──SSE/JSON──► client
    │         │
    │         └─ usage ──► pricing.record_* ──► state
    │
    └─► telemetry.RoutingEvent; optional cache.put
```

### 4.4 When is `used_reasoning_tier` set?

After final select (not raw T_i):

```
used_reasoning_tier := (T_i_prime >= 0.80) or (selected.tier == "reasoning")
```

Stored on session TurnState for next turn hysteresis. This ties hysteresis to **biased** complexity, matching user pseudo-model intent.

---

## 5. Sequence Diagrams (canonical)

### 5.1 Fast path (p90)

```
Agent → Proxy: POST chat.completions {model:autoconduck,…}
Proxy → Gatekeeper: classify
Gatekeeper → Evaluator: score (only if tier3; tier1 skips)
Gatekeeper → Proxy: Decision(fast, T_i?)
Proxy → Evaluator: score iff T_i is None
Proxy → Pricing: transform + estimate + select
Proxy → LiteLLM: acompletion(stream)
LiteLLM → Proxy → Agent: SSE
Proxy → Pricing/State/Telemetry: record
```

### 5.2 Ambiguous → resolved fast

```
… Gatekeeper → Decision(ambiguous, T_i=0.47)
Proxy → LiteLLM: force_choice (max_tokens=10, 800ms)
LiteLLM → Proxy: "FAST | …"
Proxy: path=fast; reuse T_i=0.47 (no rescore)
Proxy → Pricing → LiteLLM final → Agent
```

### 5.3 Slow path success

```
… Gatekeeper → Decision(slow, T_i?)
Proxy → Evaluator: ensure T_i
Proxy → Pricing: select plan_model (T'=0.40), worker_model (T'=0.55)
Proxy → Orchestrator: plan_and_execute(plan_model, worker_model)
Orchestrator → LiteLLM: plan JSON
Orchestrator → LiteLLM: workers (sem=4)
Orchestrator → Proxy: OrchestratorResult(context, degraded=false)
Proxy → Pricing: transform(T_i)+select execution model
Proxy: mutate + inject context
Proxy → LiteLLM final → Agent
```

### 5.4 Slow path degrade

```
Orchestrator → Proxy: degraded_to_fast=true
Proxy: skip inject; same as fast path from D7 using original messages
```

---

## 6. Inter-module call matrix (corrected)

| Caller → Callee | When | In | Out | Sync | On failure |
|-----------------|------|----|-----|------|------------|
| proxy → gatekeeper | every pseudo req | ChatRequest, TurnState | Decision | sync | Decision(fast,0.5) |
| gatekeeper → evaluator | tier3 only | last msg, TurnState | T_i | sync | raise → gatekeeper catches → fast/0.5 |
| proxy → evaluator | D5 if T_i None | last msg, TurnState | T_i | sync | T_i=0.5 |
| proxy → pricing.transform/select | after path+T_i | T_i, pseudo, models | ModelEntry | sync | cheapest |
| proxy → pricing (plan/worker ids) | before orchestrator | synthetic T' | ModelEntry | sync | cheapest balanced |
| proxy → orchestrator | path slow | req + model ids | OrchestratorResult | async | treat as degraded_to_fast |
| orchestrator → LiteLLM | plan/workers | msgs | text/json | async | retry plan once; drop worker |
| proxy → LiteLLM | D4 ambiguous, D8 final, passthrough | msgs | stream/json | async | upstream error / cancel |
| proxy → state | pre: get TurnState; post: update | keys, usage | TurnState | sync+lock | ignore write errors |
| proxy → telemetry | always end | RoutingEvent | — | sync | drop |
| proxy → cache | optional | key | body\|None | sync | miss |
| pricing → state | record_*, is_degraded | EMA/errors | — | sync+lock | log |
| main/tui → config/agents | lifecycle | — | Config | sync | user-facing error |

**Removed vs ARCHITECTURE §9:** `orchestrator → pricing.select` for final model.  
**Added:** proxy supplies plan/worker model ids; proxy owns D4.

---

## 7. Scalability Considerations

### 7.1 Current design envelope (v1)

| Dimension | v1 posture | Bottleneck | Scale lever |
|-----------|------------|------------|-------------|
| Concurrency | Single process, asyncio | GIL + LiteLLM | One proxy per user machine is enough; multi-agent clients share one port |
| Orchestrator fan-out | Semaphore(4) | Provider rate limits | Config `max_workers`; per-provider token bucket (v1.1) |
| State | In-memory + debounced JSON | Lock contention under high RPS | Shard locks by session_key; keep critical sections &lt;100µs |
| Telemetry ring | 500 events | Memory | Fixed size; JSONL async append via queue |
| Cache | Optional 100MB LRU | Disk I/O on hit path | Default off; if on, memory bloom + disk |
| Fast path CPU | tiktoken encode | Large last messages | Cap encode window (e.g. first 8k tokens of last msg) for scoring only |
| Ambiguous LLM | ~10–15% turns | Extra RTT/cost | Configurable bands; circuit-break if force-choice error_rate high → always fast |
| Streaming cancel | poll 50ms | Wasted tokens | Prefer httpx direct where LiteLLM cancel is weak |

### 7.2 What v1 explicitly does **not** scale to

- Multi-tenant / multi-user server (out of scope).
- Horizontal proxy cluster (shared state would need Redis — reject until needed).
- Persistent vector memory / long-term RAG.

### 7.3 Extension points (keep seams clean)

```
PathClassifier = Protocol  # gatekeeper implements
ComplexityScorer = Protocol  # evaluator implements
ModelSelector = Protocol  # pricing implements
PlanExecutor = Protocol  # orchestrator implements
```

Proxy depends on protocols → swap ML router later without rewriting SSE.

### 7.4 Performance budgets (gate tests)

| Metric | Budget |
|--------|--------|
| D3+D5+D7 wall (no ambiguous LLM, no orch) | p50 &lt; 5 ms |
| gatekeeper alone | p50 &lt; 3 ms |
| pricing.select | p50 &lt; 1 ms |
| D4 ambiguous | &lt; 800 ms timeout |
| orchestrator total | best-effort; client may wait 3–8 s before first SSE byte |
| state flush | every 10 records or SIGTERM; never on hot path await disk |

### 7.5 Backpressure

- Global `asyncio.Semaphore` for **outbound** LiteLLM calls (config `max_in_flight`, default 32) separate from orchestrator worker sem.
- On semaphore timeout: degrade to fast path with cheapest model **or** return 503 OpenAI-shaped if passthrough — prefer degrade for pseudo-models (C3).

---

## 8. Error & Degradation Map (control outcomes)

```
D0 fail                    → 400 to client
D4 timeout/parse           → path=fast
D5 exception               → T_i=0.5
D6 orch plan ×2 fail       → degraded_to_fast
D6 all workers fail        → degraded_to_fast
D6 partial workers         → compact survivors
D7 all degraded            → cheapest any tier
D8 provider 429/5xx        → record_error; pass through body
disconnect                 → cancel; RoutingEvent.cancelled=true
state/telemetry/cache fail → log; continue
```

---

## 9. Implementation Notes (for Phase 1–4)

1. **Implement modules in dependency order:** state/config → evaluator → pricing → gatekeeper → orchestrator → proxy.
2. **Gatekeeper unit tests must not mock LiteLLM** (no longer a dependency).
3. **Proxy integration tests** cover D4 with MockTransport.
4. **Pass `score_fn` into gatekeeper** for tests (ARCHITECTURE §8.3) — keep.
5. **Do not** put `transform` in proxy as a private copy; import `pricing.transform`.
6. **Single `RequestContext` dataclass** threaded through proxy pipeline to avoid arg soup:

```python
@dataclass
class RequestContext:
    request_id: str
    chat: ChatRequest
    pseudo: str | None
    session_key: str
    turn_state: TurnState | None
    decision: Decision | None = None
    T_i: float | None = None
    T_i_prime: float | None = None
    selected: ModelEntry | None = None
    orch: OrchestratorResult | None = None
    t0: float = 0.0
```

7. **ARCHITECTURE.md patches to apply later (doc drift):**
   - §6.3: remove ambiguous LLM from gatekeeper; return `ambiguous` only.
   - §6.7 step list: align with §3.2 here.
   - §6.5: transform called by proxy, defined on pricing.
   - §7.2: remove “max worker T”; final select always in proxy.
   - §9 matrix: delete orchestrator→pricing final select; add proxy D4.

---

## 10. Summary Recommendation

| Topic | Recommendation |
|-------|----------------|
| Pipeline owner | **proxy** is the sole control-plane conductor |
| Pure compute plane | gatekeeper + evaluator (+ pricing.select/transform) |
| Async plane | proxy + orchestrator + LiteLLM |
| State plane | state.py behind pricing/proxy writes |
| Path vs model | path ≠ model: path chooses *structure*; pricing chooses *model* |
| Slow path value | structure (DAG context), not a different scoring formula |
| Scale story | single-user asyncio + semaphores + protocol seams; no distributed state in v1 |

This schema is implementation-ready: every decision point has an owner, every cross-module edge has a typed contract, and every former ownership conflict has an explicit winner.
