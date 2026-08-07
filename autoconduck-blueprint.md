# AutoConduck — Unified Master Blueprint

## 1. Objective & Overview

AutoConduck is a local, zero-overhead model router and task orchestrator designed for open-source AI coding agents (Claude Code, OpenCode, Aider, Continue, Cursor, Kilo Code, and any OpenAI-compatible agent).

It presents itself to coding agents as three selectable pseudo-models:

- `autoconduck` (balanced)
- `autoconduck-budget` (cost-optimized)
- `autoconduck-expensive` (quality-optimized)

AutoConduck transparently routes every request turn to the cheapest real model capable of handling it, while automatically escalating complex, multi-file tasks to an asynchronous parallel subagent DAG.

---

## 2. Distribution & Packaging Architecture

- **Engine Core:** Written in Python 3.11+ using `FastAPI`, `httpx`, `LiteLLM`, `tiktoken`, `pydantic`, and `textual`.
- **Standalone Compiled Binary:** Compiled per OS/architecture target (`darwin-arm64`, `linux-x64`, `win32-x64`, etc.) via PyInstaller or Nuitka. Bundles the Python runtime, server, TUI, and stripped LiteLLM provider modules into a single binary (~40–70 MB).
- **npm / bun Distribution Shim:** Shipped as a thin npm package (`autoconduck`) using `optionalDependencies` for platform-specific binary packages (`autoconduck-darwin-arm64`, etc.).
- **User Experience:** Installed globally via a single command:

  ```bash
  npm install -g autoconduck  # or: bun add -g autoconduck
  ```

  End users require no local Python environment, pip, or virtual environments.

---

## 3. User Interaction & Auto-Configuration Engine

### First Run (`autoconduck`)

Launches an interactive, keyboard-only TUI (built with `textual`):

- **Agent Detection:** Auto-scans filesystem and PATH for supported coding agents (Claude Code, OpenCode, Aider, Continue, Cursor, Kilo Code).
- **Model Ingestion:** Users select model sources (Custom API keys or Anthropic/OpenAI/Google account presets). Live per-token pricing is fetched from LiteLLM's registry.
- **Atomic Config Patching:** Before modifying any configuration file, AutoConduck copies the original file to `~/.autoconduck/backups/<agent>/<timestamp>.bak`. Patches are written inside strictly delimited AutoConduck blocks.
- **Live Dashboard:** Launches the proxy and opens directly into a live monitoring dashboard displaying real-time routing decisions, latency, and cost savings.

### CLI Subcommands

- `autoconduck` — Opens live dashboard if configured; hotkey re-enters setup.
- `autoconduck start --headless` — Runs the background HTTP proxy only (for CI, systemd, launchd).
- `autoconduck edit` — Re-opens model selection without re-patching agent configs.
- `autoconduck uninstall` — Restores original agent configuration files from backups and stops the daemon.

---

## 4. End-to-End System Data Flow

```
Client (Coding Agent)
   │ POST /v1/chat/completions {model: "autoconduck*"}
   ▼
 proxy.py (FastAPI) ──▶ Client Disconnect Monitor & Disk Cache Check
   │
   ▼
 gatekeeper.py
   ├─ 1. Heuristic & Regex Filter (< 1ms) ───────────┐
   ├─ 2. Delta Evaluator & Score Calculation (< 3ms) ──┤
   └─ 3. Ambiguous Score Zone [0.40 - 0.55] ─────────┴─▶ Fast-Path or Slow-Path Decision
   │
   ├── [FAST-PATH] ──▶ evaluator.py (Delta Score + Error Boost + Hysteresis Clamp)
   │                       │
   │                       ▼
   │                   pricing.py (EMA Token Estimator + Logarithmic Cost Matrix)
   │                       │
   └── [SLOW-PATH] ──▶ orchestrator.py
                           ├─ Pydantic DAG Plan (Retry-Once -> Fast-Path Fallback)
                           ├─ Parallel Subagents (asyncio.Semaphore(4), 30s timeout)
                           └─ Context Compaction (<1k tokens) ──▶ Execution Model
   │
   ▼
 LiteLLM Async Core ──▶ Target Provider API ──▶ Streamed SSE Chunks back to Client
```

---

## 5. Detailed Component Specifications

### 5.1 `gatekeeper.py` — 3-Tier Dual-Path Classifier

Executes in under 3 ms to decide between Fast-Path (direct single model) and Slow-Path (DAG Orchestration):

- **Tier 1 (Fast-Path Regex Override):** If prompt length is `< 120` chars AND matches `^(fix|format|typo|rename|docstring|check syntax|where is|grep)`, route to Fast-Path.
- **Tier 2 (Slow-Path Override):** If attached files `> 3` OR prompt matches scope keywords (`["refactor application", "build feature", "architecture", "backtesting"]`), route to Slow-Path.
- **Tier 3 (Ambiguous Zone Classifier):** If the calculated complexity score falls within the ambiguous band (`0.40 ≤ T_i ≤ 0.55`), execute a single cheap, forced-choice LLM call returning a 2-token directive (`FAST` or `SLOW`) with a logged reasoning string.

### 5.2 `evaluator.py` — Delta Token Complexity Scorer

- **Delta Token Scoring:** Evaluates complexity `T_i ∈ [0.0, 1.0]` using only the newest message (`messages[-1]`) to prevent historical conversation bloat from inflating turn complexity.
- **Stack Trace Boost:** Automatically applies a bounded `+0.25` complexity boost if `messages[-1]` contains stack traces, compiler errors, or `UnhandledPromiseRejection`.
- **Hysteresis Cooldown:** If Turn `t-1` used a Reasoning Tier model (`T ≥ 0.80`), Turn `t` is clamped to `T ≤ 0.50` unless a new stack trace is detected.

### 5.3 `pricing.py` — Logarithmic Cost Matrix & EMA Token Corrector

- **Pricing Registry:** Ingests live model costs via `litellm.model_cost` with a local JSON fallback.
- **EMA Token Correction:** Estimates input tokens `T_in` (via `tiktoken`) and output tokens `T_out` (per intent category). Dynamically corrects predictions using an Exponential Moving Average (`α = 0.1`) based on actual usage returned in API completion responses.
- **Logarithmic Scaling:** Computes `Cost_m = Price_in · T_in + Price_out · T_out` and normalizes costs across available models using logarithmic scaling:

  ```
  Scaled Cost (C_m') = ln(1 + Cost_m)
  ```

- **Degraded Routing Failover:** If a target model's error rate exceeds 20% over the trailing 5 minutes, it is temporarily bypassed for the next-cheapest candidate.

### 5.4 `orchestrator.py` — Parallel Subagent DAG Engine

- **Structured Planning:** Requests a structured `TaskPlan` Pydantic schema from a fast-tier model.
- **Reliability Fallback:** If JSON schema validation fails, retries once. A second failure automatically abandons the Slow-Path and falls back to Fast-Path execution on the original request.
- **Parallel Worker Pool:** Spawns subagents using `asyncio.gather`, clamped by `asyncio.Semaphore(4)`. Each worker has an isolated file context and a strict 30-second timeout.
- **Context Compaction:** Summarizes worker findings into a structured contract (`< 1k tokens`) passed directly to the primary execution model.

### 5.5 `proxy.py` — Streaming Reverse Proxy & Endpoints

- `POST /v1/chat/completions` — Intercepts pseudo-models, runs routing pipeline, mutates payload, forwards upstream via LiteLLM, and streams SSE chunks back.
- `GET /v1/models` — Returns `autoconduck`, `autoconduck-budget`, and `autoconduck-expensive` for automatic GUI dropdown discovery.
- **Disconnect Cancellation:** Monitors `request.is_disconnected()` and cancels upstream provider HTTP requests immediately if the client disconnects.
- `/stats` & `/healthz` — Real-time telemetry, routing decision audit logs, cache hit ratios, and liveness checks.

---

## 6. Pseudo-Model Threshold Adjustments

The three selectable pseudo-models run on the same underlying engine, applying a scalar multiplier to the calculated task complexity score `T_i`:

| Pseudo-Model Alias | Threshold Formula | Routing Behavior |
|---|---|---|
| `autoconduck-budget` | `T_i' = T_i × 0.6` | Biases selection toward cheap/fast models. Escalates only for high-complexity tasks. |
| `autoconduck` (default) | `T_i' = T_i` | Balanced baseline cost/quality routing. |
| `autoconduck-expensive` | `T_i' = min(1.0, T_i × 1.4 + 0.1)` | Biases selection toward top-tier reasoning models even at moderate task complexity. |

---

## 7. File Structure

```
autoconduck/                        # Core Python source (compiled to binary)
├── main.py                         # CLI entrypoint & subcommands parser
├── proxy.py                        # FastAPI server, SSE streaming, disconnect monitor
├── gatekeeper.py                   # 3-tier fast/slow path classifier
├── evaluator.py                    # Delta-token complexity scorer & hysteresis clamp
├── pricing.py                      # LiteLLM cost matrix, EMA corrector, log scaler
├── orchestrator.py                 # Pydantic DAG planner, subagent pool, compactor
├── config.py                       # Configuration loader (YAML, env, flags)
├── tui.py                          # Textual interactive onboarding & live stats dashboard
├── model_presets.py                # Account-level provider model discovery
├── agents/                         # Agent auto-patching registry
│   ├── base.py                     # Base adapter class (detect, backup, patch, revert)
│   ├── claude_code.py
│   ├── opencode.py
│   ├── aider.py
│   ├── continue_dev.py
│   ├── kilocode.py
│   ├── cursor.py
│   └── generic_openai.py
└── requirements.txt                # Production dependencies

npm-packaging/                      # Standalone global distribution
├── autoconduck/                    # Thin npm wrapper package
│   ├── package.json                # Defines optionalDependencies per platform
│   └── bin/autoconduck.js          # Resolves and spawns OS-specific binary
├── autoconduck-darwin-arm64/
├── autoconduck-darwin-x64/
├── autoconduck-linux-x64/
├── autoconduck-linux-arm64/
├── autoconduck-win32-x64/
└── build.py                        # Cross-compilation build script (PyInstaller/Nuitka)
```

---

## 8. Hard Requirements & Technical Constraints

- **Zero Host Dependencies:** `npm install -g autoconduck` must result in a fully operational `autoconduck` command without requiring Python or pip on the host system.
- **Sub-5ms Fast-Path Latency:** Fast-Path decision overhead must remain under 5 ms.
- **Zero Hard Failures:** DAG planner or schema parsing errors must never throw an API error to the client; they must degrade gracefully to Fast-Path execution.
- **Safety & Reversibility:** Every agent config modification must create a verbatim backup in `~/.autoconduck/backups/`. Re-running setup must update only AutoConduck-owned config blocks, and `autoconduck uninstall` must fully restore original state.
- **Streaming Disconnect Policy:** Upstream provider requests must be cancelled instantly when a client disconnects to prevent wasted token billing.
- **No Placeholder Code:** Every module, regex rule, retry loop, and cost calculation must be fully implemented and production-ready.