<div align="center">

# AutoConduck 0.3.5

**Local, zero-overhead SLM model router & dynamic LangGraph task orchestrator for coding assistants.**

[![Python](https://img.shields.io/badge/python-3.11+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://python.org)
[![Fast Path Latency](https://img.shields.io/badge/turn--guard-%3C2ms-brightgreen.svg?style=flat)](https://github.com)
[![SLM Engine](https://img.shields.io/badge/SLM-Qwen%202.5%20Coder%200.5B%20(ONNX%2FGGUF)-purple.svg?style=flat)](https://github.com)
[![LangGraph](https://img.shields.io/badge/orchestrator-LangGraph%20Dynamic%20DAG-blue.svg?style=flat)](https://github.com/langchain-ai/langgraph)
[![LiteLLM](https://img.shields.io/badge/proxy-LiteLLM-orange.svg?style=flat)](https://github.com/BerriAI/litellm)
[![FastAPI](https://img.shields.io/badge/server-FastAPI-009688.svg?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat)](LICENSE)

[Why AutoConduck?](#why-autoconduck) • [Architecture](#architecture) • [Quick Start](#quick-start) • [Supported Agents](#supported-agents) • [How It Works](#how-it-works-internally) • [TUI Dashboard](#interactive-tui-dashboard) • [CLI Reference](#cli-command-reference) • [Configuration](#configuration-reference) • [Development](#development)

</div>

---

## Why AutoConduck?

Coding agents (**Claude Code**, **OpenCode**, **Pi**, and **Oh My Pi**) frequently send every prompt, from a single-line typo fix, git status check, or docstring lookup to a 20-file architecture migration, to a single expensive frontier model. This incurs massive token spend on routine turns while bottlenecking large multi-step changes without structured task decomposition.

**AutoConduck 0.3.5** transforms local model routing into an autonomous, SLM-driven execution engine:

- **Turn Guard (0ms / <2ms):** Synchronous, regex-only classifier that detects active tool loops and dispatches them directly to the active model tier without replanning overhead. Stagnation is detected only on 3+ identical consecutive calls or 2+ consecutive errors.
- **Embedded SLM Task Architect (Qwen 2.5 Coder 0.5B ONNX / GGUF):** Local small language model generates validated Pydantic task plans in <100ms, specifying exact DAG topology, subtask constraints, and SLA requirements.
- **"Fit-Gate Then Cheapest" 4D Capability Selection:** Filters models against a 4-dimensional capability vector (`reasoning`, `tool_reliability`, `code_quality`, `latency_class`) weighted per task type, then picks the absolute cheapest qualifying model.
- **Confidence-Tightened Capability Floor:** Low SLM plan confidence dynamically raises the capability floor ($\min(\text{base} + 0.15 \times (1 - \text{conf}), 0.60)$) to ensure difficult prompts land on capable models.
- **Dynamic DAG LangGraph Factory:** Compiles tailored runtime StateGraphs for complex multi-subtask workflows with parallel fan-out execution and unified markdown handoff synthesis.
- **LanceDB Knowledge & RAG Subsystem:** Embedded vector store with zero-cost 16-dimensional term-hash embeddings retrieves relevant codebase snippets without external API dependencies.
- **Session Lifecycle & Context Guard:** Preserves immutable prompt-caching prefixes (turns 0 & 1) across 40+ turns and applies structural compaction at the 80% context window ceiling.
- **Real-Time Reasoning SSE Streamer:** Streams live cognitive deliberations directly to client coding agents using OpenAI `reasoning_content` and Anthropic `thinking_delta` protocols.

---

## Architecture

```text
Agent Request (Claude Code / OpenCode / Pi / OMP)
                          │
                          ▼
            FastAPI Server (LiteLLM Proxy)
                          │
             Interceptor for Pseudo-Models:
       [autoconduck | autoconduck-budget | autoconduck-expensive]
                          │
                          ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                 Turn Guard (0ms / <2ms)                     │
    │  - Synchronous regex turn classifier                        │
    │  - Healthy tool loop bypass -> DIRECT_ACTIVE_TIER           │
    │  - Stagnation detector (3+ identical calls or 2+ errors)     │
    └─────────────────────────────┬───────────────────────────────┘
                                  │
                   Is Task a New Turn / Escalation?
                                  │
                 ┌────────────────┴────────────────┐
                 ▼ (Tool Loop / Simple)            ▼ (New Turn / Complex)
      ┌──────────────────────┐        ┌─────────────────────────────┐
      │  ModelPool.select_   │        │     SLM Task Architect      │
      │       by_sla()       │        │ (Qwen 2.5 Coder 0.5B ONNX)  │
      │   4-Dim Capability   │        │  ExecutionPlan Synthesis    │
      │  Fit-Gate & Cheapest │        └──────────────┬──────────────┘
      └──────────┬───────────┘                       │
                 │                                   ▼
                 │                    ┌─────────────────────────────┐
                 │                    │     Dynamic LangGraph DAG   │
                 │                    │  ┌───────────────────────┐  │
                 │                    │  │ RAG Node (LanceDB)    │  │
                 │                    │  └───────────┬───────────┘  │
                 │                    │              ▼              │
                 │                    │  ┌───────────────────────┐  │
                 │                    │  │ Parallel Subtasks     │  │
                 │                    │  │ (Annotated Fan-out)   │  │
                 │                    │  └───────────┬───────────┘  │
                 │                    │              ▼              │
                 │                    │  ┌───────────────────────┐  │
                 │                    │  │ Synthesizer Node      │  │
                 │                    │  └───────────────────────┘  │
                 │                    └──────────────┬──────────────┘
                 │                                   │
                 └─────────────────┬─────────────────┘
                                   ▼
                   Model Response / Reasoning SSE Stream
```

### Core Architecture Pillars

1. **Turn Guard (`server/turn_guard.py`):** Pure synchronous regex classifier executing in <2ms. Distinguishes clean user turns (`SLM_PLAN`), active healthy tool loops (`DIRECT_ACTIVE_TIER`), and loop stagnation (`ESCALATE_SLM`). Healthy multi-file workflows stay direct without replanning churn.
2. **SLM Task Architect (`routing/slm_planner.py`):** Local ONNX/GGUF model generating typed Pydantic `ExecutionPlan` structures with strict 100ms circuit breakers and deterministic fallbacks.
3. **Capability Vector Model Selection (`routing/model_pool.py`):** Multi-dimensional capability scoring (`reasoning`, `tool_reliability`, `code_quality`, `latency_class`) weighted across 10 task types. Models are fit-gated and sorted by absolute cost ascending.
4. **Dynamic Graph Factory (`orchestrator/dynamic_factory.py`):** Compiles per-turn transient LangGraph execution topologies with parallel worker fan-outs and typed reducers.
5. **Session Guard (`orchestrator/session_guard.py`):** Enforces byte-identical prompt prefix immutability across turns for upstream provider cache hits, and compacts non-structural message history at 80% context capacity.
6. **Knowledge Vector Store (`knowledge/vector_store.py`):** Embedded LanceDB vector index using deterministic 16-dimensional term-hash embeddings for zero-overhead local code symbol retrieval.

---

## Quick Start

### Installation

Install globally via **npm** or directly with **pip**:

```bash
# Global installation via npm
npm install -g autoconduck

# Or Python installation via pip
pip install -r requirements.txt
pip install -e .
```

### Launching

```bash
# 1. Interactive onboarding & TUI dashboard
autoconduck

# 2. Start headless daemon in background
autoconduck start --headless --daemon

# 3. Stop background daemon
autoconduck stop

# 4. Direct agent launch shortcuts
autoconduck start --claude
autoconduck start --opencode
autoconduck start --pi
```

The server listens on `http://127.0.0.1:11434/v1` by default.

---

## Supported Agents

AutoConduck provides automated configuration, shims, and lifecycle hooks for **Claude Code**, **OpenCode**, **Pi**, and **Oh My Pi (OMP)**:

```bash
# Automatically configure and install launcher shims for all agents:
autoconduck install all

# Or install specific agents:
autoconduck install claude opencode pi omp
```

Agent configuration edits are bounded between `# BEGIN AUTOCONDUCK` and `# END AUTOCONDUCK` markers, with automated backups saved to `~/.autoconduck/backups/<agent>/<timestamp>.bak` (retaining the last 5 backups). Running `autoconduck reset` or `autoconduck uninstall` cleanly restores all original configuration files.

### Claude Code

AutoConduck provides an Anthropic-compatible `/v1/messages` translation shim and configures `settings.json`:

```bash
# Launch directly through AutoConduck
autoconduck start --claude

# Or manually configure Claude Code environment:
export ANTHROPIC_BASE_URL="http://127.0.0.1:11434"
export ANTHROPIC_AUTH_TOKEN="autoconduck-local"
export ANTHROPIC_MODEL="autoconduck"
claude
```

### OpenCode

AutoConduck configures `opencode.json` with structured OpenAI-compatible provider endpoints:

```json
{
  "provider": {
    "autoconduck": {
      "type": "openai",
      "baseURL": "http://127.0.0.1:11434/v1"
    }
  },
  "model": "autoconduck/autoconduck"
}
```

```bash
# Launch directly through AutoConduck
autoconduck start --opencode
```

### Pi Coding Agent

AutoConduck installs a dedicated TypeScript extension `<pi_dir>/extensions/autoconduck.ts` using `pi.registerProvider()` and sets `defaultProvider: "autoconduck"` in `settings.json`:

```bash
# Launch directly through AutoConduck
autoconduck start --pi
```

### Oh My Pi (OMP)

AutoConduck supports Oh My Pi through dedicated link commands and config patching (`~/.omp/agent/models.yml` and `config.yml`):

```bash
# Link Oh My Pi to AutoConduck
autoconduck omp link

# Unlink Oh My Pi
autoconduck omp unlink
```

---

## The Three Pseudo-Models

AutoConduck presents three virtual models to coding agents:

| Pseudo-Model | Selection Behavior | Best For |
| :--- | :--- | :--- |
| **`autoconduck`** | Standard SLA capability fit-gate, selects the **cheapest** qualifying model | Everyday software engineering & mixed workflows |
| **`autoconduck-budget`** | Standard SLA capability fit-gate, selects the **cheapest** qualifying model | Repetitive tasks, small edits, and high-frequency runs |
| **`autoconduck-expensive`** | Standard SLA capability fit-gate, selects the **most expensive / capable** qualifying model | Complex architectural refactoring, deep reasoning, greenfield design |

---

## How It Works Internally

### 1. Turn Guard (`server/turn_guard.py`)

Turn Guard evaluates incoming messages synchronously in <2ms:
- **`TurnAction.SLM_PLAN`**: Clean user turn without prior tool execution $\rightarrow$ invokes the embedded SLM planner.
- **`TurnAction.DIRECT_ACTIVE_TIER`**: Healthy active tool loop $\rightarrow$ bypasses SLM replanning and routes directly to the active model tier. Healthy loops touching multiple files or turns are never interrupted.
- **`TurnAction.ESCALATE_SLM`**: Genuine stagnation detected (3+ identical consecutive tool calls or 2+ consecutive tool execution errors) $\rightarrow$ triggers SLM diagnostic re-planning.

### 2. SLM Task Planning (`routing/slm_planner.py`)

The local Small Language Model (defaulting to Qwen 2.5 Coder 0.5B ONNX) generates an `ExecutionPlan`:
- **`route`**: `"fast_direct"` (single turn / simple tool loop) or `"dynamic_dag"` (multi-step workflow).
- **`task_type`**: One of 10 types (`chat`, `explain`, `recon`, `single_edit`, `multi_edit`, `debug`, `refactor`, `full_workflow`, `git_ops`, `routine`).
- **`confidence`**: Plan confidence score between `0.0` and `1.0`.
- **`suggested_sla`**: `CapabilitySLA` containing context limits, tool requirements, reasoning requirements, and capability thresholds.
- **`subtasks`**: Structured `SubTaskSpec` items declaring task scope, roles, constraints, and dependencies.

### 3. "Fit-Gate Then Cheapest" Model Selection (`routing/model_pool.py`)

AutoConduck matches candidate models using a multi-dimensional capability vector:

$$\text{Dimensions} = (\text{reasoning},\; \text{tool\_reliability},\; \text{code\_quality},\; \text{latency\_class})$$

1. **Task-Specific Weighting:** Dimension weights are assigned based on `plan.task_type` (`TASK_TYPE_WEIGHTS`).
2. **Capability Fit Scoring:** Computes `capability_fit(vector, weights)` as:

$$\text{fit} = \min_{d \in \text{dominant}} (\text{vector}[d]) + 0.1 \times \sum_{d} (w_d \times \text{vector}[d])$$

3. **Confidence Floor Tightening:** Lower SLM plan confidence raises the capability floor:

$$\text{floor} = \min(\text{base} + 0.15 \times (1 - \text{confidence}),\; 0.60)$$

4. **Hard Filters:** Filters candidates by active status, tool support, reasoning support, minimum context window, and capability floor.
5. **Opt-In Price Cap:** `CapabilitySLA.max_price_usd_per_mtok` (configured via `selection.path_price_cap_usd_per_mtok`, disabled by default `{}`) sets an optional price ceiling in USD per 1M tokens. If the price cap empties the pool, AutoConduck falls back to the cheapest qualifying model with `fallback_reason = "price_cap_emptied_pool"`.
6. **Cheapest Selection:** Qualifying models are sorted by absolute cost ($P = \text{cost}_{\text{input}} + 0.5 \times \text{cost}_{\text{output}}$) ascending, picking the cheapest model (or most expensive if `autoconduck-expensive`).

### 4. Dynamic LangGraph Pipeline (`orchestrator/`)

When `plan.route == "dynamic_dag"`, `dynamic_factory.py` compiles a runtime `StateGraph`:
- **RAG Node:** Ingests semantic code snippets from LanceDB when `plan.needs_rag == True`.
- **Parallel Subtasks:** Independent subtask nodes execute concurrently with typed reducers (`Annotated[dict, _merge_dict]`).
- **Synthesizer Node:** Combines findings, context, and code diffs into a unified markdown handoff response.

---

## Interactive TUI Dashboard

AutoConduck includes an interactive terminal UI built with Textual:

```bash
autoconduck
```

### Main Navigation Hub

From the main menu, navigate directly to all major views:

- **Live Routing Stats (`d`):** Real-time routing decisions, latency histograms, token volume, and decision drill-down.
- **Model Catalog (`m`):** Browse curated model presets and token pricing.
- **Configure Integrations (`c`):** Set up providers, API keys, and custom endpoints.
- **Edit Models (`e`):** Customize active models and capability overrides.
- **Check for Updates (`u`):** Check latest version and upgrade in-place.
- **Settings (`s`):** Interactive editor for thresholds, ports, and execution parameters.
- **Launch Agent (`a`):** Pick and launch a configured coding agent.

### Keymap Reference

| Key | Action |
| :---: | :--- |
| `Up` / `Down` | Move selection cursor |
| `Enter` / `Space` | Open / toggle selection |
| `Left` / `Esc` / `b` | Back / step float down |
| `Right` / `+` | Advance / step float up |
| `d` | Open detailed routing & latency drill-down |
| `m` | Open model catalog |
| `c` | Configure integrations & API keys |
| `e` | Edit models |
| `u` | Check for updates & upgrade |
| `s` | Open settings screen |
| `a` | Open launch agent picker |
| `p` | Pause / resume proxy routing |
| `?` | Toggle keymap help |
| `/` | Filter list items |
| `Ctrl+C` | Quit current screen / exit AutoConduck (`Ctrl+Q` disabled) |

---

## CLI Command Reference

| Command | Description | Options & Flags |
| :--- | :--- | :--- |
| `autoconduck` | Launches interactive TUI dashboard | `--version` |
| `autoconduck start` | Starts the AutoConduck proxy server | `--headless`, `--daemon`, `--port <int>`, `--host <str>`, `--claude`, `--opencode`, `--pi`, `--new-terminal` |
| `autoconduck stop` | Stops the running proxy server & supervisor | `--port <int>` |
| `autoconduck install [agents...]` | Configures agents & installs launcher shims | Positional: `claude`, `opencode`, `pi`, `omp`, `all` |
| `autoconduck omp link` | Links Oh My Pi configuration to AutoConduck | |
| `autoconduck omp unlink` | Reverts Oh My Pi configuration | |
| `autoconduck edit` | Opens TUI directly on model/provider editor | |
| `autoconduck stats` | Displays routing audit telemetry & cost stats | `--json`, `--days <int>`, `--reset`, `--force` |
| `autoconduck update` | Upgrades AutoConduck to the latest release | `--dry-run` |
| `autoconduck reset` | Reverts all agent configurations & cleans shims | `--force` |
| `autoconduck uninstall` | Reverts configs, removes shims, & uninstalls | `--force` |

---

## Configuration Reference

Configuration is stored in `~/.autoconduck/config.yaml` (or `$AUTOCONDUCK_HOME/config.yaml`), and credentials are kept in `~/.autoconduck/auth.yaml` (`0o600` permissions).

```yaml
host: "127.0.0.1"
port: 11434
log_level: "INFO"
ambiguous_low: 0.60
ambiguous_high: 0.75
escalation_threshold: 0.80

selection:
  # SLM Engine Settings
  slm_model_path: "models/qwen2.5-coder-0.5b-instruct-q4.onnx"
  slm_circuit_breaker_timeout_ms: 100
  
  # Session Guard & RAG
  session_guard_compaction_ratio: 0.80
  rag_max_tokens: 250
  rag_db_path: "~/.autoconduck/rag_db"
  
  # Selection & Capability Floor Tunables
  confidence_floor_k: 0.15
  confidence_floor_max: 0.60
  min_orchestrator_complexity: 0.72
  slow_threshold: 0.75
  deescalation_threshold: 0.40
  
  # Price Caps & Spend Guard (USD per 1M tokens)
  path_price_cap_usd_per_mtok: {}
  spend_guard_enabled: true
  spend_guard_max_usd_per_min: 0.20
  spend_guard_window_s: 300
  
  # Workspace Tools
  executor_enable_tools: true
  executor_max_tool_rounds: 10
  executor_enable_bash: false
```

### Custom Provider Registration

```yaml
custom_models:
  - provider: openrouter
    base_url: https://openrouter.ai/api/v1
    api_key_env: OPENROUTER_API_KEY
    enabled: true

model_list:
  - model_name: openrouter/anthropic/claude-3.5-sonnet
    provider: openrouter
  - model_name: openrouter/deepseek/deepseek-chat
    provider: openrouter
```

---

## Audit & Observability

AutoConduck exposes standard operational and proxy endpoints:

- `GET /healthz`: Health check endpoint (`{"status": "ok"}`).
- `GET /v1/models`: OpenAI-compatible active models list.
- `POST /v1/chat/completions`: OpenAI-compatible chat completion proxy (supports SSE `delta.reasoning_content`).
- `POST /v1/messages`: Anthropic-compatible messages proxy (supports SSE `thinking_delta`).
- `POST /v1/messages/count_tokens`: Anthropic token counting endpoint.
- `GET /stats`: Returns live audit telemetry, decision breakdowns, token volume, latency histograms, and explainability metrics (`candidates_considered`, `binding_constraint`, `spend_cap_engaged`, `fallback_reason`).

```bash
# View live telemetry in terminal
autoconduck stats

# Export JSON audit telemetry
autoconduck stats --json
```

---

## Development

```bash
# Clone repository and set up environment
git clone https://github.com/plum-sorathorn/AutoConduck.git
cd AutoConduck
pip install -r requirements.txt
pip install -e .

# Run test suite
python -m pytest

# Run smoke test
python scripts/end_to_end_smoke.py

# Version bump (syncs pyproject.toml, __init__.py, package.json, docs)
python scripts/bump_version.py --patch

# Update graphify knowledge graph
graphify update .
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
