<div align="center">

# AutoConduck 0.3.2

**Local, zero-overhead SLM model router & dynamic LangGraph task orchestrator for coding assistants.**

[![Python](https://img.shields.io/badge/python-3.11+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://python.org)
[![Fast Path Latency](https://img.shields.io/badge/turn--guard-%3C2ms-brightgreen.svg?style=flat)](https://github.com)
[![SLM Engine](https://img.shields.io/badge/SLM-Qwen%202.5%20Coder%200.5B-purple.svg?style=flat)](https://github.com)
[![LangGraph](https://img.shields.io/badge/orchestrator-LangGraph%20Dynamic%20DAG-blue.svg?style=flat)](https://github.com/langchain-ai/langgraph)
[![LiteLLM](https://img.shields.io/badge/proxy-LiteLLM-orange.svg?style=flat)](https://github.com/BerriAI/litellm)
[![FastAPI](https://img.shields.io/badge/server-FastAPI-009688.svg?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat)](LICENSE)

[Why AutoConduck?](#why-autoconduck) • [Architecture](#architecture) • [Quick Start](#quick-start) • [Supported Agents](#supported-agents) • [How It Works](#how-it-works-internally) • [Budget Tuning](#budget-driven-tuning) • [TUI Dashboard](#interactive-tui-dashboard) • [Configuration](#configuration-reference) • [Development](#development)

</div>

---

## Why AutoConduck?

Coding agents (**Claude Code**, **OpenCode**, and **Pi**) frequently send every prompt—from a single-line typo fix or docstring lookup to a 20-file architecture migration—to a single expensive frontier model. This incurs unnecessary token spend on trivial turns while bottlenecking large multi-step changes.

**AutoConduck 0.3.2** transforms model routing into an intelligent, autonomous SLM-driven dynamic execution engine:

- **Turn Guard (0ms / <2ms):** Routine turns (active tool loops, single-file edits, command outputs) bypass heavy orchestration instantly, maintaining sub-millisecond agent responsiveness with zero LLM overhead.
- **Embedded SLM Task Architect (Qwen 2.5 Coder 0.5B Instruct):** Local GGUF-quantized small language model generates structured, validated task plans in <100ms, determining exact DAG topology, tier assignments, and subtask dependencies.
- **Dynamic DAG LangGraph Factory:** Compiles tailored runtime execution graphs for every complex workflow, executing parallel subtask nodes and deterministic context retrieval.
- **LanceDB Knowledge & RAG Subsystem:** Vector store semantic search automatically retrieves relevant codebase snippets without manual file discovery.
- **Session Lifecycle & Context Guard:** Preserves immutable prompt-caching prefixes across 40+ turns and applies intelligent compaction at the 80% context window ceiling.
- **Real-Time Reasoning SSE Streamer:** Streams live SLM cognitive deliberations directly to client coding agents using OpenAI `reasoning_content` and Anthropic `thinking_delta` protocols.
- **Dynamic Pool-Relative 3-Tier Model Matching:** Tiers models dynamically based on the user's active/selected models across `cheap_fast`, `balanced`, and `frontier_reasoning` quantiles, scaling seamlessly whether the user configures 1, 2, 3, 6, or 20+ models.

---

## Architecture

```text
Agent Request (Claude Code / OpenCode / Pi)
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
    │  - Active tool turn suppression (direct model execution)    │
    │  - Stagnation & error streak detector                       │
    └─────────────────────────────┬───────────────────────────────┘
                                  │
                   Is Task a New Turn / Escalation?
                                  │
                 ┌────────────────┴────────────────┐
                 ▼ (Tool Loop / Simple)            ▼ (New Turn / Complex)
      ┌──────────────────────┐        ┌─────────────────────────────┐
      │  ModelPool.select_   │        │     SLM Task Architect      │
      │       for_tier()     │        │  (Qwen 2.5 Coder 0.5B GGUF) │
      │  (cheap_fast /       │        │  ExecutionPlan Synthesis    │
      │   balanced tier)     │        └──────────────┬──────────────┘
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

1. **Turn Guard (`server/turn_guard.py`):** Pure regex classifier that bypasses tool loops in <2ms, detect stagnation loops, and triggers SLM re-planning only when necessary.
2. **SLM Task Architect (`routing/slm_planner.py`):** Embedded local Qwen 2.5 Coder 0.5B model producing validated Pydantic `ExecutionPlan` JSON structures.
3. **Dynamic Graph Factory (`orchestrator/dynamic_factory.py`):** Dynamic LangGraph compilation compiling parallel subtask fan-outs with SQLite state checkpointing.
4. **Knowledge Vector Store (`knowledge/vector_store.py`):** LanceDB embedded vector database for fast semantic code context retrieval.
5. **Session Guard (`orchestrator/session_guard.py`):** Prompt-cache-friendly prefix immutability and context window ceiling protection.

---

## Quick Start

### Installation

Install globally via **npm** or directly with **pip**:

```bash
# Global installation (recommended for launcher shims)
npm install -g autoconduck

# Or local Python installation
pip install -r requirements.txt
pip install -e .
```

### Launching

```bash
# 1. Interactive onboarding & TUI dashboard
autoconduck

# 2. Start as a background daemon
autoconduck start --headless --daemon

# 3. Stop the background service
autoconduck stop

# 4. Direct agent shortcuts
autoconduck start --claude
autoconduck start --opencode
autoconduck start --pi
```

The server listens on `http://127.0.0.1:11434/v1` by default.

---

## Supported Agents

AutoConduck provides automated configuration and launcher shims for **Claude Code**, **OpenCode**, and **Pi**:

```bash
# Automatically configure and install shims for supported agents:
autoconduck install claude_code opencode pi
```

Agent configuration edits are bounded between `# BEGIN AUTOCONDUCK` and `# END AUTOCONDUCK`, with automated backups saved to `~/.autoconduck/backups/<agent>/<timestamp>.bak`. Running `autoconduck reset` or `autoconduck uninstall` cleanly restores original configs.

### Claude Code

AutoConduck provides an Anthropic-compatible `/v1/messages` translation shim:

```bash
# Launch directly through AutoConduck
autoconduck start --claude

# Or manually point Claude Code to the proxy:
export ANTHROPIC_BASE_URL="http://127.0.0.1:11434"
export ANTHROPIC_AUTH_TOKEN="autoconduck-local"
export ANTHROPIC_MODEL="autoconduck"
claude
```

### OpenCode

AutoConduck connects directly to OpenCode via standard OpenAI-compatible endpoints:

```json
// In opencode.json
{
  "provider": "openai",
  "base_url": "http://127.0.0.1:11434/v1",
  "model": "autoconduck"
}
```

```bash
# Launch directly through AutoConduck
autoconduck start --opencode
```

### Pi Coding Agent

Pi connects via settings or CLI arguments:

```bash
# Launch directly through AutoConduck
autoconduck start --pi

# Or pass the local API base:
pi --api-base http://127.0.0.1:11434/v1 --model autoconduck
```

---

## The Three Pseudo-Models

AutoConduck presents three virtual models to coding agents:

| Pseudo-Model | Target Bias | Escalation Threshold | Best For |
| :--- | :---: | :---: | :--- |
| **`autoconduck`** | Neutral (`0.00`) | Standard (`0.60–0.75`) | Everyday software engineering & mixed workflows |
| **`autoconduck-budget`** | Cost-saving (`-0.20`) | Higher (`× 1.15`) | Repetitive tasks, small edits, cost minimization |
| **`autoconduck-expensive`** | Quality-first (`+0.20`) | Lower (`× 0.85`) | Architecture refactors, deep reasoning, greenfield code |

---

## How It Works Internally

### 1. 10-Factor Complexity Scoring (`routing/complexity.py`)

Prompts are scored deterministically in microseconds using normalized factors:

```text
complexity = min(1.0, sum(w_i * factor_i))
```

- **Length (0.08):** Soft log-scale token estimate.
- **Structural (0.12):** Bullet lists, numbered steps, code fences, Markdown headers.
- **Scope Breadth (0.12):** Mentioned files, paths, and CamelCase identifier count.
- **Code Density (0.05):** Inline backticks, CLI flags, and environment variables.
- **Abstraction Level (0.12):** Ratio of architectural concepts vs concrete syntax.
- **Uncertainty & Diagnostics (0.08):** Debugging and root-cause keywords.
- **Cross-Domain (0.12):** Multi-disciplinary language balance.
- **Task Novelty (0.08):** Greenfield creation vs in-place modification.
- **Imperative Strength (0.15):** Graded action verb intensity.
- **Multi-Step Markers (0.08):** Sequential transition tokens (*then*, *next*, *finally*).

### 2. Closest-Cost Model Matching (`routing/pricing.py`)

Models are mapped to a continuous logarithmic cost space:

```text
scaled_cost(m) = ln(1 + price(m)) / ln(1 + max_price_in_pool)
```

AutoConduck matches the task value directly against candidate models:
- **EMA Realized-Cost Blending:** After 3 turns, observed per-token costs blend with advertised pricing (`alpha = 0.1`).
- **Degraded Provider Exclusion:** Automatically routes around providers with >20% error rates over a 300s sliding window.
- **Spend Guard:** Bounded hourly/minute spend protection prevents runaway loops.

### 3. LangGraph Multi-Phase Pipeline (`orchestrator/`)

For complex requests (Complexity >= 0.75), AutoConduck executes an asynchronous DAG:

1. **Recon (`recon.py`):** Pre-reads up to 5 target files deterministically without LLM cost.
2. **Planner (`planner.py`):** Creates a structured `TaskPlan` and extracts up to 8 concise (<=15 words) `verified_context` bullet points per file.
3. **Parallel Subagents (`subagents.py`):** Evaluates independent dependency waves concurrently via `asyncio.gather`. Subagents only receive verified context and verbatim sibling outputs—no redundant file reads.
4. **Zero-Cost Compactor (`compactor.py`):** Deterministically deduplicates lines, preserves `file:line` citations, and truncates analyst reports to ~1k tokens at zero token cost.
5. **Tool-Loop Executor (`executor_loop.py`):** Executes bounded function-calling tools (`read`, `grep`, `glob`, `list`, `edit`, `write`, and optional `bash`) guarded by a `FileClaimRegistry` to prevent overlapping edits.

---

## Budget-Driven Tuning

`autoconduck tune` is an open-loop calibration engine. It converts a monthly USD or token budget into routing controls for the active model pool:

```bash
# Launch interactive budget tuning UI
autoconduck tune

# Select specific tuning mode
autoconduck tune --mode simple
autoconduck tune --mode advanced
```

### Tuning Mechanics

- **Target Rate:** Computes per-minute target spend from your monthly limit and active working hours.
- **Budget Pressure (p in [0, 1]):** Computed using logarithmic cost bounds.
- **Dynamic Adjustments Under Pressure:**
  - Gamma scaling: Exponent scales as `1 + 2.0p`, curving cost targets steeply toward cheaper models.
  - Pseudo-model biases: Budget bias adjusts to `-0.20 - 0.20p`; expensive bias adjusts to `0.20 - 0.35p`.
  - Phase bands: Orchestrator phase bands shift down proportionally (planner `-0.20p`, subagents `-0.20p`, executor `-0.25p`).
  - Ambiguity zone: Bounds shift to `(0.60 + 0.05p, 0.75 + 0.05p)`.
  - EMA alpha: Adjusts to `0.10 + 0.10p`.
- **Profiles:** Saved to `~/.autoconduck/tune_profile.json` with automatic backup of existing configuration.

---

## Interactive TUI Dashboard

AutoConduck includes an interactive terminal UI built with Textual:

```bash
autoconduck
```

### Main Navigation Hub

From the main menu, navigate directly to all major screens:

- **Live Routing Stats (`d`):** Real-time routing decisions, latency histograms, and cost tracker.
- **Configure Models (`m` / `e`):** Add custom providers, select models, and manage credentials.
- **Tune Budget (`t`):** Configure monthly budget limits, headroom, and ambiguity bands.
- **Settings (`s`):** Configure launch behavior, thresholds, and logging.
- **Launch Agent (`a`):** Pick and launch a configured coding agent (Claude Code, OpenCode, Pi).

### Keymap Reference

| Key | Action |
| :---: | :--- |
| `Up` / `Down` | Move selection cursor |
| `Enter` | Open / toggle selection |
| `d` | Open detailed routing & latency drill-down |
| `m` / `e` | Edit model sources & catalog |
| `t` | Open budget tuning screen |
| `s` | Open settings screen |
| `a` | Open launch agent picker |
| `p` | Pause / resume proxy routing |
| `?` | Toggle keymap help |
| `Ctrl+C` | Quit current screen / exit AutoConduck |

---

## Configuration Reference

Configuration is stored in `~/.autoconduck/config.yaml` (or `$AUTOCONDUCK_HOME/config.yaml`), and credentials are kept securely in `~/.autoconduck/auth.yaml`.

```yaml
selection:
  value_to_cost_gamma: 1.0
  pseudo_bias_budget: -0.20
  pseudo_bias_expensive: 0.20
  pseudo_bias_enabled: true
  ema_min_samples: 3
  ema_alpha: 0.1
  degraded_error_rate: 0.20
  degraded_window_s: 300
  ambiguous_low: 0.60
  ambiguous_high: 0.75
  escalation_threshold: 0.80
  hysteresis_floor: 0.50
  deescalation_threshold: 0.40
  min_orchestrator_complexity: 0.62
  tiebreaker_enabled: false
  enable_fast_path_graph: true
  executor_enable_tools: true
  executor_max_tool_rounds: 10
  executor_enable_bash: false
  expose_value_in_stats: true
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

AutoConduck exposes dedicated operational endpoints:

- `GET /stats`: Returns live audit telemetry, routing breakdown (fast vs slow vs ambiguous), latency histograms, and estimated USD cost savings.
- `GET /healthz`: Liveness and readiness health check endpoint.

```bash
# Inspect routing audit logs and cost savings via CLI
autoconduck stats

# Export JSON metrics
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

# Refresh dynamic model catalog
python scripts/refresh_catalog.py
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
