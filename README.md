<div align="center">

# AutoConduck 0.3.4

**Local, zero-overhead SLM model router & dynamic LangGraph task orchestrator for coding assistants.**

[![Python](https://img.shields.io/badge/python-3.11+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://python.org)
[![Fast Path Latency](https://img.shields.io/badge/turn--guard-%3C2ms-brightgreen.svg?style=flat)](https://github.com)
[![SLM Engine](https://img.shields.io/badge/SLM-Qwen%202.5%20Coder%200.5B-purple.svg?style=flat)](https://github.com)
[![LangGraph](https://img.shields.io/badge/orchestrator-LangGraph%20Dynamic%20DAG-blue.svg?style=flat)](https://github.com/langchain-ai/langgraph)
[![LiteLLM](https://img.shields.io/badge/proxy-LiteLLM-orange.svg?style=flat)](https://github.com/BerriAI/litellm)
[![FastAPI](https://img.shields.io/badge/server-FastAPI-009688.svg?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat)](LICENSE)

[Why AutoConduck?](#why-autoconduck) • [Architecture](#architecture) • [Quick Start](#quick-start) • [Supported Agents](#supported-agents) • [How It Works](#how-it-works-internally) • [TUI Dashboard](#interactive-tui-dashboard) • [Configuration](#configuration-reference) • [Development](#development)

</div>

---

## Why AutoConduck?

Coding agents (**Claude Code**, **OpenCode**, and **Pi**) frequently send every prompt—from a single-line typo fix or docstring lookup to a 20-file architecture migration—to a single expensive frontier model. This incurs unnecessary token spend on trivial turns while bottlenecking large multi-step changes.

**AutoConduck 0.3.4** transforms model routing into an intelligent, autonomous SLM-driven dynamic execution engine:

- **Turn Guard (0ms / <2ms):** Routine turns (active tool loops, single-file edits, command outputs) bypass heavy orchestration instantly, maintaining sub-millisecond agent responsiveness with zero LLM overhead.
- **Embedded SLM Task Architect (Qwen 2.5 Coder 0.5B Instruct):** Local GGUF-quantized small language model generates structured, validated task plans in <100ms, determining exact DAG topology, SLA requirements, and subtask dependencies.
- **Dynamic DAG LangGraph Factory:** Compiles tailored runtime execution graphs for every complex workflow, executing parallel subtask nodes and deterministic context retrieval.
- **LanceDB Knowledge & RAG Subsystem:** Vector store semantic search automatically retrieves relevant codebase snippets without manual file discovery.
- **Session Lifecycle & Context Guard:** Preserves immutable prompt-caching prefixes across 40+ turns and applies intelligent compaction at the 80% context window ceiling.
- **Real-Time Reasoning SSE Streamer:** Streams live SLM cognitive deliberations directly to client coding agents using OpenAI `reasoning_content` and Anthropic `thinking_delta` protocols.
- **Dynamic SLA-Based Model Matching:** The SLM assigns each subtask a `CapabilitySLA` (`min_context`, `requires_tools`, `requires_reasoning`, `max_cost`), and the Query Optimizer selects the absolute cheapest active model that meets it.

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
     │       by_sla()       │        │  (Qwen 2.5 Coder 0.5B GGUF) │
     │  (CapabilitySLA /    │        │  ExecutionPlan Synthesis    │
     │   Query Optimizer)   │        └──────────────┬──────────────┘
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

1. **Turn Guard (`server/turn_guard.py`):** Pure regex classifier that bypasses tool loops in <2ms, detects stagnation loops, and triggers SLM re-planning only when necessary; selection uses `CapabilitySLA` requirements with `ModelPool.select_by_sla()`.
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

| Pseudo-Model | Target Bias | Selection Behavior | Best For |
| :--- | :---: | :---: | :--- |
| **`autoconduck`** | Neutral (`0.00`) | Standard SLA-based selection | Everyday software engineering & mixed workflows |
| **`autoconduck-budget`** | Cost-saving (`-0.20`) | Biases SLA-qualified selection toward lower cost | Repetitive tasks, small edits, cost minimization |
| **`autoconduck-expensive`** | Quality-first (`+0.20`) | Biases SLA-qualified selection toward higher capability | Architecture refactors, deep reasoning, greenfield code |

---

## How It Works Internally

### 1. SLM Planning and SLA Selection (`routing/slm_planner.py`)

The embedded SLM generates an `ExecutionPlan` with per-subtask `CapabilitySLA` requirements. `ModelPool.select_by_sla()` and `pricing.select_for_sla()` filter active models by context, tools, reasoning, and cost, then choose the cheapest qualifying model. The fallback complexity signal is the simple word-count heuristic `orchestrator/helpers.py:complexity_of`.

### 2. Cost and Health Controls (`routing/pricing.py`)

Models are mapped to a continuous logarithmic cost space:

```text
scaled_cost(m) = ln(1 + price(m)) / ln(1 + max_price_in_pool)
```

AutoConduck matches the task value directly against candidate models:
- **EMA Realized-Cost Blending:** After 3 turns, observed per-token costs blend with advertised pricing (`alpha = 0.1`).
- **Degraded Provider Exclusion:** Automatically routes around providers with >20% error rates over a 300s sliding window.
- **Spend Guard:** Bounded hourly/minute spend protection prevents runaway loops.

### 3. Dynamic LangGraph Pipeline (`orchestrator/`)

The SLM plan is compiled by `dynamic_factory.py` and `runner.py` into a dynamic LangGraph DAG with an optional LanceDB RAG node, parallel subtask fan-out via `subagents.py`, and a terminal Synthesizer node. Progress and reasoning are emitted to the harness through the SSE stream.

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
- **Check for Updates (`u`):** Check latest version and upgrade in-app.
- **Settings (`s`):** Configure launch behavior, thresholds, and logging.
- **Launch Agent (`a`):** Pick and launch a configured coding agent (Claude Code, OpenCode, Pi).

### Keymap Reference

| Key | Action |
| :---: | :--- |
| `Up` / `Down` | Move selection cursor |
| `Enter` | Open / toggle selection |
| `d` | Open detailed routing & latency drill-down |
| `m` / `e` | Edit model sources & catalog |
| `u` | Check for updates |
| `s` | Open settings screen |
| `a` | Open launch agent picker |
| `p` | Pause / resume proxy routing |
| `?` | Toggle keymap help |
| `/` | Filter dashboard items |
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
