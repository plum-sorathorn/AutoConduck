<div align="center">

# AutoConduck

**Local, zero-overhead model router & multi-agent task orchestrator for coding assistants.**

[![Python](https://img.shields.io/badge/python-3.11+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://python.org)
[![Fast Path Latency](https://img.shields.io/badge/fast--path-%3C5ms-brightgreen.svg?style=flat)](https://github.com)
[![LangGraph](https://img.shields.io/badge/orchestrator-LangGraph-blue.svg?style=flat)](https://github.com/langchain-ai/langgraph)
[![LiteLLM](https://img.shields.io/badge/proxy-LiteLLM-orange.svg?style=flat)](https://github.com/BerriAI/litellm)
[![FastAPI](https://img.shields.io/badge/server-FastAPI-009688.svg?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat)](LICENSE)

[Quick Start](#quick-start) • [Architecture](#architecture) • [Agent Setup](#agent-integrations) • [Configuration](#configuration-reference) • [TUI Dashboard](#interactive-tui-dashboard)

</div>

---

## Why AutoConduck?

Coding agents (**Claude Code**, **OpenCode**, **Pi**, **Aider**, **Cursor**, **Continue**, **Kilocode**) often send every request—from a 3-word typo fix to a massive 20-file architecture migration—to a single expensive frontier model. This wastes significant token budget on trivial turns while bottlenecking large multi-step changes.

**AutoConduck** acts as a lightweight, zero-overhead local proxy between your coding agent and LLM providers:

- **Sub-5ms Fast Path:** Simple turns (lookups, docstrings, single-line edits, tool loops) execute through a compiled zero-reflection micro-DAG without I/O or LLM latency.
- **Dynamic Closest-Cost Model Selection:** Instead of static tiers, AutoConduck evaluates prompt complexity on the fly and picks the closest model on a logarithmic cost continuum with real-time Exponential Moving Average (EMA) tracking.
- **LangGraph Multi-Agent Orchestrator:** Complex multi-file prompts automatically escalate to an asynchronous pipeline featuring deterministic recon, file-context distillation, parallel subagent analysts, zero-LLM compaction, and a grounded executor tool loop.
- **Drop-in Compatibility:** Exposes standard OpenAI `/v1/chat/completions` and `/v1/models` endpoints alongside an Anthropic `/v1/messages` translation shim.

---

## Architecture

```
Agent Request (Claude Code / OpenCode / Pi / Aider / Cursor)
                          │
                          ▼
            FastAPI Server (LiteLLM Proxy)
                          │
             Interceptor for Pseudo-Models:
       [autoconduck | autoconduck-budget | autoconduck-expensive]
                          │
                          ▼
    ┌─────────────────────────────────────────────────────────────┐
    │              Fast Path Micro-DAG (<0.1 ms)                  │
    │  input_sanitize ──► route_match ──► evaluate_score          │
    │         (Aurelio Semantic Router + 10-Factor Complexity)    │
    └─────────────────────────────┬───────────────────────────────┘
                                  │
                   Is Task Complex (≥ 0.75)?
                                  │
                 ┌────────────────┴────────────────┐
                 ▼ (No / Simple)                   ▼ (Yes / Multi-File)
      ┌──────────────────────┐        ┌─────────────────────────────┐
      │  pricing.select_     │        │    LangGraph Orchestrator   │
      │       closest()      │        │  ┌───────────────────────┐  │
      │ (Closest-Cost Model) │        │  │ Recon (Files pre-read)│  │
      └──────────┬───────────┘        │  └───────────┬───────────┘  │
                 │                    │              ▼              │
                 │                    │  ┌───────────────────────┐  │
                 │                    │  │ Planner (Context Dist)│  │
                 │                    │  └───────────┬───────────┘  │
                 │                    │              ▼              │
                 │                    │  ┌───────────────────────┐  │
                 │                    │  │ Parallel Subagents    │  │
                 │                    │  └───────────┬───────────┘  │
                 │                    │              ▼              │
                 │                    │  ┌───────────────────────┐  │
                 │                    │  │ Zero-Cost Compactor   │  │
                 │                    │  └───────────┬───────────┘  │
                 │                    │              ▼              │
                 │                    │  ┌───────────────────────┐  │
                 │                    │  │ Tool-Loop Executor    │  │
                 │                    │  └───────────────────────┘  │
                 │                    └──────────────┬──────────────┘
                 │                                   │
                 └─────────────────┬─────────────────┘
                                   ▼
                      Model Response / Stream
```

### Three Core Pillars

1. **LiteLLM Proxy & Server:** Provides unified provider abstraction, streaming translation, token usage tracking, and drop-in endpoints (`/v1/chat/completions`, `/v1/models`, `/v1/messages`).
2. **Deterministic Fast Path (`routing/`):** Evaluates prompt complexity across 10 single-pass regex factors with stack-trace boosts (+0.25), escalation boosts (+0.30), hysteresis clamping (≤0.50), and Aurelio Semantic Router embedding classification.
3. **LangGraph Task DAG (`orchestrator/`):** Manages multi-stage task breakdown with zero inter-phase LLM summarization cost and strict file-claim concurrency protection.

---

## Quick Start

### Installation

Install globally via **npm** or directly with **pip**:

```bash
# Global installation (recommended for agent shims)
npm install -g autoconduck

# Or Python package installation
pip install autoconduck
```

### Launching

```bash
# 1. Interactive onboarding & TUI dashboard
autoconduck

# 2. Start as a background daemon
autoconduck start --headless --daemon

# 3. Stop the background service
autoconduck stop
```

The server listens on `http://127.0.0.1:11434/v1` by default.

---

## The Three Pseudo-Models

AutoConduck presents three virtual models to your coding agents:

| Pseudo-Model | Bias | Escalation Sensitivity | Best For |
| :--- | :--- | :--- | :--- |
| **`autoconduck`** | Neutral (`0.00`) | Balanced default (`0.60–0.75`) | Everyday software engineering & mixed workflows |
| **`autoconduck-budget`** | Cost-saving (`-0.20`) | Higher threshold (`× 1.15`) | Repetitive tasks, small scripts, cost minimization |
| **`autoconduck-expensive`**| Quality-first (`+0.20`) | Lower threshold (`× 0.85`) | Mission-critical refactors, deep reasoning, greenfield code |

---

## Agent Integrations

AutoConduck provides automated configuration and launcher shims for all major coding tools:

```bash
# Automatically configure and wrap your agents:
autoconduck install claude-code opencode pi aider
```

### Manual Configuration

<details>
<summary><b>Claude Code</b></summary>

```bash
# Configure Claude Code to use AutoConduck's Anthropic /v1/messages shim
export ANTHROPIC_BASE_URL="http://127.0.0.1:11434"
export ANTHROPIC_MODEL="autoconduck"
claude
```
</details>

<details>
<summary><b>OpenCode</b></summary>

```json
// In opencode.json
{
  "provider": "openai",
  "base_url": "http://127.0.0.1:11434/v1",
  "model": "autoconduck"
}
```
</details>

<details>
<summary><b>Pi Coding Agent</b></summary>

```bash
pi --api-base http://127.0.0.1:11434/v1 --model autoconduck
```
</details>

<details>
<summary><b>Aider</b></summary>

```bash
aider --openai-api-base http://127.0.0.1:11434/v1 --model openai/autoconduck
```
</details>

<details>
<summary><b>Cursor & Continue.dev</b></summary>

In Cursor or Continue settings:
- **Model Name:** `autoconduck`
- **Base URL:** `http://127.0.0.1:11434/v1`
- **API Key:** `dummy` (or any string)
</details>

---

## How It Works Internally

### 1. 10-Factor Complexity Scoring (`routing/complexity.py`)

Prompts are scored deterministically in microseconds using normalized factors:

$$\text{Complexity} = \min\left(1.0, \sum_{i=1}^{10} w_i \cdot \text{factor}_i\right)$$

- **Length ($0.08$):** Soft log-scale token estimate.
- **Structural ($0.12$):** Bullet lists, numbered steps, code fences, Markdown headers.
- **Scope Breadth ($0.12$):** Mentioned files, paths, and CamelCase identifier count.
- **Code Density ($0.05$):** Inline backticks, CLI flags, and environment variables.
- **Abstraction Level ($0.12$):** Ratio of architectural concepts vs concrete syntax.
- **Uncertainty & Diagnostics ($0.08$):** Debugging and root-cause keywords.
- **Cross-Domain ($0.12$):** Multi-disciplinary language balance.
- **Task Novelty ($0.08$):** Greenfield creation vs in-place modification.
- **Imperative Strength ($0.15$):** Graded action verb intensity.
- **Multi-Step Markers ($0.08$):** Sequential transition tokens (*then*, *next*, *finally*).

### 2. Closest-Cost Model Matching (`routing/pricing.py`)

Models are mapped to a continuous logarithmic cost space:

$$\text{scaled\_cost}(m) = \frac{\ln(1 + \text{price}(m))}{\ln(1 + \max(\text{prices}))}$$

AutoConduck matches the task value directly against candidate models:
- **EMA Realized-Cost Blending:** After 3 turns, observed per-token costs blend with advertised pricing ($\alpha = 0.1$).
- **Degraded Provider Exclusion:** Automatically routes around providers with $>20\%$ error rates over a 300s sliding window.
- **Spend Guard:** Bounded hourly/minute spend protection prevents runaway loops.

### 3. LangGraph Multi-Phase Pipeline (`orchestrator/`)

For complex requests ($\text{Complexity} \ge 0.75$), AutoConduck executes an asynchronous DAG:

1. **Recon (`recon.py`):** Pre-reads up to 5 target files deterministically without LLM cost.
2. **Planner (`planner.py`):** Creates a structured `TaskPlan` and extracts up to 8 concise (≤15 words) `verified_context` bullet points per file.
3. **Parallel Subagents (`subagents.py`):** Evaluates independent dependency waves concurrently via `asyncio.gather`. Subagents only receive verified context and verbatim sibling outputs—no redundant file reads.
4. **Zero-Cost Compactor (`compactor.py`):** Deterministically deduplicates lines, preserves `file:line` citations, and truncates analyst reports to ~1k tokens at zero token cost.
5. **Tool-Loop Executor (`executor_loop.py`):** Executes bounded function-calling tools (`read`, `grep`, `glob`, `list`, `edit`, `write`, and optional `bash`) guarded by a `FileClaimRegistry` to prevent overlapping edits.

---

## Interactive TUI Dashboard

AutoConduck features an interactive Textual terminal UI:

```bash
autoconduck
```

| Key | Action |
| :---: | :--- |
| `j` / `k` | Navigate routing history and model lists |
| `d` | Open detailed routing & latency drill-down |
| `p` | Pause / resume proxy routing |
| `e` | Open interactive model list editor |
| `?` | Toggle keymap help |
| `Ctrl+C` | Quit AutoConduck |

---

## Configuration Reference

AutoConduck stores its configuration at `~/.autoconduck/config.yaml` (or `$AUTOCONDUCK_HOME/config.yaml`) and provider keys at `~/.autoconduck/auth.yaml`.

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
- `GET /healthz`: Standard Kubernetes/liveness readiness probe.

```bash
# Query routing stats via CLI
autoconduck stats --json
```

---

## Development

```bash
# Clone and install development environment
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
