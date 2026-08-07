# Integrating AutoConduck with OpenCode

## 1. Overview

AutoConduck is a **local OpenAI-compatible router/proxy**. It sits between any
OpenAI-API-compatible agent (OpenCode, Claude Code, Cursor, etc.) and your
real LLM providers and **presents three pseudo-models**:

| Pseudo-model | Meaning |
|---|---|
| `autoconduck` | Balanced — cost/quality trade-off (`transform="x"`) |
| `autoconduck-budget` | Cheaper bias (`transform="x*0.6"`) |
| `autoconduck-expensive` | Quality bias (`transform="min(1,x*1.4+0.1)"`) |

The proxy classifies each request, optionally runs a plan/worker
orchestration, transforms the score through the pseudo-model's function,
estimates tokens, and selects the cheapest real model that satisfies the
required quality threshold via `pricing.select()`. Non-pseudo model ids are
passed straight through to LiteLLM.

Default listen address is **`127.0.0.1:11434`** — the same port Ollama uses so
most tools already know it. Override with `--port` or `AUTOCONDUCK_PORT`.

> Evidence: `autoconduck/main.py:14` `DEFAULT_PORT = 11434`,
> `autoconduck/config.py:10` `DEFAULT_PORT = 11434`,
> `autoconduck/proxy.py:88` `DEFAULT_PORT = 11434`,
> `autoconduck/config.py:46-52` pseudo-model definitions.

---

## 2. Prerequisites

- **Python 3.11+** (`pyproject.toml:6` `requires-python = ">=3.11"`).
- Dependencies are declared in `pyproject.toml:7-16`:

  ```
  fastapi>=0.110
  uvicorn>=0.29
  httpx>=0.27
  pydantic>=2.6
  pyyaml>=6.0
  tiktoken>=0.6
  textual>=0.70
  litellm>=1.40
  ```

  There is no committed `autoconduck/requirements.txt`; install from
  `pyproject.toml` directly:

  ```bash
  pip install -e .                # editable install (recommended)
  # or
  pip install fastapi uvicorn httpx pydantic pyyaml tiktoken textual litellm
  ```

  `pytest` is **not** required to run the proxy.

---

## 3. Run the Proxy

### Entrypoint

`pyproject.toml:17-18` declares the console script:

```toml
[project.scripts]
autoconduck = "autoconduck.main:main"
```

After `pip install -e .` the command `autoconduck` is available. Without
installing, use `python -m autoconduck.main` — both are equivalent.

### Headless (server only)

```bash
# installed
autoconduck start --headless
autoconduck start --headless --port 11434
autoconduck start --headless --port 8080

# without install
python -m autoconduck.main start --headless
python -m autoconduck.main start --headless --port 11434
```

### TUI (onboarding + dashboard)

```bash
autoconduck start                 # launches Textual TUI, which spawns the proxy
python -m autoconduck.main        # same — no subcommand defaults to TUI
autoconduck edit                  # re-open model selection
```

On first run with no `~/.autoconduck/config.yaml` the TUI walks you through
model selection. If `textual` is not installed the proxy falls back to
headless automatically (`autoconduck/main.py:75-77,160-164`).

### Port conflict handling

If the requested port is busy, AutoConduck scans the next 11 ports and prints
`[autoconduck] port 11434 in use, using 11435` (`autoconduck/main.py:17-25,
62-65`).

### Environment overrides

| Env var | Effect | Source |
|---|---|---|
| `AUTOCONDUCK_PORT` | Overrides `port` in config | `autoconduck/config.py:124-129` |
| `AUTOCONDUCK_CACHE_ENABLED` | `1`/`true`/`yes` enables cache | `autoconduck/config.py:130-132` |
| `AUTOCONDUCK_LOG_LEVEL` | `debug`/`info`/`warning`/`error` | `autoconduck/config.py:133-135` |
| `AUTOCONDUCK_HOME` | Moves `~/.autoconduck/` elsewhere | `autoconduck/config.py:73-74` |

Persistent config lives at `~/.autoconduck/config.yaml`
(`autoconduck/config.py:77-78`), state at `~/.autoconduck/state.json`
(`config.py:81-82`), logs at `~/.autoconduck/logs/routing.jsonl`
(`config.py:85-86`).

---

## 4. OpenCode Integration

### What the adapter does

`autoconduck/agents/opencode.py` defines `OpenCodeAdapter`:

- `config_paths()` returns (`opencode.py:14-15`):

  ```python
  [Path.cwd() / "opencode.json", Path.home() / ".config" / "opencode" / "config.json"]
  ```

  The adapter patches the **first existing file**, otherwise `opencode.json` in
  the current working directory (`opencode.py:23-28`).

- `patch()` sets (`opencode.py:17-22`):

  ```python
  endpoint = f"http://127.0.0.1:{config.port}/v1"
  data["providers"]["autoconduck"]["api_base"] = endpoint
  data["providers"]["autoconduck"]["models"] = [
      "autoconduck", "autoconduck-budget", "autoconduck-expensive"
  ]
  ```

  `api_base` is the key the adapter uses (some OpenCode docs call it
  `baseURL` — use `api_base` to match the adapter; `baseURL` also works in
  newer OpenCode versions).

### Authentication

**The proxy itself has no auth gate.** `autoconduck/proxy.py` never checks the
`Authorization` header — any value (including a dummy) is accepted. Upstream
authentication is handled by **LiteLLM** reading provider env vars (see §5).
For OpenCode you can set a dummy key:

```
sk-autoconduck
```

or any non-empty string. If you prefer, set `OPENAI_API_KEY` to a real key —
the proxy will ignore it and use the per-model `api_key_env` instead.

### Manual config (if you don't run `autoconduck start`)

Edit whichever file exists:

- `./opencode.json` (project-local), or
- `~/.config/opencode/config.json` (global)

Add/merge this snippet (replace `11434` if you changed the port):

```json
{
  "providers": {
    "autoconduck": {
      "api_base": "http://127.0.0.1:11434/v1",
      "models": ["autoconduck", "autoconduck-budget", "autoconduck-expensive"]
    }
  }
}
```

For OpenCode versions that expect `baseURL` instead of `api_base`:

```json
{
  "providers": {
    "autoconduck": {
      "baseURL": "http://127.0.0.1:11434/v1",
      "models": ["autoconduck", "autoconduck-budget", "autoconduck-expensive"]
    }
  }
}
```

Then select one of the three models in OpenCode's model picker, e.g.
`autoconduck` (balanced). Use `autoconduck-budget` for cheap/fast tasks and
`autoconduck-expensive` for maximum quality.

---

## 5. Testing It Works

### List models

```bash
curl http://127.0.0.1:11434/v1/models
# variant with explicit port
curl http://127.0.0.1:$AUTOCONDUCK_PORT/v1/models
```

Expected response (`autoconduck/proxy.py:445-454`):

```json
{
  "object": "list",
  "data": [
    {"id": "autoconduck", "object": "model", "created": 0, "owned_by": "autoconduck"},
    {"id": "autoconduck-budget", "object": "model", "created": 0, "owned_by": "autoconduck"},
    {"id": "autoconduck-expensive", "object": "model", "created": 0, "owned_by": "autoconduck"}
  ]
}
```

Health check:

```bash
curl http://127.0.0.1:11434/healthz
curl http://127.0.0.1:11434/stats
```

### Chat completion (non-streaming)

```bash
curl -X POST http://127.0.0.1:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"autoconduck","messages":[{"role":"user","content":"hello"}]}'
```

### Chat completion (streaming — SSE)

```bash
curl -N -X POST http://127.0.0.1:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"autoconduck","messages":[{"role":"user","content":"hello"}],"stream":true}'
```

Expect `text/event-stream` with `data: {...}` lines ending in
`data: [DONE]` (`autoconduck/proxy.py:532-547,543`).

### Upstream provider credentials

The proxy forwards via **LiteLLM** (`autoconduck/proxy.py:286-293`). LiteLLM
reads standard provider env vars — **not** `AUTOCONDUCK_*` prefixed vars
(except the three overrides above). Configure the ones you need:

| Provider | Env var read by LiteLLM / AutoConduck | Notes |
|---|---|---|
| OpenAI | `OPENAI_API_KEY` | Default `api_key_env` in `ModelEntry` (`config.py:19` `Field(default="OPENAI_API_KEY")`) |
| Anthropic | `ANTHROPIC_API_KEY` | Set `api_key_env: ANTHROPIC_API_KEY` on that `ModelEntry` |
| Azure OpenAI | `AZURE_API_KEY` / `AZURE_API_BASE` | |
| Google / Vertex | `GOOGLE_API_KEY` / `VERTEX_*` | |
| Any LiteLLM provider | Whatever LiteLLM documents for that `model` prefix | `model` id prefix selects provider (e.g. `anthropic/claude-3-5-sonnet`) |

Per-model credential binding is explicit in `~/.autoconduck/config.yaml`:

```yaml
models:
  - id: gpt-4o-mini
    provider: openai
    api_key_env: OPENAI_API_KEY
  - id: anthropic/claude-3-5-haiku-20241022
    provider: anthropic
    api_key_env: ANTHROPIC_API_KEY
```

If a model is enabled but its `api_key_env` is unset, LiteLLM will raise an
upstream error which the proxy surfaces with the upstream status code
(`proxy.py:589-603,1074-1106`).

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `port 11434 in use, using 11435` | Another process (Ollama, previous proxy) holds 11434 | Use the printed port, or `lsof -i :11434` / `netstat -ano` to free it, or `autoconduck start --headless --port 8080` |
| `503 no models configured` | `~/.autoconduck/config.yaml` has empty `models:` | Run `autoconduck edit` or `autoconduck start` (TUI) to add models |
| `401` / `No API key` from upstream (proxied as-is) | Provider env var not set | `echo $OPENAI_API_KEY` — set it: `export OPENAI_API_KEY=sk-...` (or `setx` on Windows) and restart proxy |
| `502 proxy_error` | LiteLLM forwarding failed (network/model not found) | Check `http://127.0.0.1:11434/stats` → `degraded_models`; verify model id exists with provider |
| `503 proxy overloaded, retry later` + `Retry-After: 2` | `max_in_flight` (default 32, `config.py:66`) exceeded; `asyncio.wait_for(sem.acquire(), 5s)` timed out | Wait 2s and retry; raise `max_in_flight` in `~/.autoconduck/config.yaml` if sustained |
| Streaming returns JSON instead of SSE | `stream: false` (default) or client not handling SSE | Add `"stream": true` and read `text/event-stream`; `curl -N` disables buffering |
| `499 client disconnected` / `cancelled` | Client closed connection mid-stream | Normal — proxy does **not** count disconnects as model errors (`proxy.py:904-921,1051-1069`) |
| `invalid JSON` 400 | Malformed request body | Validate JSON — `messages` is required (`proxy.py:495-499`) |
| TUI doesn't start | `textual` not installed | `pip install textual>=0.70` or use `--headless` |
| OpenCode still hits OpenAI directly | `opencode.json` not patched / wrong file | Check which file OpenCode loads (`./opencode.json` wins over `~/.config/opencode/config.json`); verify `api_base`/`baseURL` is `http://127.0.0.1:11434/v1` and `models` lists the three ids |

### Useful diagnostics

```bash
curl http://127.0.0.1:11434/healthz   # version, uptime, port
curl http://127.0.0.1:11434/stats     # EMA, error windows, degraded models
cat ~/.autoconduck/config.yaml       # effective config (never contains raw keys)
cat ~/.autoconduck/logs/routing.jsonl | tail -20
```
