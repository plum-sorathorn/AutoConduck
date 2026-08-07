# AutoConduck

Local zero-overhead model router + task orchestrator for OpenAI-compatible AI coding agents.

## Install

```bash
npm install -g autoconduck
# or
bun add -g autoconduck
```

No Python required — npm wrapper resolves to a platform binary.

## Usage

```bash
autoconduck                 # interactive TUI (onboarding or dashboard)
autoconduck start --headless  # background proxy for CI/systemd
autoconduck edit            # re-open model selection
autoconduck uninstall       # restore agent configs
```

Three pseudo-models appear in agent pickers: `autoconduck`, `autoconduck-budget`, `autoconduck-expensive`.

## Dev (Python)

```bash
pip install -r requirements.txt
pip install -e .
pytest
python -m autoconduck.main --help
uvicorn autoconduck.proxy:app --host 127.0.0.1 --port 11434
```

## Architecture

See `ARCHITECTURE.md` and `CONTROL_DATA_FLOW.md` (CONTROL_DATA_FLOW is authoritative on ownership).
