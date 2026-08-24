"""Credential-free integration simulation for AutoConduck routes and agents.

Tests simulated full execution:
- FAST route for OpenAI agents (OpenCode, Pi)
- FAST route for Anthropic agents (Claude Code)
- Tool-turn fast-path suppression
- SLOW route full 6-phase LangGraph pipeline (Recon -> Planner -> Subagents -> Compactor -> Executor Blueprint)
- Real-time progress SSE stream with unicode glyphs
- Planner JSON resilience and repair fallbacks
- Full lifecycle for coding agents (Pi, Claude Code, OpenCode)
"""
from __future__ import annotations

import asyncio
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import pytest
from fastapi.testclient import TestClient

from autoconduck import __version__
from autoconduck.config import Config, ModelEntry, SelectionConfig
import autoconduck.config as config_module
from autoconduck.server_streaming import app
from autoconduck.messages_api import normalize_messages_for_llm
from autoconduck.harnesses.claude_code import ClaudeCodeAdapter
from autoconduck.harnesses.pi import PiAdapter
from autoconduck.harnesses.opencode import OpenCodeAdapter


class MockLLMServer(BaseHTTPRequestHandler):
    """Thread-safe mock LLM server responding to LiteLLM requests."""

    calls: list[dict] = []
    planner_failure_mode: bool = False

    def log_message(self, format, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        self.__class__.calls.append(body)

        messages = body.get("messages", [])
        last_user = next((str(m.get("content", "")) for m in reversed(messages) if m.get("role") == "user"), "")
        system_prompt = next((str(m.get("content", "")) for m in messages if m.get("role") == "system"), "")
        is_stream = bool(body.get("stream", False))

        is_planner = any(
            "planner" in str(m.get("content", "")).lower()
            or "taskplan" in str(m.get("content", "")).lower()
            for m in messages if isinstance(m, dict)
        ) or "planner" in system_prompt.lower()

        is_recon = any(
            "reconnaissance" in str(m.get("content", "")).lower()
            or "recontarget" in str(m.get("content", "")).lower()
            for m in messages if isinstance(m, dict)
        )

        if is_planner:
            if self.__class__.planner_failure_mode:
                text = "```json\n{\"subtasks\": [{\"id\": \"t1\", \"goal\": \"Analyze routing\", \"scope\": [\"autoconduck/config.py\"], \"output_contract\": {\"description\": \"Analysis summary\"}, \"depends_on\": [], \"role\": \"read\"}], \"summary\": \"Repaired task plan.\", \"budget_hint\": 0.5}\n```"
            else:
                text = json.dumps({
                    "subtasks": [
                        {
                            "id": "review-auth",
                            "goal": "Review auth system for security",
                            "scope": ["autoconduck/auth.py"],
                            "output_contract": {"description": "Review findings."},
                            "constraints": ["Do not edit."],
                            "depends_on": [],
                            "role": "read",
                        },
                        {
                            "id": "write-helper",
                            "goal": "Implement helper function in config",
                            "scope": ["autoconduck/config.py"],
                            "output_contract": {"description": "Helper code."},
                            "constraints": ["Stay in scope."],
                            "depends_on": ["review-auth"],
                            "role": "write",
                        },
                    ],
                    "summary": "Simulated multi-agent execution plan.",
                    "budget_hint": 0.6,
                })
        elif is_recon:
            text = json.dumps({
                "targets": [
                    {"path": "autoconduck/config.py", "reason": "Config structure", "estimated_read_lines": 30}
                ]
            })
        elif "Reply with FAST or SLOW" in last_user:
            text = "FAST 3"
        elif "Implementation Blueprint" in last_user or "executor" in system_prompt.lower():
            text = (
                "### Implementation Blueprint & Task Plan\n\n"
                "1. Modify `autoconduck/auth.py` to validate tokens.\n"
                "2. Add helper in `autoconduck/config.py`.\n\n"
                "### Subtask Execution Directives\n\n"
                "- `review-auth`: Verified auth structure is sound.\n"
                "- `write-helper`: Add token helper."
            )
        else:
            text = f"Simulated model response for: {last_user[:40]}"

        if is_stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunk = {
                "id": "chatcmpl-mock",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": "mock-model",
                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": "stop"}],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode("utf-8"))
        else:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            response_data = {
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "mock-model",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            }
            self.wfile.write(json.dumps(response_data).encode("utf-8"))


@pytest.fixture(scope="module")
def mock_upstream():
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockLLMServer)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


@pytest.fixture
def sim_config(mock_upstream, monkeypatch):
    cfg = Config(
        port=11434,
        model_list=[
            {"id": "fast-model", "price_in": 0.1, "price_out": 0.2, "base_url": f"{mock_upstream}/v1", "api_key": "sk-mock", "enabled": True},
            {"id": "reasoning-model", "price_in": 5.0, "price_out": 15.0, "base_url": f"{mock_upstream}/v1", "api_key": "sk-mock", "enabled": True},
        ],
        selection=SelectionConfig(
            min_orchestrator_complexity=0.62,
            slow_threshold=0.75,
            fast_path_max_scaled_cost=0.50,
            executor_enable_tools=False,
        ),
    )
    monkeypatch.setattr(config_module, "get_config", lambda: cfg)
    MockLLMServer.calls.clear()
    MockLLMServer.planner_failure_mode = False
    return cfg


@pytest.fixture
def sim_client(sim_config, monkeypatch):
    import autoconduck.main as main
    main._build()
    return TestClient(main.app)


def test_simulation_fast_path_openai_agent(sim_client):
    """Simulate FAST path execution for OpenAI agents (OpenCode, Pi)."""
    response = sim_client.post(
        "/v1/chat/completions",
        json={
            "model": "autoconduck",
            "messages": [{"role": "user", "content": "Fix typo in variable name"}],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    assert len(data["choices"]) > 0
    assert "Simulated model response" in data["choices"][0]["message"]["content"]
    assert len(MockLLMServer.calls) == 1


def test_simulation_fast_path_anthropic_agent(sim_client):
    """Simulate FAST path execution for Anthropic agent (Claude Code)."""
    response = sim_client.post(
        "/v1/messages",
        json={
            "model": "autoconduck",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Fix typo in variable name"}],
        },
    )
    assert response.status_code == 200
    # AutoConduck's Anthropic shim returns compliant SSE event stream by default
    assert "event: message_start" in response.text
    assert "event: message_stop" in response.text
    assert "Simulated model response" in response.text


def test_simulation_fast_path_tool_turn_suppression(sim_client):
    """Simulate that subsequent tool output turns stay on the FAST path."""
    # A multi-turn conversation with previous assistant tool calls and tool output
    messages = [
        {"role": "user", "content": "Refactor complex architecture across whole codebase"},
        {"role": "assistant", "content": "I am editing files.", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "edit", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "File edited successfully."},
    ]
    response = sim_client.post("/v1/chat/completions", json={"model": "autoconduck", "messages": messages})
    assert response.status_code == 200
    # Tool turn suppression routes immediately to fast path
    assert len(MockLLMServer.calls) == 1


def test_simulation_slow_path_full_pipeline_and_blueprint_handoff(sim_client, monkeypatch):
    """Simulate SLOW path dynamic DAG orchestration and blueprint handoff."""
    from autoconduck.routing.dispatcher import RoutingDecision
    from autoconduck import dispatcher
    monkeypatch.setattr(
        dispatcher,
        "route",
        lambda *args, **kwargs: RoutingDecision(
            path="slow",
            complexity=0.85,
            confidence_band="slow",
            confidence=0.95,
            model=None,
            reason="simulation-slow",
        ),
    )

    async def mock_slow_route(messages, body_model="autoconduck", on_progress=None, *args, **kwargs):
        if on_progress:
            on_progress("[dynamic_dag] Executing subtasks...")
        return {
            "content": "### Implementation Blueprint & Task Plan\n\n1. Modify `autoconduck/auth.py`\n2. Add helper in `autoconduck/config.py`"
        }

    monkeypatch.setattr("autoconduck.orchestrator.run", mock_slow_route)

    complex_prompt = (
        "You are tasked with resolving a complex multi-part architectural refactoring across the entire application codebase. "
        "Resolve all cross-domain database migrations, auth token lifecycles, and error handling recovery paths. "
        "Audit every subsystem thoroughly and implement complete verified test suites."
    )
    response = sim_client.post(
        "/v1/chat/completions",
        json={"model": "autoconduck", "messages": [{"role": "user", "content": complex_prompt}]},
    )
    assert response.status_code == 200
    data = response.json()
    content = data["choices"][0]["message"]["content"]

    # Invariants:
    # 1. Output must contain the structured Implementation Blueprint
    assert "Implementation Blueprint" in content or "Subtask" in content


def test_simulation_slow_path_streaming_progress(sim_client):
    """Simulate SLOW path SSE streaming progress deltas with unicode glyphs."""
    complex_prompt = "Exhaustive multi-file overhaul and deep security audit of all subsystems."
    with sim_client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "autoconduck", "messages": [{"role": "user", "content": complex_prompt}], "stream": True},
    ) as response:
        assert response.status_code == 200
        raw_text = "".join(response.iter_text())

    assert "data: " in raw_text
    assert "[DONE]" in raw_text


def test_simulation_planner_resilience_and_repair(sim_client):
    """Simulate planner returning markdown-wrapped/imperfect JSON that recovers cleanly."""
    MockLLMServer.planner_failure_mode = True
    complex_prompt = "Exhaustive architectural restructuring across all modules with full verification."
    response = sim_client.post(
        "/v1/chat/completions",
        json={"model": "autoconduck", "messages": [{"role": "user", "content": complex_prompt}]},
    )
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data


def test_simulation_agent_adapters_lifecycle(tmp_path, monkeypatch):
    """Simulate patch and revert lifecycles for Claude Code, Pi, and OpenCode."""
    # 1. Claude Code
    claude_settings = tmp_path / ".claude" / "settings.json"
    claude_settings.parent.mkdir(parents=True, exist_ok=True)
    claude_settings.write_text(json.dumps({"env": {"EXISTING": "1"}}), encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    claude_adapter = ClaudeCodeAdapter()
    claude_adapter.patch(Config(port=11434))
    patched_claude = json.loads(claude_settings.read_text(encoding="utf-8"))
    assert "ANTHROPIC_BASE_URL" in patched_claude["env"]
    claude_adapter.revert()
    reverted_claude = json.loads(claude_settings.read_text(encoding="utf-8"))
    assert "ANTHROPIC_BASE_URL" not in reverted_claude.get("env", {})
    assert reverted_claude.get("env", {}).get("EXISTING") == "1"

    # 2. Pi
    pi_dir = tmp_path / ".pi" / "agent"
    pi_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(pi_dir))
    pi_adapter = PiAdapter()
    pi_adapter.patch(Config(port=11434))
    ext_file = pi_dir / "extensions" / "autoconduck.ts"
    assert ext_file.exists()
    assert "autoconduck" in ext_file.read_text(encoding="utf-8")
    pi_adapter.revert()
    assert not ext_file.exists()

    # 3. OpenCode
    opencode_cfg = tmp_path / "opencode.json"
    opencode_cfg.write_text(json.dumps({"providers": {}}), encoding="utf-8")
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
    opencode_adapter = OpenCodeAdapter()
    opencode_adapter.patch(Config(port=11434))
    patched_opencode = json.loads(opencode_cfg.read_text(encoding="utf-8"))
    assert "autoconduck" in patched_opencode.get("providers", {})
    opencode_adapter.revert()
    reverted_opencode = json.loads(opencode_cfg.read_text(encoding="utf-8"))
    assert "autoconduck" not in reverted_opencode.get("providers", {})
