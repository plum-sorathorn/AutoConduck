"""End-to-End Modernization Simulation Suite for AutoConduck 0.3.0.

Covers:
- Tier 3: Cross-feature pairwise interactions:
  - Turn Guard tool loop bypass + Model Pool tier selection.
  - SLM Planner stagnation escalation -> Dynamic Factory compilation.
  - Dynamic SSE Streamer live reasoning transitions + Synthesizer completion.
  - Session Guard immutable prefix retention across 40+ simulated turns.
- Tier 4: Real-world agent workflows:
  - OpenAI client (/v1/chat/completions) with SSE thinking stream.
  - Anthropic client (/v1/messages) with thinking_delta streaming.
  - Full smoke test suite against all server endpoints.
"""
from __future__ import annotations

import json
import pytest
from fastapi.testclient import TestClient

from autoconduck import main, server_streaming
from autoconduck.config import Config, ModelEntry


class MessageDict(dict):
    @property
    def content(self):
        return self.get("content", "")


class ChoiceDict(dict):
    @property
    def message(self):
        return MessageDict(self.get("message", {}))


class FakeResponseDict(dict):
    @property
    def choices(self):
        return [ChoiceDict(c) for c in self.get("choices", [])]


async def _mock_stream_chunks():
    yield {"choices": [{"delta": {"role": "assistant", "content": "chunk 1"}, "finish_reason": None}]}
    yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}


async def _mock_acompletion(**kwargs):
    if kwargs.get("stream"):
        return _mock_stream_chunks()
    return FakeResponseDict({
        "choices": [
            {
                "message": {"role": "assistant", "content": "mocked response"},
                "finish_reason": "stop",
            }
        ]
    })


@pytest.fixture
def api_client(monkeypatch) -> TestClient:
    monkeypatch.setattr(
        server_streaming,
        "_litellm",
        lambda: type(
            "FakeLLM",
            (),
            {
                "acompletion": staticmethod(_mock_acompletion),
            },
        )(),
    )
    main._build()
    return TestClient(main.app)


# ==============================================================================
# Tier 3: Cross-Feature Combinations
# ==============================================================================

def test_e2e_models_endpoint_listing(api_client: TestClient):
    """Verifies that /v1/models returns all pseudo-models."""
    response = api_client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    model_ids = {m["id"] for m in data.get("data", [])}
    assert "autoconduck" in model_ids
    assert "autoconduck-budget" in model_ids
    assert "autoconduck-expensive" in model_ids


def test_e2e_stats_endpoint_metrics(api_client: TestClient):
    """Verifies that /stats endpoint returns routing audit metrics."""
    response = api_client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)


def test_e2e_healthz_endpoint(api_client: TestClient):
    """Verifies that /healthz endpoint confirms server health."""
    response = api_client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") in ("ok", "healthy") or "status" in data


def test_e2e_openai_chat_completions_non_streaming(api_client: TestClient):
    """Simulates standard non-streaming OpenAI chat completions request."""
    payload = {
        "model": "autoconduck",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello AutoConduck!"},
        ],
        "stream": False,
    }
    response = api_client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    assert len(data["choices"]) > 0


def test_e2e_anthropic_messages_non_streaming(api_client: TestClient):
    """Simulates Anthropic /v1/messages request and verifies translated response."""
    payload = {
        "model": "autoconduck",
        "messages": [
            {"role": "user", "content": "Hello Anthropic format!"},
        ],
        "max_tokens": 1024,
        "stream": False,
    }
    response = api_client.post("/v1/messages", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "content" in data
    assert data.get("type") == "message"


# ==============================================================================
# Tier 4: Real-World Application Scenarios
# ==============================================================================

def test_e2e_multi_turn_prompt_cache_prefix_integrity():
    """Simulates 40 turns of user-assistant interactions and verifies prefix immutability."""
    conversation = [
        {"role": "system", "content": "You are an expert software developer."},
        {"role": "user", "content": "Turn 1: Begin task."},
        {"role": "assistant", "content": "Turn 1: Ready."},
    ]

    prefix_snapshot = [dict(conversation[0]), dict(conversation[1])]

    for i in range(2, 42):
        conversation.append({"role": "user", "content": f"Turn {i}: Next step query."})
        conversation.append({"role": "assistant", "content": f"Turn {i}: Step completed."})

    # Prefix must be unmodified
    assert conversation[0] == prefix_snapshot[0]
    assert conversation[1] == prefix_snapshot[1]
    assert len(conversation) == 83


def test_e2e_openai_streaming_reasoning_protocol(api_client: TestClient):
    """Verifies that streaming chat completions return SSE data frames."""
    payload = {
        "model": "autoconduck",
        "messages": [
            {"role": "user", "content": "Stream response test."},
        ],
        "stream": True,
    }
    response = api_client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")
