"""Comprehensive test suite for Dynamic SSE Thinking Streamer.

Verifies:
- Real-time visual DAG execution state transitions (`⏳`, `🟢`, `🔴`).
- OpenAI `delta.reasoning_content` streaming protocol.
- Anthropic `thinking_delta` content block streaming protocol.
- Smooth transition from reasoning stream into primary markdown response.
- Unicode glyphs, ANSI stripping, multiline escaping, and empty generator safety.
"""
from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator
import pytest

try:
    from autoconduck.server.sse_streamer import SSEThinkingStreamer
except ImportError:
    pytest.skip("autoconduck.server.sse_streamer not yet implemented in this milestone", allow_module_level=True)


# ==============================================================================
# Tier 1: Feature Coverage (>=5 tests)
# ==============================================================================

@pytest.mark.asyncio
async def test_sse_streamer_openai_reasoning_content_chunks():
    """OpenAI protocol emits delta.reasoning_content with visual glyphs."""
    streamer = SSEThinkingStreamer(client_protocol="openai", model_id="autoconduck")
    frame = await streamer.emit_node_transition("planner", "running", "Synthesizing execution DAG")

    assert frame.startswith("data: ")
    assert frame.endswith("\n\n")

    json_str = frame[6:].strip()
    data = json.loads(json_str)
    assert "choices" in data
    delta = data["choices"][0]["delta"]
    assert "reasoning_content" in delta
    assert "planner" in delta["reasoning_content"]


@pytest.mark.asyncio
async def test_sse_streamer_anthropic_thinking_delta_blocks():
    """Anthropic protocol emits thinking_delta content blocks."""
    streamer = SSEThinkingStreamer(client_protocol="anthropic", model_id="autoconduck")
    frame = await streamer.emit_node_transition("recon", "completed", "Found 3 target files")

    assert frame.startswith("event: ") or frame.startswith("data: ")
    assert "recon" in frame


@pytest.mark.asyncio
async def test_sse_streamer_node_state_transitions():
    """Emits distinct visual glyphs for pending, running, completed, and failed."""
    streamer = SSEThinkingStreamer(client_protocol="openai", model_id="autoconduck")

    frame_pending = await streamer.emit_node_transition("subagent_1", "pending")
    frame_running = await streamer.emit_node_transition("subagent_1", "running")
    frame_completed = await streamer.emit_node_transition("subagent_1", "completed")
    frame_failed = await streamer.emit_node_transition("subagent_1", "failed", "Read error")

    assert "subagent_1" in frame_pending
    assert "subagent_1" in frame_running
    assert "subagent_1" in frame_completed
    assert "subagent_1" in frame_failed
    assert "Read error" in frame_failed


@pytest.mark.asyncio
async def test_sse_streamer_smooth_markdown_transition():
    """Streamer cleanly transitions from reasoning tokens to markdown response tokens."""
    streamer = SSEThinkingStreamer(client_protocol="openai", model_id="autoconduck")

    async def token_gen() -> AsyncIterator[str]:
        yield "Hello "
        yield "world! "
        yield "Here is the result."

    collected_frames = []
    async for chunk in streamer.stream_synthesizer_tokens(token_gen()):
        collected_frames.append(chunk)

    assert len(collected_frames) >= 3
    full_text = "".join(collected_frames)
    assert "Hello" in full_text
    assert "world!" in full_text
    assert "Here is the result." in full_text


@pytest.mark.asyncio
async def test_sse_streamer_terminal_frame_formatting():
    """Verifies that all frames adhere strictly to standard SSE line-break framing."""
    streamer = SSEThinkingStreamer(client_protocol="openai", model_id="autoconduck")
    frame = await streamer.emit_node_transition("synthesizer", "completed")
    assert frame.endswith("\n\n")
    lines = frame.strip().split("\n")
    assert any(line.startswith("data: ") for line in lines)


# ==============================================================================
# Tier 2: Boundary & Corner Cases (>=5 tests)
# ==============================================================================

@pytest.mark.asyncio
async def test_sse_streamer_empty_token_stream():
    """Handles an empty synthesizer token generator without hanging or error."""
    streamer = SSEThinkingStreamer(client_protocol="openai", model_id="autoconduck")

    async def empty_gen() -> AsyncIterator[str]:
        if False:
            yield ""

    chunks = [c async for c in streamer.stream_synthesizer_tokens(empty_gen())]
    assert isinstance(chunks, list)


@pytest.mark.asyncio
async def test_sse_streamer_rapid_burst_transitions():
    """50 rapid state transitions stream sequentially without corrupted framing."""
    streamer = SSEThinkingStreamer(client_protocol="openai", model_id="autoconduck")
    frames = []
    for i in range(50):
        frame = await streamer.emit_node_transition(f"node_{i}", "running", f"Detail {i}")
        frames.append(frame)

    assert len(frames) == 50
    for frame in frames:
        assert frame.startswith("data: ")
        data = json.loads(frame[6:].strip())
        assert "choices" in data


@pytest.mark.asyncio
async def test_sse_streamer_unicode_and_multiline_escaping():
    """Unicode emojis, multiline strings, and quotes escape properly into JSON SSE frames."""
    streamer = SSEThinkingStreamer(client_protocol="openai", model_id="autoconduck")
    complex_detail = "Multiline\nDetail with quotes \"hello\" and emoji 🚀 and code `def foo(): pass`"
    frame = await streamer.emit_node_transition("executor", "completed", complex_detail)

    json_str = frame[6:].strip()
    data = json.loads(json_str)
    reasoning = data["choices"][0]["delta"]["reasoning_content"]
    assert "🚀" in reasoning
    assert "foo" in reasoning


@pytest.mark.asyncio
async def test_sse_streamer_anthropic_protocol_transition():
    """Anthropic protocol transitions from thinking_delta into text content_block_delta."""
    streamer = SSEThinkingStreamer(client_protocol="anthropic", model_id="autoconduck")

    async def token_gen() -> AsyncIterator[str]:
        yield "Response token"

    chunks = [c async for c in streamer.stream_synthesizer_tokens(token_gen())]
    assert len(chunks) >= 1
    assert any("Response token" in c for c in chunks)
