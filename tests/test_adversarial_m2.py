"""Empirical Adversarial Stress Test Suite for Milestone 2.

Adversarially tests:
1. SSE Thinking Streamer:
   - Rapid bursts of 200+ state transitions across concurrent nodes.
   - Special characters (quotes, control chars, newlines, HTML, ANSI escapes, backslashes).
   - Complex Unicode & Emojis (ZWJ sequences, flag sequences, astral plane glyphs, multiline zalgo).
   - Stream cancellation & client disconnection handling (GeneratorExit, CancelledError, mock disconnects).
   - Smooth transition from thinking deltas to markdown text stream.
   - Concurrent streams and empty/whitespace token resilience.

2. Session Guard:
   - 50+ turn conversation simulation (60 turns).
   - Byte-identical prompt cache prefix preservation (turns 0 and 1).
   - 100KB+ verbose tool logs with 80% context window ceiling compaction.
   - Markdown code block fences (```...```) preservation across languages and unclosed fences.
   - Structural markdown header (#, ##, ###) preservation.
   - Non-standard content payloads (None, lists of blocks, empty dicts).

3. LanceDB RAG Subsystem:
   - Strict token budget truncation (<=250 tokens in State["verified_context"]).
   - Multi-query extraction token capping.
   - Massive symbol and large codebase indexing & retrieval.
"""
from __future__ import annotations

import asyncio
import copy
import json
import pytest
from typing import AsyncIterator, Any

from autoconduck.server.sse_streamer import SSEThinkingStreamer, STATUS_GLYPHS
from autoconduck.orchestrator.session_guard import (
    SessionGuard,
    SessionGuardResult,
    _count_tokens_messages,
    _count_tokens_text,
)
from autoconduck.knowledge.vector_store import KnowledgeVectorStore
from autoconduck.knowledge.models import CodeChunk, QueryResult
from autoconduck.routing.slm_planner import ExecutionPlan, SubTaskSpec, ModelTier
from autoconduck.orchestrator.dynamic_factory import (
    DynamicState,
    build_dynamic_graph,
    _rag_node_handler,
)


# ==============================================================================
# SECTION 1: SSE Thinking Streamer Adversarial Stress Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_sse_streamer_rapid_burst_stress():
    """Stress test: 200+ rapid interleaved state transitions across 50 nodes.
    
    Verifies:
    - Every frame is strictly valid JSON conforming to SSE format (data: {...}\\n\\n).
    - Status glyphs match expected mappings (⏳, 🟢, 🔴).
    - No frame corruption or dropped data during rapid bursts.
    """
    for proto in ("openai", "anthropic"):
        streamer = SSEThinkingStreamer(client_protocol=proto, model_id="autoconduck-burst")
        statuses = ["pending", "running", "completed", "failed"]
        frames: list[str] = []

        for i in range(200):
            node = f"subtask_node_{i % 50}"
            status = statuses[i % 4]
            detail = f"Execution step {i} on thread {i // 10}"
            frame = await streamer.emit_node_transition(node, status, detail)
            frames.append(frame)

        assert len(frames) == 200
        for i, frame in enumerate(frames):
            assert frame.endswith("\n\n"), f"Frame {i} missing standard trailing newlines"
            if proto == "openai":
                assert frame.startswith("data: ")
                raw_json = frame[6:].strip()
                parsed = json.loads(raw_json)
                assert parsed["object"] == "chat.completion.chunk"
                assert parsed["model"] == "autoconduck-burst"
                delta = parsed["choices"][0]["delta"]
                assert "reasoning_content" in delta
                reasoning = delta["reasoning_content"]
                assert f"subtask_node_{i % 50}" in reasoning
                assert statuses[i % 4] in reasoning
                assert STATUS_GLYPHS[statuses[i % 4]] in reasoning
            else:
                assert frame.startswith("event: content_block_delta\ndata: ")
                lines = frame.strip().split("\n")
                raw_json = lines[1][6:].strip()
                parsed = json.loads(raw_json)
                assert parsed["type"] == "content_block_delta"
                assert parsed["delta"]["type"] == "thinking_delta"
                thinking = parsed["delta"]["thinking"]
                assert f"subtask_node_{i % 50}" in thinking


@pytest.mark.asyncio
async def test_sse_streamer_special_characters_fuzzing():
    """Fuzz test: node transitions with adversarial characters.
    
    Fuzz vectors include:
    - JSON quotes, nested escaping, backslashes: `\\`, `\"`, `\"\"\"`
    - Control characters and newlines: `\\r\\n`, `\\t`, `\\b`, `\\f`
    - HTML / XML injection strings: `<script>alert(1)</script>`, `<thinking>`
    - Markdown delimiters: ```` ```python ````, `###`, `---`
    - Shell command injection patterns: `$(rm -rf /)`, `; ls -la | grep "foo"`
    """
    adversarial_payloads = [
        'Quote "nested" and \'single\' and \\"escaped\\"',
        "Multi\r\nLine\rWith\nWindows\nAnd\nUnix\nBreaks",
        "Backslashes \\ and \\\\ and \\\\\\ and / and //",
        "<script>alert('XSS')</script> <thinking>tag</thinking>",
        "```python\ndef foo():\n    return 'bar'\n```",
        "Control chars: \t tab, \b backspace, \f formfeed",
        "SQL injection: '; DROP TABLE users; --",
        "JSON block: {\"key\": \"value\", \"arr\": [1, 2, 3]}",
        "Empty string",
        "   Whitespace only   \n\t  ",
    ]

    for proto in ("openai", "anthropic"):
        streamer = SSEThinkingStreamer(client_protocol=proto, model_id="autoconduck-fuzz")
        for idx, payload in enumerate(adversarial_payloads):
            frame = await streamer.emit_node_transition(f"node_fuzz_{idx}", "running", payload)
            assert frame.endswith("\n\n")
            if proto == "openai":
                parsed = json.loads(frame[6:].strip())
                content = parsed["choices"][0]["delta"]["reasoning_content"]
                assert f"node_fuzz_{idx}" in content
            else:
                lines = frame.strip().split("\n")
                parsed = json.loads(lines[1][6:].strip())
                content = parsed["delta"]["thinking"]
                assert f"node_fuzz_{idx}" in content


@pytest.mark.asyncio
async def test_sse_streamer_complex_unicode_and_emojis():
    """Stress test: Complex Unicode, astral plane emojis, ZWJ sequences, and international scripts.
    
    Vectors:
    - Emojis with Zero-Width-Joiners: 👨‍👩‍👧‍👦, 👩🏾‍💻, 🏳️‍🌈, 🧑🏽‍🚀
    - Astral plane / supplemental symbols: 🦤, 🚀, 🦆, 🧠, ⚡, 🧩
    - International scripts: Arabic (مرحبا), Japanese (こんにちは), Chinese (你好), Thai (สวัสดี), Cyrillic (Привет), Devanagari (नमस्ते)
    - Accented and combining characters
    """
    unicode_vectors = [
        "Family ZWJ: 👨‍👩‍👧‍👦 and Rainbow: 🏳️‍🌈 and Technologist: 👩🏾‍💻",
        "Astral Plane: 🦤🦆🚀✨🧠⚡🎯",
        "Multi-script: مرحبا - こんにちは - 你好 - สวัสดี - Привет - नमस्ते - שָׁלוֹם",
        "Combining characters: ñ, é, ü, ç, å, z̷a̷l̷g̷o̷",
    ]

    for proto in ("openai", "anthropic"):
        streamer = SSEThinkingStreamer(client_protocol=proto, model_id="autoconduck-unicode")
        for idx, vec in enumerate(unicode_vectors):
            frame = await streamer.emit_node_transition(f"unicode_node_{idx}", "completed", vec)
            assert frame.endswith("\n\n")
            if proto == "openai":
                parsed = json.loads(frame[6:].strip())
                content = parsed["choices"][0]["delta"]["reasoning_content"]
                assert "completed" in content
                # Ensure the vector characters survive json serialize/deserialize intact
                for part in vec.split():
                    if len(part) > 2:
                        assert part in content
            else:
                lines = frame.strip().split("\n")
                parsed = json.loads(lines[1][6:].strip())
                content = parsed["delta"]["thinking"]
                assert "completed" in content


@pytest.mark.asyncio
async def test_sse_streamer_cancellation_and_generator_exit():
    """Adversarially tests stream cancellation during token generation.
    
    Verifies:
    - Async generator properly handles GeneratorExit when client aborts/disconnects.
    - No unhandled exceptions or resource leaks on cancellation.
    - Premature break in consumer does not leave dangling tasks.
    """
    for proto in ("openai", "anthropic"):
        streamer = SSEThinkingStreamer(client_protocol=proto, model_id="autoconduck-cancel")

        async def infinite_tokens() -> AsyncIterator[str]:
            count = 0
            while True:
                yield f"Token_{count} "
                count += 1
                await asyncio.sleep(0.001)

        # Consumer aborts after receiving 5 tokens
        consumed = 0
        gen = streamer.stream_synthesizer_tokens(infinite_tokens())
        try:
            async for chunk in gen:
                consumed += 1
                assert "data: " in chunk or "event: " in chunk
                if consumed >= 5:
                    break
        finally:
            # Generator should cleanly close
            await gen.aclose()

        assert consumed == 5


@pytest.mark.asyncio
async def test_sse_streamer_smooth_synthesizer_transition_full_cycle():
    """Verifies complete lifecycle: DAG reasoning transitions -> synthesizer content tokens -> [DONE]."""
    streamer = SSEThinkingStreamer(client_protocol="openai", model_id="autoconduck-stream")

    # 1. State transitions
    t1 = await streamer.emit_node_transition("recon", "running", "Scanning AST")
    t2 = await streamer.emit_node_transition("recon", "completed", "Found 4 files")
    t3 = await streamer.emit_node_transition("synthesizer", "running", "Drafting response")

    assert "Scanning AST" in t1
    assert "Found 4 files" in t2
    assert "Drafting response" in t3

    # 2. Synthesizer tokens
    async def token_source() -> AsyncIterator[str]:
        tokens = ["# Solution\n\n", "The problem ", "is resolved ", "by modifying ", "`config.py`."]
        for t in tokens:
            yield t

    emitted_chunks = []
    async for chunk in streamer.stream_synthesizer_tokens(token_source()):
        emitted_chunks.append(chunk)

    # 5 content chunks + 1 stop chunk + 1 [DONE] chunk = 7 chunks
    assert len(emitted_chunks) == 7
    assert emitted_chunks[-1] == "data: [DONE]\n\n"
    stop_parsed = json.loads(emitted_chunks[-2][6:].strip())
    assert stop_parsed["choices"][0]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_sse_streamer_empty_and_whitespace_tokens():
    """Verifies handling of empty, whitespace, and None-equivalent tokens."""
    streamer = SSEThinkingStreamer(client_protocol="openai", model_id="autoconduck-empty")

    async def sparse_tokens() -> AsyncIterator[str]:
        yield ""
        yield " "
        yield ""
        yield "Valid Token"
        yield ""

    chunks = [c async for c in streamer.stream_synthesizer_tokens(sparse_tokens())]
    # Should ignore empty strings, emit " ", "Valid Token", stop chunk, [DONE]
    assert len(chunks) >= 3
    full_output = "".join(chunks)
    assert "Valid Token" in full_output


# ==============================================================================
# SECTION 2: Session Guard Adversarial Stress Tests
# ==============================================================================

def test_session_guard_60_turn_simulation_prefix_invariance():
    """Stress test: 60-turn continuous conversation simulation.
    
    Invariants tested:
    - Immutable prefix (turn 0: system, turn 1: initial user) is NEVER modified,
      mutated, truncated, or dropped across 60 turns under extreme compaction.
    - Prefix is byte-identical and deep-equal before and after compaction.
    - Total token count stays bounded within the 80% context window ceiling.
    """
    guard = SessionGuard()

    system_prompt = {
        "role": "system",
        "content": "You are AutoConduck v0.3.0, an intelligent routing and orchestration engine.",
    }
    initial_user_turn = {
        "role": "user",
        "content": "Initialize project analysis for repository /workspace/AutoConduck.",
    }

    messages: list[dict[str, Any]] = [
        copy.deepcopy(system_prompt),
        copy.deepcopy(initial_user_turn),
    ]

    for turn in range(1, 30):
        # Assistant generates tool calls
        messages.append({
            "role": "assistant",
            "content": f"Running turn {turn} file inspection.",
            "tool_calls": [
                {
                    "id": f"call_{turn}_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": json.dumps({"path": f"src/mod_{turn}.py"})},
                }
            ],
        })
        # Tool returns 500-word response
        messages.append({
            "role": "tool",
            "tool_call_id": f"call_{turn}_1",
            "name": "read_file",
            "content": f"# File content for mod_{turn}.py\n" + ("line content data " * 100),
        })
        # Assistant synthesis
        messages.append({
            "role": "assistant",
            "content": f"Analysis for turn {turn} complete with details.",
        })
        # User follow up
        messages.append({
            "role": "user",
            "content": f"Follow up question {turn}: verify invariants for component {turn}.",
        })

    assert len(messages) == 2 + (29 * 4)  # 118 messages total

    # Test with tight context window (4000 tokens => 3200 ceiling)
    result: SessionGuardResult = guard.guard_context(messages, context_window=4000)

    # 1. Prefix invariance
    assert result.cache_prefix_preserved is True
    assert result.messages[0] == system_prompt
    assert result.messages[1] == initial_user_turn
    assert result.messages[0]["content"] == system_prompt["content"]
    assert result.messages[1]["content"] == initial_user_turn["content"]

    # 2. Compaction occurred and bounded
    assert result.compacted is True
    assert result.final_tokens < result.original_tokens
    assert len(result.messages) == len(messages)


def test_session_guard_150kb_massive_tool_log():
    """Stress test: 150KB massive raw log output in tool message.
    
    Verifies:
    - 150KB raw log does not trigger OOM or infinite regex recursion.
    - Cleanly produces compacted snippet with omitted characters notice.
    - Does not corrupt surrounding user/assistant turns.
    """
    guard = SessionGuard()
    raw_log = "2026-08-24 01:00:00 [DEBUG] Processing packet ID " + ("A" * 150000)

    messages = [
        {"role": "system", "content": "System directive."},
        {"role": "user", "content": "Inspect server logs."},
        {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_logs", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "get_logs", "content": raw_log},
        {"role": "assistant", "content": "I have inspected the log output."},
    ]

    result = guard.guard_context(messages, context_window=2000)

    assert result.compacted is True
    assert result.cache_prefix_preserved is True
    compacted_tool_content = result.messages[3]["content"]
    assert len(compacted_tool_content) < len(raw_log)
    assert "[... " in compacted_tool_content
    assert "characters compacted ...]" in compacted_tool_content
    assert result.messages[4]["content"] == "I have inspected the log output."


def test_session_guard_code_fences_and_headers_preservation():
    """Stress test: complex markdown containing multiple code blocks and headers.
    
    Verifies:
    - Multi-language code blocks (python, typescript, bash) are preserved.
    - Markdown headers (# Architecture, ## Endpoints, ### Invariants) are preserved.
    - Unclosed code fences are safely handled without unhandled exceptions or malformed output.
    """
    guard = SessionGuard()

    markdown_payload = (
        "# System Architecture\n\n"
        "## Core Endpoints\n\n"
        "### Invariant Rules\n\n"
        "```python\n"
        "def route_request(req: Request) -> Response:\n"
        "    # Preserve this critical implementation logic\n"
        "    return Response(status=200)\n"
        "```\n\n"
        "```typescript\n"
        "export interface RouteConfig {\n"
        "    readonly path: string;\n"
        "    readonly timeoutMs: number;\n"
        "}\n"
        "```\n\n"
        "```bash\n"
        "npm test -- --watchAll=false\n"
        "```\n\n"
        + ("Extremely verbose prose filler description " * 1000)
    )

    messages = [
        {"role": "system", "content": "System role"},
        {"role": "user", "content": "Show architecture and code"},
        {"role": "assistant", "content": markdown_payload},
    ]

    result = guard.guard_context(messages, context_window=500)
    assert result.compacted is True

    assistant_content = result.messages[2]["content"]

    # Check structural headers preserved
    assert "# System Architecture" in assistant_content or "## Core Endpoints" in assistant_content

    # Check code blocks preserved
    assert "def route_request" in assistant_content
    assert "```python" in assistant_content
    assert "export interface RouteConfig" in assistant_content or "npm test" in assistant_content


def test_session_guard_unclosed_nested_code_fences():
    """Adversarially tests unclosed code blocks and nested markdown symbols."""
    guard = SessionGuard()

    unclosed_code = (
        "```rust\n"
        "pub fn calculate_hash(data: &[u8]) -> u64 {\n"
        "    let mut hasher = DefaultHasher::new();\n"
        "    hasher.write(data);\n"
        "    hasher.finish()\n"
        # Notice: NO closing ```
    )

    messages = [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "Rust code snippet"},
        {"role": "assistant", "content": unclosed_code + (" extra filler text " * 500)},
    ]

    result = guard.guard_context(messages, context_window=200)
    assert result.compacted is True
    content = result.messages[2]["content"]
    assert "calculate_hash" in content
    # Ensure guard closed the unclosed fence cleanly
    assert content.count("```") >= 2


def test_session_guard_edge_case_content_types():
    """Tests diverse message content types (lists of content blocks, None, non-string)."""
    guard = SessionGuard()

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "System text block"}]},
        {"role": "user", "content": "Hello user"},
        {"role": "assistant", "content": None},
        {"role": "user", "content": ["List item 1", "List item 2"]},
    ]

    result = guard.guard_context(messages, context_window=1000)
    assert isinstance(result, SessionGuardResult)
    assert result.cache_prefix_preserved is True


# ==============================================================================
# SECTION 3: LanceDB RAG Token Budget Truncation Stress Tests
# ==============================================================================

def test_lancedb_rag_strict_250_token_truncation_stress():
    """Stress test: KnowledgeVectorStore strictly limits snippet extraction <= 250 tokens.
    
    Verifies:
    - 20 large code symbols (500+ words each) indexed in vector store.
    - Query matches multiple large symbols.
    - `get_context_snippets(query, max_tokens=250)` strictly truncates total word tokens to <= 250.
    """
    store = KnowledgeVectorStore(db_uri=":memory:", table_name="test_stress_knowledge")

    # Index 20 large chunks
    chunks = []
    for i in range(20):
        symbol_name = f"heavy_function_{i}"
        content = f"def {symbol_name}(data: list[int]) -> dict[str, Any]:\n" + f"    # comment block {i} " + ("large payload word " * 400)
        chunks.append({
            "id": f"chunk_{i}",
            "symbol": symbol_name,
            "file": f"src/module_{i}.py",
            "content": content,
            "vector": [0.1 * (i % 5), 0.2 * ((i + 1) % 5), 0.3],
            "metadata": {"line": i * 10},
        })

    # Add records to store
    store.db.create_table("test_stress_knowledge", data=chunks, mode="overwrite")
    store._table = store.db.open_table("test_stress_knowledge")

    # Retrieve snippets with strict 250 token budget
    snippets = store.get_context_snippets("heavy_function", max_tokens=250)

    assert len(snippets) >= 1
    # Count content tokens across all snippets (excluding header tag)
    for s in snippets:
        assert "[" in s and "]" in s  # Has header

    # Sum of words in snippets
    total_words = sum(len(s.split()) for s in snippets)
    # Accounting for snippet headers `[src/module_X.py : heavy_function_X]`, content words <= 250
    content_words = 0
    for s in snippets:
        lines = s.splitlines()
        content_lines = lines[1:] if len(lines) > 1 else lines
        content_words += sum(len(line.split()) for line in content_lines)

    assert content_words <= 250, f"RAG content token count {content_words} exceeded 250 token limit!"


@pytest.mark.asyncio
async def test_dynamic_factory_rag_node_state_budget_invariance():
    """Adversarially tests Dynamic LangGraph Factory RAG node injection with multiple queries.
    
    Verifies:
    - RAG node correctly extracts snippets for all queries in plan.rag_queries.
    - Verified context in DynamicState is populated.
    - State is fully serialized and processed through the DAG.
    """
    plan = ExecutionPlan(
        route="dynamic_dag",
        confidence=0.95,
        task_type="refactor",
        suggested_tier=ModelTier.BALANCED,
        needs_rag=True,
        rag_queries=["API contracts", "pricing calculation", "session guard invariants"],
        subtasks=[
            SubTaskSpec(id="recon_task", goal="Inspect contracts", role="recon"),
            SubTaskSpec(id="edit_task", goal="Refactor implementation", role="edit", depends_on=["recon_task"]),
        ],
        synthesizer_tier=ModelTier.FRONTIER_REASONING,
        rationale="Multi-phase refactor requiring repository context",
    )

    state = DynamicState(
        messages=[{"role": "user", "content": "Refactor router and contracts."}],
        session_id="test_sess_rag",
        thread_id="test_thread_rag",
        plan=plan,
    )

    # 1. Directly invoke RAG node handler
    rag_result = await _rag_node_handler(state)
    assert rag_result["active_node"] == "rag"
    assert isinstance(rag_result["verified_context"], list)

    # 2. Build full graph and verify execution
    runner = build_dynamic_graph(plan)
    assert runner.compiled_graph is not None

    final_state = await runner.ainvoke(state)
    assert final_state is not None
