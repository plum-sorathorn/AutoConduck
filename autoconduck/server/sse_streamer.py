"""Dynamic SSE Thinking Streamer.

Emits real-time visual DAG execution state transitions ([..], [>>], [OK], [ERR]) as
`delta.reasoning_content` (OpenAI) and `thinking_delta` (Anthropic), transitioning
smoothly into markdown response tokens without stream stalls or duplicate chunks.
"""
from __future__ import annotations

import json
import logging
import time
from typing import AsyncIterator, Literal

logger = logging.getLogger(__name__)

STATUS_GLYPHS = {
    "pending": "[..]",
    "running": "[>>]",
    "completed": "[OK]",
    "failed": "[ERR]",
}

PROGRESS_LABELS = {
    "recon": "recon",
    "recon_subagent_pool": "reading files",
    "planner": "planner",
    "subagent_pool": "subagents",
    "compactor": "compactor",
    "executor": "executor",
    "synthesizer": "synthesizer",
    "rag": "rag",
    "slm_plan": "slm_plan",
}


def render_progress_event(event: dict[str, object]) -> str:
    """Render one orchestration progress event as compact plain ASCII."""
    node = str(event.get("node") or "progress")
    state = str(event.get("state", "running"))
    detail = str(event.get("step_detail") or node)
    label = PROGRESS_LABELS.get(node, node)
    if node in {"slm_plan", "subagent_pool"}:
        lines = detail.splitlines() or [detail]
        return "\n".join([f"+-- [{label}]", *[f"| {line}" for line in lines], "'--"]) + "\n"
    glyph = STATUS_GLYPHS.get(state, "[..]")
    return f"{glyph} [{label}] {detail}\n"


class SSEThinkingStreamer:
    """Streams thinking DAG node state transitions and synthesizer tokens via SSE."""

    def __init__(
        self,
        client_protocol: Literal["openai", "anthropic"] | str = "openai",
        model_id: str = "autoconduck",
    ) -> None:
        proto = str(client_protocol).lower()
        self.client_protocol: Literal["openai", "anthropic"] = (
            "anthropic" if proto == "anthropic" else "openai"
        )
        self.model_id = model_id
        self._created = int(time.time())

    async def emit_node_transition(
        self,
        node_name: str,
        status: Literal["pending", "running", "completed", "failed"] | str,
        detail: str = "",
    ) -> str:
        """Format an SSE frame for a node transition."""
        glyph = STATUS_GLYPHS.get(status, "[..]")
        detail_suffix = f": {detail}" if detail else ""
        reasoning_line = f"{glyph} [{node_name}] {status}{detail_suffix}\n"

        if self.client_protocol == "openai":
            chunk = {
                "id": f"chatcmpl-dag-{self._created}",
                "object": "chat.completion.chunk",
                "created": self._created,
                "model": self.model_id,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": reasoning_line},
                        "finish_reason": None,
                    }
                ],
            }
            return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        else:
            # Anthropic thinking delta block
            event_data = {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "thinking_delta",
                    "thinking": reasoning_line,
                },
            }
            return f"event: content_block_delta\ndata: {json.dumps(event_data, ensure_ascii=False)}\n\n"

    async def stream_synthesizer_tokens(
        self, token_async_iter: AsyncIterator[str]
    ) -> AsyncIterator[str]:
        """Stream token generator tokens smoothly into content tokens."""
        has_emitted = False

        if self.client_protocol == "openai":
            async for token in token_async_iter:
                if not token:
                    continue
                has_emitted = True
                chunk = {
                    "id": f"chatcmpl-syn-{self._created}",
                    "object": "chat.completion.chunk",
                    "created": self._created,
                    "model": self.model_id,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": token},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

            # Terminal stop chunk
            stop_chunk = {
                "id": f"chatcmpl-syn-{self._created}",
                "object": "chat.completion.chunk",
                "created": self._created,
                "model": self.model_id,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(stop_chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        else:
            # Anthropic text delta stream
            async for token in token_async_iter:
                if not token:
                    continue
                has_emitted = True
                delta_payload = {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {
                        "type": "text_delta",
                        "text": token,
                    },
                }
                yield f"event: content_block_delta\ndata: {json.dumps(delta_payload, ensure_ascii=False)}\n\n"

            stop_ev = {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 1 if has_emitted else 0},
            }
            yield f"event: message_delta\ndata: {json.dumps(stop_ev, ensure_ascii=False)}\n\n"
            stop_msg = {"type": "message_stop"}
            yield f"event: message_stop\ndata: {json.dumps(stop_msg, ensure_ascii=False)}\n\n"
