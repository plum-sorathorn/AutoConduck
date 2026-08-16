"""Anthropic SSE translation helpers."""
from __future__ import annotations
import json
import uuid
from typing import Any
def _coerce_content_text(content: Any) -> str:
    """Import the API helper lazily to avoid the API/SSE import cycle."""
    from .messages_api import coerce_content_text

    return coerce_content_text(content)

def count_tokens(text: str) -> int:
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text or ""))
    except Exception:
        return max(1, len(text or "") // 4)


def anthropic_response_text(
    content: Any, model: str = "", stop_reason: str = "end_turn", input_text: str = ""
) -> dict:
    text = _coerce_content_text(content)
    return {
        "id": "msg_" + uuid.uuid4().hex[:12],
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": count_tokens(input_text),
            "output_tokens": count_tokens(text),
        },
    }


class AnthropicSSETranslator:
    """Stateful translator: OpenAI stream chunks -> Anthropic SSE event dicts."""

    def __init__(self, model: str, input_text: str = ""):
        self.model = model
        self.message_id = "msg_" + uuid.uuid4().hex[:12]
        self.input_tokens = count_tokens(input_text) if input_text else 0
        self.started = False
        self.blocks: dict[int, dict] = {}
        self.text_index: int | None = None
        self.tool_indices: dict[int, int] = {}
        self.next_block_index = 0
        self.finished = False
        self.output_tokens = 0

    def _ensure_message_start(self) -> list[dict]:
        if self.started:
            return []
        self.started = True
        return [
            {
                "type": "message_start",
                "message": {
                    "id": self.message_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": self.model,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": self.input_tokens, "output_tokens": 0},
                },
            }
        ]

    def translate(self, chunk: dict) -> list[dict]:
        events: list[dict] = []
        events.extend(self._ensure_message_start())

        choices = chunk.get("choices") or []
        if not choices:
            return events
        choice = choices[0]
        delta = choice.get("delta") or {}

        if delta.get("content"):
            if self.text_index is None:
                self.text_index = self.next_block_index
                self.next_block_index += 1
                self.blocks[self.text_index] = {"kind": "text", "started": True}
                events.append(
                    {
                        "type": "content_block_start",
                        "index": self.text_index,
                        "content_block": {"type": "text", "text": ""},
                    }
                )
            self.output_tokens += count_tokens(delta["content"])
            events.append(
                {
                    "type": "content_block_delta",
                    "index": self.text_index,
                    "delta": {"type": "text_delta", "text": delta["content"]},
                }
            )
        elif delta.get("role") == "assistant" and self.text_index is None and not delta.get("tool_calls"):
            self.text_index = self.next_block_index
            self.next_block_index += 1
            self.blocks[self.text_index] = {"kind": "text", "started": True}
            events.append(
                {
                    "type": "content_block_start",
                    "index": self.text_index,
                    "content_block": {"type": "text", "text": ""},
                }
            )

        for tc in delta.get("tool_calls") or []:
            idx = tc.get("index", 0)
            if idx not in self.tool_indices:
                self.tool_indices[idx] = self.next_block_index
                self.next_block_index += 1
            block_index = self.tool_indices[idx]
            fn = tc.get("function") or {}
            if block_index not in self.blocks:
                tool_id = tc.get("id") or "toolu_" + uuid.uuid4().hex[:12]
                self.blocks[block_index] = {
                    "kind": "tool_use",
                    "started": True,
                    "id": tool_id,
                    "synthetic_id": not bool(tc.get("id")),
                }
                events.append(
                    {
                        "type": "content_block_start",
                        "index": block_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": self.blocks[block_index]["id"],
                            "name": fn.get("name"),
                            "input": {},
                        },
                    }
                )
            elif tc.get("id") and self.blocks[block_index].get("synthetic_id"):
                self.blocks[block_index]["id"] = tc["id"]
                self.blocks[block_index]["synthetic_id"] = False
            if fn.get("arguments"):
                self.output_tokens += count_tokens(fn["arguments"])
                events.append(
                    {
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {"type": "input_json_delta", "partial_json": fn["arguments"]},
                    }
                )

        finish_reason = choice.get("finish_reason")
        if finish_reason:
            if not self.blocks:
                self.text_index = self.next_block_index
                self.next_block_index += 1
                self.blocks[0] = {"kind": "text", "started": True}
                events.append({
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                })
            events.extend(self._close(finish_reason, chunk))

        return events

    def _close(self, finish_reason: Any, chunk: dict | None = None) -> list[dict]:
        from .messages_api import STOP_REASON_MAP

        events: list[dict] = []
        for idx in sorted(self.blocks.keys()):
            block = self.blocks[idx]
            if block.get("started") and not block.get("stopped"):
                block["stopped"] = True
                events.append({"type": "content_block_stop", "index": idx})
        stop_reason = STOP_REASON_MAP.get(finish_reason, finish_reason)
        usage_out = self.output_tokens
        if chunk is not None:
            usage = chunk.get("usage")
            if usage and usage.get("completion_tokens") is not None:
                usage_out = usage["completion_tokens"]
        events.append(
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": usage_out},
            }
        )
        events.append({"type": "message_stop"})
        self.finished = True
        return events

    def finish(self) -> list[dict]:
        if self.finished:
            return []
        # Some providers send only a finish_reason chunk. Anthropic still
        # requires a content block in the message, even when it is empty.
        events = self._ensure_message_start()
        if not self.blocks:
            self.text_index = self.next_block_index
            self.next_block_index += 1
            self.blocks[0] = {"kind": "text", "started": True}
            events.append({
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            })
        events.extend(self._close("stop"))
        return events

    def error(self, message: str) -> list[dict]:
        return [{
            "event": "error",
            "data": json.dumps({
                "type": "error",
                "error": {"type": "api_error", "message": message},
            }),
        }]
