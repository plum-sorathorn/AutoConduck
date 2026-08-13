import asyncio

import pytest

from autoconduck.config import Config
from autoconduck.messages_sse import AnthropicSSETranslator
from autoconduck.orchestrator.graph import _is_subagent_error
from autoconduck.orchestrator.planner import SubTask
from autoconduck.orchestrator.subagents import run_subagent


def test_subagent_stability_defaults():
    selection = Config().selection
    assert selection.subagent_timeout_s == 120.0
    assert selection.subagent_max_tokens == 4096


@pytest.mark.asyncio
async def test_run_subagent_timeout_is_tagged():
    class Client:
        def completion(self, **kwargs):
            import time
            time.sleep(0.05)
            return {"choices": [{"message": {"content": "late"}}]}

    task = SubTask(id="slow", goal="Inspect", scope=["a.py"], output_contract="text", constraints=[])
    cfg = Config()
    cfg.selection.subagent_timeout_s = 0.001
    result = await run_subagent(task, "", Client(), cfg=cfg)
    assert result.startswith("__SUBAGENT_ERROR__[slow]: timed out")


def test_is_subagent_error():
    assert _is_subagent_error("__SUBAGENT_ERROR__[a]: failed")
    assert not _is_subagent_error("ordinary analyst output")
    assert not _is_subagent_error(None)


def _tool_chunk(tool_id=None, arguments=""):
    tool_call = {"index": 0, "function": {"name": "lookup", "arguments": arguments}}
    if tool_id is not None:
        tool_call["id"] = tool_id
    return {"choices": [{"delta": {"tool_calls": [tool_call]}}]}


def test_tool_call_id_is_stable_and_adopts_late_id():
    translator = AnthropicSSETranslator("test")
    first = translator.translate(_tool_chunk(arguments='{"a":'))
    block = translator.blocks[translator.tool_indices[0]]
    synthetic_id = block["id"]
    assert synthetic_id.startswith("toolu_")
    assert first[1]["content_block"]["id"] == synthetic_id

    translator.translate(_tool_chunk(arguments="1}"))
    assert translator.blocks[translator.tool_indices[0]]["id"] == synthetic_id

    translator.translate(_tool_chunk("call_real", ""))
    assert translator.blocks[translator.tool_indices[0]]["id"] == "call_real"
