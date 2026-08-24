"""Adversarial stress-test suite for autoconduck._compat fallback layer and M1 artifacts."""

import asyncio
import json
import math
import sqlite3
import pytest
from pydantic import BaseModel, Field
from packaging.requirements import Requirement

import autoconduck
from autoconduck import _compat
from autoconduck._compat import (
    is_llama_cpp_available,
    get_llama_model,
    LlamaFallback,
    LlamaGrammarFallback,
    is_outlines_available,
    OutlinesFallback,
    generate_structured_json,
    is_lancedb_available,
    lancedb_connect,
    LanceDBFallbackConnection,
    LanceDBFallbackTable,
    LanceDBFallbackQuery,
    is_sqlite_checkpointer_available,
    get_sqlite_checkpointer,
    SqliteSaverFallback,
    CheckpointTupleFallback,
)
from autoconduck._compat.lancedb_fallback import _cosine_distance


# ==============================================================================
# 1. Version and Dependency Syntax Verification
# ==============================================================================

def test_adversarial_version_and_metadata():
    """Verify package version is 0.3.1 and matches pyproject.toml."""
    assert autoconduck.__version__ == "0.3.1"

    import tomllib
    from pathlib import Path
    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        pyproject_data = tomllib.load(f)

    assert pyproject_data["project"]["version"] == "0.3.1"
    assert pyproject_data["project"]["name"] == "autoconduck"


def test_adversarial_dependency_syntax_and_sync():
    """Verify PEP 508 syntax validity and strict parity between pyproject.toml and requirements.txt."""
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    pyproject_path = root / "pyproject.toml"
    requirements_path = root / "requirements.txt"

    with open(pyproject_path, "rb") as f:
        pyproject_data = tomllib.load(f)

    pyproject_deps = pyproject_data["project"]["dependencies"]
    with open(requirements_path, "r", encoding="utf-8") as f:
        req_lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    # Validate PEP 508 syntax
    pyproject_req_objs = [Requirement(dep) for dep in pyproject_deps]
    req_objs = [Requirement(req) for req in req_lines]

    pyproject_names = {r.name for r in pyproject_req_objs}
    req_names = {r.name for r in req_objs}

    assert pyproject_names == req_names, f"Mismatch: {pyproject_names.symmetric_difference(req_names)}"

    # Required 0.3.0 dependencies
    expected_new = {"onnxruntime", "outlines", "lancedb", "langgraph-checkpoint-sqlite"}
    for dep in expected_new:
        assert dep in pyproject_names, f"Missing new dependency: {dep}"

    # Verify semantic-router is removed
    assert "semantic-router" not in pyproject_names


# ==============================================================================
# 2. LlamaFallback Adversarial & Boundary Tests
# ==============================================================================

def test_adversarial_llama_fallback_edge_inputs():
    """Test LlamaFallback under extreme and adversarial inputs."""
    fallback = LlamaFallback(model_path="test-model.gguf")

    # Empty prompt completion
    resp_empty = fallback.create_completion(prompt="")
    assert resp_empty["choices"][0]["text"] == "{}"
    assert resp_empty["usage"]["prompt_tokens"] == 0

    # Stream completion
    stream_iter = fallback.create_completion(prompt="Hello", stream=True)
    chunks = list(stream_iter)
    assert len(chunks) == 1
    assert chunks[0]["choices"][0]["text"] == "{}"

    # Callable __call__
    resp_call = fallback(prompt="Test prompt")
    assert "choices" in resp_call

    # Chat completion with empty messages
    chat_empty = fallback.create_chat_completion(messages=[])
    assert chat_empty["usage"]["prompt_tokens"] == 0

    # Chat completion with missing content and non-string content
    chat_edge = fallback.create_chat_completion(
        messages=[
            {"role": "user"},  # Missing content key
            {"role": "user", "content": None},
            {"role": "assistant", "content": 12345},
            {"role": "user", "content": ["multimodal", "parts"]},
        ]
    )
    assert "choices" in chat_edge
    assert chat_edge["choices"][0]["message"]["content"] == "{}"

    # Tokenize edge cases
    assert fallback.tokenize("") == []
    assert fallback.tokenize(b"") == []
    assert fallback.tokenize("abc") == [97, 98, 99]
    assert fallback.tokenize(b"abc") == [97, 98, 99]

    # Detokenize edge cases
    assert fallback.detokenize([]) == b""
    assert fallback.detokenize([97, 98, 99]) == b"abc"

    # Eval and reset calls
    fallback.eval([1, 2, 3])
    fallback.reset()


def test_adversarial_get_llama_model_mock_failure(monkeypatch):
    """Test get_llama_model when native llama_cpp raises during instantiation."""
    import autoconduck._compat.llama_fallback as lf

    class FakeBrokenLlama:
        def __init__(self, *args, **kwargs):
            raise OSError("Shared library not found: llama.dll")

    monkeypatch.setattr(lf, "HAS_LLAMA_CPP", True)
    monkeypatch.setattr(lf, "llama_cpp", type("FakeModule", (), {"Llama": FakeBrokenLlama})())

    model = lf.get_llama_model("test.gguf")
    assert isinstance(model, LlamaFallback)
    assert model._is_fallback is True


# ==============================================================================
# 3. OutlinesFallback Adversarial & Boundary Tests
# ==============================================================================

class StrictSchema(BaseModel):
    id: str
    count: int
    tags: list[str] = Field(default_factory=list)


def test_adversarial_outlines_malformed_and_edge_json():
    """Test Outlines fallback when LLM output is malformed, partial, or non-JSON."""
    # Case 1: Model returning malformed JSON repairable by json_repair
    class MalformedModel:
        def create_completion(self, prompt, **kwargs):
            return {"choices": [{"text": "{id: 'doc-1', count: 42, tags: ['a', 'b']}"}]}

    gen = OutlinesFallback(MalformedModel()).build_json_generator(StrictSchema)
    res = gen("prompt")
    assert isinstance(res, StrictSchema)
    assert res.id == "doc-1"
    assert res.count == 42
    assert res.tags == ["a", "b"]

    # Case 2: Model returning completely unparseable garbage text
    class GarbageModel:
        def create_completion(self, prompt, **kwargs):
            return {"choices": [{"text": "Fatal error: unexpected token <"}]}

    gen_garbage = OutlinesFallback(GarbageModel()).build_json_generator(StrictSchema)
    res_garbage = gen_garbage("prompt")
    assert isinstance(res_garbage, StrictSchema)
    # Note: For StrictSchema where 'id' has no default, model_construct() creates unvalidated instance
    assert getattr(res_garbage, "tags", None) == []

    # Case 2b: Test with ExecutionPlan schema from PROJECT.md
    from enum import Enum
    from typing import Literal

    class ModelTier(str, Enum):
        CHEAP_FAST = "cheap_fast"
        BALANCED = "balanced"
        FRONTIER_REASONING = "frontier_reasoning"

    class SubTaskSpec(BaseModel):
        id: str = "subtask-1"
        goal: str = "goal"
        role: Literal["recon", "read", "edit", "verify", "bash", "reasoning"] = "read"

    class ExecutionPlan(BaseModel):
        route: Literal["fast_direct", "dynamic_dag"] = "fast_direct"
        confidence: float = Field(ge=0.0, le=1.0, default=1.0)
        suggested_tier: ModelTier = ModelTier.BALANCED
        subtasks: list[SubTaskSpec] = Field(default_factory=list)

    gen_plan = OutlinesFallback(GarbageModel()).build_json_generator(ExecutionPlan)
    res_plan = gen_plan("prompt")
    assert isinstance(res_plan, ExecutionPlan)
    assert res_plan.route == "fast_direct"
    assert res_plan.suggested_tier == ModelTier.BALANCED

    # Case 3: Model returning non-dict JSON array
    class ArrayModel:
        def create_completion(self, prompt, **kwargs):
            return {"choices": [{"text": "[1, 2, 3]"}]}

    gen_array = OutlinesFallback(ArrayModel()).build_json_generator(StrictSchema)
    res_array = gen_array("prompt")
    assert isinstance(res_array, StrictSchema)

    # Case 4: Model is callable function returning raw string
    callable_model = lambda prompt, **kwargs: '{"id": "call-1", "count": 7}'
    gen_call = OutlinesFallback(callable_model).build_json_generator(StrictSchema)
    res_call = gen_call("prompt")
    assert isinstance(res_call, StrictSchema)
    assert res_call.id == "call-1"
    assert res_call.count == 7

    # Case 5: Model is None
    gen_none = OutlinesFallback(None).build_json_generator(StrictSchema)
    res_none = gen_none("prompt")
    assert isinstance(res_none, StrictSchema)

    # Case 6: Dict schema
    dict_schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    gen_dict = OutlinesFallback(callable_model).build_json_generator(dict_schema)
    res_dict = gen_dict("prompt")
    assert isinstance(res_dict, dict)
    assert res_dict.get("id") == "call-1"


# ==============================================================================
# 4. LanceDBFallback Adversarial & Boundary Tests
# ==============================================================================

def test_adversarial_cosine_distance_edge_vectors():
    """Test _cosine_distance against zero vectors, mismatched dimensions, and empty vectors."""
    # Empty vectors
    assert _cosine_distance([], []) == 1.0
    assert _cosine_distance([], [1.0, 2.0]) == 1.0
    assert _cosine_distance([1.0], []) == 1.0

    # Dimension mismatch
    assert _cosine_distance([1.0, 2.0], [1.0, 2.0, 3.0]) == 1.0

    # Zero norm vectors
    assert _cosine_distance([0.0, 0.0], [0.0, 0.0]) == 1.0
    assert _cosine_distance([0.0, 0.0], [1.0, 1.0]) == 1.0

    # Orthogonal vectors
    assert math.isclose(_cosine_distance([1.0, 0.0], [0.0, 1.0]), 1.0, rel_tol=1e-5)

    # Opposite vectors
    assert math.isclose(_cosine_distance([1.0, 0.0], [-1.0, 0.0]), 2.0, rel_tol=1e-5)

    # Identical vectors
    assert math.isclose(_cosine_distance([1.0, 0.0], [1.0, 0.0]), 0.0, rel_tol=1e-5)


def test_adversarial_lancedb_table_operations():
    """Test LanceDB fallback table with edge schemas, missing vector fields, and filter edge cases."""
    conn = LanceDBFallbackConnection(":memory:")
    table = conn.create_table("edge_docs")

    class ItemObject:
        def __init__(self, id, vector):
            self.id = id
            self.vector = vector

    class PydanticItem(BaseModel):
        id: str
        vector: list[float]

    # Add heterogeneous items (dict, Pydantic, plain class instance)
    table.add([
        {"id": "doc1", "vector": [1.0, 0.0, 0.0], "meta": "first"},
        {"id": "doc2", "no_vector_field": True},
        {"id": "doc3", "vector": None},
        {"id": "doc4", "vector": "invalid_type"},
        PydanticItem(id="doc5", vector=[0.0, 1.0, 0.0]),
        ItemObject(id="doc6", vector=[0.0, 0.0, 1.0]),
    ])

    assert table.count_rows() == 6

    # Search with valid vector: closest to doc1
    results = table.search(query=[0.9, 0.1, 0.0]).limit(2).to_list()
    assert len(results) == 2
    assert results[0]["id"] == "doc1"
    assert results[0]["_distance"] < 0.2

    # Search with None query vector
    none_query_res = table.search(query=None).limit(10).to_list()
    assert len(none_query_res) == 6
    assert none_query_res[0]["_distance"] == 0.0

    # Search on empty table
    empty_table = conn.create_table("empty_table")
    assert empty_table.search(query=[1.0, 0.0]).to_list() == []

    # Select columns filter
    selected = table.search(query=[1.0, 0.0, 0.0]).select(["id", "meta"]).limit(1).to_list()
    assert len(selected) == 1
    assert "id" in selected[0]
    assert "_distance" in selected[0]
    assert "vector" not in selected[0]

    # Delete with where filter
    table.delete(where="id == 'doc1'")
    assert table.count_rows() == 5

    # Delete all
    table.delete()
    assert table.count_rows() == 0

    # Table creation conflict
    conn.create_table("unique_table", exist_ok=True)
    with pytest.raises(ValueError):
        conn.create_table("unique_table", exist_ok=False, mode="create")


# ==============================================================================
# 5. SqliteSaverFallback Adversarial & Boundary Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_adversarial_sqlite_saver_fallback():
    """Test SqliteSaverFallback with unhashable metadata, corrupted data, and async operations."""
    saver = SqliteSaverFallback(":memory:")

    # Setup is idempotent
    saver.setup()
    saver.setup()

    # Put checkpoint with non-serializable objects in metadata/checkpoint
    class CustomObj:
        def __str__(self):
            return "CustomObjStr"

    config = {"configurable": {"thread_id": "thread-1", "checkpoint_ns": "ns-1"}}
    chk = {"id": "chk-1", "data": {"key": CustomObj()}}
    meta = {"timestamp": CustomObj()}

    saved_cfg = saver.put(config, chk, meta)
    assert saved_cfg["configurable"]["checkpoint_id"] == "chk-1"

    # Retrieve tuple
    tup = saver.get_tuple(config)
    assert tup is not None
    assert tup.checkpoint["id"] == "chk-1"
    assert tup.checkpoint["data"]["key"] == "CustomObjStr"

    # Put writes with multiple channels
    saver.put_writes(config, [("chan1", {"val": 1}), ("chan2", CustomObj())], task_id="task-1")

    # List checkpoints
    items = list(saver.list(config))
    assert len(items) == 1

    # List without config
    all_items = list(saver.list(config=None))
    assert len(all_items) == 1

    # Corrupt data in database directly to verify safe handling
    with saver.conn:
        saver.conn.execute("UPDATE checkpoints SET checkpoint = 'CORRUPTED_JSON' WHERE checkpoint_id = 'chk-1'")

    corrupted_tup = saver.get_tuple(config)
    assert corrupted_tup is not None
    assert corrupted_tup.checkpoint == {}

    # Async methods
    async_config = {"configurable": {"thread_id": "async-thread", "checkpoint_ns": "main"}}
    await saver.aput(async_config, {"id": "async-chk", "state": 1}, {"meta": 2})
    await saver.aput_writes(async_config, [("c1", "v1")], task_id="t1")

    async_tup = await saver.aget_tuple(async_config)
    assert async_tup is not None
    assert async_tup.checkpoint["id"] == "async-chk"

    async_list = []
    async for item in saver.alist(async_config):
        async_list.append(item)
    assert len(async_list) == 1
