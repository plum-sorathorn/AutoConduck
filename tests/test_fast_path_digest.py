import asyncio
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from autoconduck.digest import maybe_digest_messages
from autoconduck.main import _build
from autoconduck import main, server_streaming


def cfg(**kwargs):
    return SimpleNamespace(**kwargs)


@pytest.mark.asyncio
async def test_skip_when_flag_disabled(tmp_path):
    assert await maybe_digest_messages(
        [{"role": "user", "content": "a.py b.py"}],
        cfg(fast_path_digest_enabled=False), tmp_path) is None


@pytest.mark.asyncio
async def test_skip_when_not_first_turn(tmp_path):
    assert await maybe_digest_messages([
        {"role": "user", "content": "a.py b.py"}, {"role": "assistant", "content": "ok"}
    ], base_dir=tmp_path) is None
    assert await maybe_digest_messages([
        {"role": "user", "content": "a.py b.py", "tool_calls": []}
    ], base_dir=tmp_path) is None


@pytest.mark.asyncio
async def test_skip_when_few_candidates(tmp_path):
    (tmp_path / "a.py").write_text("a\n", encoding="utf-8")
    assert await maybe_digest_messages([{"role": "user", "content": "a.py"}], base_dir=tmp_path) is None


@pytest.mark.asyncio
async def test_extraction_resolves_existing_files(tmp_path):
    for name in ("a.py", "b.py", "ignored.py"):
        (tmp_path / name).write_text(f"# {name}\nvalue = 1\n", encoding="utf-8")
    result = await maybe_digest_messages(
        [{"role": "user", "content": "review `a.py` and b.py"}], base_dir=tmp_path)
    assert result and "### a.py" in result[0]["content"] and "### b.py" in result[0]["content"]


@pytest.mark.asyncio
async def test_parallel_and_bounded(tmp_path):
    for i in range(4):
        (tmp_path / f"f{i}.py").write_text((f"head{i}\n" * 10000), encoding="utf-8")
    result = await maybe_digest_messages(
        [{"role": "user", "content": "f0.py f1.py f2.py f3.py"}],
        cfg(fast_path_digest_max_total_bytes=500), tmp_path)
    assert result and len(result[0]["content"]) <= 500


@pytest.mark.asyncio
async def test_timeout_degrades_to_none(tmp_path, monkeypatch):
    for name in ("a.py", "b.py"):
        (tmp_path / name).write_text("head\n", encoding="utf-8")
    async def slow(*args, **kwargs):
        await asyncio.sleep(1)
    monkeypatch.setattr("autoconduck.digest._read_one", slow)
    result = await maybe_digest_messages(
        [{"role": "user", "content": "a.py b.py"}],
        cfg(fast_path_digest_timeout_ms=1), tmp_path)
    assert result is None


@pytest.mark.asyncio
async def test_error_degrades_to_readable_files(tmp_path, monkeypatch):
    for name in ("a.py", "b.py"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    original = __import__("autoconduck.digest", fromlist=["_read_head"])._read_head
    def broken(path, size):
        if path.name == "a.py":
            raise OSError("no access")
        return original(path, size)
    monkeypatch.setattr("autoconduck.digest._read_head", broken)
    assert await maybe_digest_messages(
        [{"role": "user", "content": "a.py b.py"}], base_dir=tmp_path) is None


@pytest.mark.asyncio
async def test_binary_file_skipped(tmp_path):
    (tmp_path / "a.py").write_bytes(b"ok\x00binary")
    (tmp_path / "b.py").write_text("text", encoding="utf-8")
    assert await maybe_digest_messages(
        [{"role": "user", "content": "a.py b.py"}], base_dir=tmp_path) is None


class _FakeResponse:
    model = "fake"

    def __init__(self):
        self.choices = [SimpleNamespace(message=SimpleNamespace(content="ok"))]

    def model_dump(self):
        return {"id": "fake", "model": self.model,
                "choices": [{"message": {"role": "assistant", "content": "ok"}}]}


class _FakeChunk:
    def model_dump(self):
        return {"choices": [{"delta": {"role": "assistant", "content": "ok"},
                              "finish_reason": "stop"}]}


class _FakeStream:
    def __aiter__(self):
        return self

    async def __anext__(self):
        if hasattr(self, "done"):
            raise StopAsyncIteration
        self.done = True
        return _FakeChunk()


@pytest.fixture
def server_harness(monkeypatch, tmp_path):
    captured = []
    from types import SimpleNamespace
    monkeypatch.setattr("autoconduck.dispatcher.route",
                        lambda *args, **kwargs: SimpleNamespace(path="FAST", model="cheap-model"))
    monkeypatch.setattr("autoconduck.stats.stats_path", lambda: tmp_path / "stats.jsonl")
    main.app = None
    main._cached.clear()
    async def acompletion(**kwargs):
        captured.append(kwargs)
        return _FakeStream() if kwargs.get("stream") else _FakeResponse()
    monkeypatch.setattr(server_streaming, "_litellm", lambda: type("FakeLiteLLM", (), {
        "acompletion": staticmethod(acompletion),
    })())
    _build()
    return TestClient(main.app), captured


def _multi_file_prompt():
    return "Please review autoconduck/digest.py and autoconduck/config.py and summarize them."


def test_completions_fast_path_injects_digest(server_harness):
    client, calls = server_harness
    response = client.post("/v1/chat/completions", json={"model": "autoconduck",
        "messages": [{"role": "user", "content": _multi_file_prompt()}]})
    assert response.status_code == 200
    digest = next(message for message in calls[-1]["messages"] if message["role"] == "system")
    assert digest["content"].startswith("[AutoConduck file digests]")
    assert "### autoconduck/digest.py" in digest["content"]


def test_completions_trivial_prompt_no_digest(server_harness):
    client, calls = server_harness
    response = client.post("/v1/chat/completions", json={"model": "autoconduck",
        "messages": [{"role": "user", "content": "fix typo"}]})
    assert response.status_code == 200
    assert calls[-1]["messages"] == [{"role": "user", "content": "fix typo"}]


def test_messages_endpoint_injects_digest(server_harness):
    client, calls = server_harness
    response = client.post("/v1/messages", json={"model": "autoconduck", "max_tokens": 64,
        "messages": [{"role": "user", "content": _multi_file_prompt()}]})
    assert response.status_code == 200
    digest = next(message for message in calls[-1]["messages"] if message["role"] == "system")
    assert digest["content"].startswith("[AutoConduck file digests]")
    assert "### autoconduck/digest.py" in digest["content"]
