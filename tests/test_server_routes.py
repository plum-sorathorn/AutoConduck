from __future__ import annotations

import types

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from autoconduck import config as config_module
from autoconduck import messages_api as messages
from autoconduck import server_routes


class _App:
    def __init__(self):
        self.handlers = {}

    def get(self, path):
        return self._decorator("GET", path)

    def post(self, path):
        return self._decorator("POST", path)

    def _decorator(self, method, path):
        def register(handler):
            self.handlers[(method, path)] = handler
            return handler
        return register


class _Request:
    async def is_disconnected(self):
        return False


@pytest.mark.asyncio
async def test_streaming_completion_uses_qualified_model_and_api_base(monkeypatch):
    cfg = types.SimpleNamespace(custom_models=[{
        "id": "deepseek-v4-flash",
        "base_url": "https://example.test/v1",
        "enabled": True,
    }])
    monkeypatch.setattr(config_module, "get_config", lambda: cfg)

    captured = {}

    class _LLM:
        async def acompletion(self, **kwargs):
            captured.update(kwargs)

            async def chunks():
                yield {"choices": []}

            return chunks()

    import autoconduck.server_streaming as streaming
    monkeypatch.setattr(streaming, "_litellm", lambda: _LLM())

    app = _App()
    cache = {}
    helpers = (
        lambda body: [], lambda tools: [], lambda choice: None,
        messages.litellm_params_for, lambda text: 0,
        lambda model, **kwargs: None, lambda result: "", lambda value: value,
        lambda tools: tools, messages.messages_litellm_kwargs,
    )
    server_routes.install_routes(
        app, Request, JSONResponse, StreamingResponse, BaseModel, Field,
        helpers, lambda cfg: [], messages.PSEUDO_MODELS, cache,
    )
    body = cache["CompletionRequest"](model="deepseek-v4-flash", stream=True,
                                       messages=[{"role": "user", "content": "hi"}])

    response = await app.handlers[("POST", "/v1/chat/completions")](body, _Request())
    async for _ in response.body_iterator:
        pass

    assert captured["model"] == "openai/deepseek-v4-flash"
    assert captured["api_base"] == "https://example.test/v1"
    assert captured["drop_params"] is True
