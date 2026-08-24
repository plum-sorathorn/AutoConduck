"""Compatibility layer and fallback for ONNX Runtime embedded SLM models."""
from __future__ import annotations

import logging
import time
from typing import Any, Iterator, Sequence

logger = logging.getLogger(__name__)

try:
    import onnxruntime
    HAS_ONNX = True
except Exception:
    onnxruntime = None
    HAS_ONNX = False

try:
    import onnxruntime_genai
    HAS_ONNX_GENAI = True
except Exception:
    onnxruntime_genai = None
    HAS_ONNX_GENAI = False


def is_onnx_available() -> bool:
    """Return True if onnxruntime is installed and importable."""
    return HAS_ONNX and onnxruntime is not None


def is_onnx_genai_available() -> bool:
    """Return True if onnxruntime-genai is installed and importable."""
    return HAS_ONNX_GENAI and onnxruntime_genai is not None


class ONNXModelFallback:
    """Pure-Python fallback emulator for ONNX embedded SLM inference."""

    def __init__(self, model_path: str = "", **kwargs: Any) -> None:
        self.model_path = model_path
        self.extra_kwargs = kwargs
        self._is_fallback = True

    def create_completion(
        self,
        prompt: str = "",
        max_tokens: int = 256,
        temperature: float = 0.2,
        stop: list[str] | str | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any] | Iterator[dict[str, Any]]:
        """Simulate completion generation for testing and graceful fallback."""
        text_response = "{}"
        result = {
            "id": f"cmpl-onnx-fallback-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": self.model_path or "fallback-slm-onnx",
            "choices": [
                {
                    "text": text_response,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(text_response.split()),
                "total_tokens": len(prompt.split()) + len(text_response.split()),
            },
        }
        if stream:
            return iter([result])
        return result

    def create_chat_completion(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.2,
        stop: list[str] | str | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any] | Iterator[dict[str, Any]]:
        """Simulate chat completion for testing and graceful fallback."""
        content = "{}"
        result = {
            "id": f"chatcmpl-onnx-fallback-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_path or "fallback-slm-onnx",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": sum(len(str(m.get("content", "")).split()) for m in messages),
                "completion_tokens": len(content.split()),
                "total_tokens": sum(len(str(m.get("content", "")).split()) for m in messages) + len(content.split()),
            },
        }
        if stream:
            return iter([result])
        return result

    def __call__(self, prompt: str = "", **kwargs: Any) -> dict[str, Any]:
        return self.create_completion(prompt=prompt, **kwargs)  # type: ignore[return-value]


def get_onnx_model(model_path: str, **kwargs: Any) -> Any:
    """Instantiate an ONNX GenAI model instance if available, or return ONNXModelFallback."""
    if is_onnx_genai_available():
        try:
            return onnxruntime_genai.Model(model_path)
        except Exception as exc:
            logger.warning("Failed to initialize ONNX model from %s (%s). Using fallback.", model_path, exc)
            return ONNXModelFallback(model_path=model_path, **kwargs)
    return ONNXModelFallback(model_path=model_path, **kwargs)
