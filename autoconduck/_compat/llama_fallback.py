"""Compatibility fallback for llama_cpp when native C++ binaries are absent."""

from __future__ import annotations

import logging
import time
from typing import Any, Iterator, Sequence

logger = logging.getLogger(__name__)

try:
    import llama_cpp
    HAS_LLAMA_CPP = True
except Exception:  # catches ImportError, OSError on missing C++ libs
    llama_cpp = None
    HAS_LLAMA_CPP = False


def is_llama_cpp_available() -> bool:
    """Return True if native llama_cpp is installed and importable."""
    return HAS_LLAMA_CPP and llama_cpp is not None


class LlamaGrammarFallback:
    """Safe fallback for llama_cpp.LlamaGrammar."""

    def __init__(self, grammar_str: str = "", root: str = "root", **kwargs: Any) -> None:
        self.grammar_str = grammar_str
        self.root = root
        self.kwargs = kwargs

    @classmethod
    def from_string(cls, grammar_str: str, **kwargs: Any) -> LlamaGrammarFallback:
        return cls(grammar_str=grammar_str, **kwargs)


class LlamaFallback:
    """Pure-Python fallback emulator for llama_cpp.Llama."""

    def __init__(
        self,
        model_path: str = "",
        n_ctx: int = 2048,
        n_gpu_layers: int = 0,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose
        self.extra_kwargs = kwargs
        self._is_fallback = True

    def create_completion(
        self,
        prompt: str = "",
        max_tokens: int = 256,
        temperature: float = 0.2,
        stop: list[str] | str | None = None,
        stream: bool = False,
        grammar: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any] | Iterator[dict[str, Any]]:
        """Simulate completion generation for testing and graceful fallback."""
        text_response = "{}"
        result = {
            "id": f"cmpl-fallback-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": self.model_path or "fallback-slm",
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
            "id": f"chatcmpl-fallback-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_path or "fallback-slm",
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

    def tokenize(self, text: bytes | str, add_bos: bool = True, special: bool = False) -> list[int]:
        if isinstance(text, str):
            text_bytes = text.encode("utf-8")
        else:
            text_bytes = text
        return list(text_bytes)

    def detokenize(self, tokens: Sequence[int]) -> bytes:
        return bytes(tokens)

    def eval(self, tokens: Sequence[int]) -> None:
        pass

    def reset(self) -> None:
        pass

    def __call__(self, prompt: str = "", **kwargs: Any) -> dict[str, Any]:
        return self.create_completion(prompt=prompt, **kwargs)  # type: ignore[return-value]


def get_llama_model(model_path: str, **kwargs: Any) -> Any:
    """Instantiate a llama_cpp.Llama instance if available, or return LlamaFallback."""
    if is_llama_cpp_available():
        try:
            return llama_cpp.Llama(model_path=model_path, **kwargs)  # type: ignore[union-attr]
        except Exception as exc:
            logger.warning(
                "Failed to initialize native llama_cpp model from %s (%s). Using fallback.",
                model_path,
                exc,
            )
            return LlamaFallback(model_path=model_path, **kwargs)
    return LlamaFallback(model_path=model_path, **kwargs)
