"""Compatibility fallback for Outlines structured generation when outlines is absent."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, TypeVar, Type
from pydantic import BaseModel

logger = logging.getLogger(__name__)

try:
    import outlines
    from outlines import generate as outlines_generate
    HAS_OUTLINES = True
except Exception:
    outlines = None
    outlines_generate = None
    HAS_OUTLINES = False

T = TypeVar("T", bound=BaseModel)


def is_outlines_available() -> bool:
    """Return True if outlines library is installed and available."""
    return HAS_OUTLINES and outlines is not None


class OutlinesFallback:
    """Fallback handler for structured JSON generation with or without outlines."""

    def __init__(self, model: Any = None) -> None:
        self.model = model
        self._is_fallback = not is_outlines_available()

    def build_json_generator(self, schema_or_model: type[T] | dict[str, Any] | str) -> Callable[..., Any]:
        """Build a callable generator for the given schema."""
        if is_outlines_available() and self.model is not None and not getattr(self.model, "_is_fallback", False):
            try:
                return outlines_generate.json(self.model, schema_or_model)  # type: ignore[union-attr]
            except Exception as exc:
                logger.warning("outlines.generate.json failed: %s. Using fallback.", exc)

        schema_cls: type[T] | None = None
        if isinstance(schema_or_model, type) and issubclass(schema_or_model, BaseModel):
            schema_cls = schema_or_model

        def fallback_generator(prompt: str, **kwargs: Any) -> Any:
            raw_text = "{}"
            if self.model is not None:
                if hasattr(self.model, "create_completion"):
                    resp = self.model.create_completion(prompt, **kwargs)
                    if isinstance(resp, dict) and "choices" in resp and len(resp["choices"]) > 0:
                        raw_text = resp["choices"][0].get("text", "{}")
                elif callable(self.model):
                    resp = self.model(prompt, **kwargs)
                    if isinstance(resp, dict) and "choices" in resp and len(resp["choices"]) > 0:
                        raw_text = resp["choices"][0].get("text", "{}")
                    elif isinstance(resp, str):
                        raw_text = resp

            # Try parsing with json_repair if available, else json.loads
            try:
                import json_repair
                parsed = json_repair.loads(raw_text)
            except Exception:
                try:
                    parsed = json.loads(raw_text)
                except Exception:
                    parsed = {}

            if schema_cls is not None:
                if isinstance(parsed, dict):
                    try:
                        return schema_cls.model_validate(parsed)
                    except Exception:
                        return schema_cls.model_construct(**parsed)
                return schema_cls.model_construct()
            return parsed

        return fallback_generator


def generate_structured_json(
    model: Any,
    prompt: str,
    schema_or_model: type[T] | dict[str, Any],
    **kwargs: Any,
) -> Any:
    """Generate structured output adhering to schema_or_model."""
    handler = OutlinesFallback(model)
    generator = handler.build_json_generator(schema_or_model)
    return generator(prompt, **kwargs)
