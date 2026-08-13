from __future__ import annotations

from pathlib import Path

from .base import BaseAdapter
from ..config import Config


class GenericOpenAIAdapter(BaseAdapter):
    id = "generic_openai"
    display_name = "Generic OpenAI"

    def detect(self) -> bool:
        return True

    def config_paths(self) -> list[Path]:
        return []

    def patch(self, config: Config, port: int | None = None) -> None:
        endpoint = f"http://127.0.0.1:{port if port is not None else config.port}/v1"
        print(f"[autoconduck] Generic OpenAI: set OPENAI_API_BASE={endpoint}")
        print(f"  export OPENAI_API_BASE={endpoint}")

    def revert(self) -> None:
        print("[autoconduck] Generic OpenAI: unset OPENAI_API_BASE if you set it manually")
