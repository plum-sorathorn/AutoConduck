from __future__ import annotations

from pathlib import Path

from ..config import Config
from .base import BaseAdapter


class GenericOpenAIAdapter(BaseAdapter):
    id = "generic_openai"
    display_name = "Generic OpenAI"

    def detect(self) -> bool:
        return True

    def config_paths(self) -> list[Path]:
        return []

    def patch(self, config: Config) -> None:
        print(f"[autoconduck] Generic OpenAI: set OPENAI_API_BASE=http://127.0.0.1:{config.port}/v1")
        print(f"  export OPENAI_API_BASE=http://127.0.0.1:{config.port}/v1")

    def revert(self) -> None:
        print("[autoconduck] Generic OpenAI: unset OPENAI_API_BASE if you set it manually")
