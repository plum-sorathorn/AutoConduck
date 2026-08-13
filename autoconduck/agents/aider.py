from __future__ import annotations

import shutil
from pathlib import Path

from .base import BaseAdapter
from ..config import Config


class AiderAdapter(BaseAdapter):
    binary_name = "aider"
    id = "aider"
    display_name = "Aider"

    def detect(self) -> bool:
        return shutil.which(self.binary_name) is not None or any(
            path.exists() for path in self.config_paths()
        )

    def config_paths(self) -> list[Path]:
        return [
            Path.cwd() / ".aider.conf.yml",
            Path.home() / ".aider.conf.yml",
            Path.home() / ".aider" / "config.yml",
        ]

    def patch(self, config: Config, port: int | None = None) -> None:
        endpoint = f"http://127.0.0.1:{port if port is not None else config.port}/v1"
        target = next((path for path in self.config_paths() if path.exists()), self.config_paths()[0])
        self._upsert_block(target, f"openai_api_base: {endpoint}\nopenai_api_type: openai")
