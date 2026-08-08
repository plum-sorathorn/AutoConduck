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
        return shutil.which("aider") is not None or any(p.exists() for p in self.config_paths())
    def config_paths(self) -> list[Path]:
        return [Path.cwd() / ".aider.conf.yml", Path.home() / ".aider.conf.yml", Path.home() / ".aider" / "config.yml"]
    def patch(self, config: Config) -> None:
        endpoint = f"http://127.0.0.1:{config.port}/v1"
        block = f"openai_api_base: {endpoint}\nopenai_api_type: openai"
        target = self.config_paths()[0]
        for p in self.config_paths():
            if p.exists():
                target = p
                break
        self._upsert_block(target, block)
