from __future__ import annotations
import shutil
from pathlib import Path
from .base import BaseAdapter
from ..config import Config

class OpenCodeAdapter(BaseAdapter):
    id = "opencode"
    display_name = "OpenCode"
    def detect(self) -> bool:
        if shutil.which("opencode") is not None:
            return True
        return any(p.exists() for p in self.config_paths())
    def config_paths(self) -> list[Path]:
        return [Path.cwd() / "opencode.json", Path.home() / ".config" / "opencode" / "config.json"]
    def patch(self, config: Config) -> None:
        endpoint = f"http://127.0.0.1:{config.port}/v1"
        def updater(data: dict):
            providers = data.setdefault("providers", {})
            ac = providers.setdefault("autoconduck", {})
            ac["api_base"] = endpoint
            ac["models"] = ["autoconduck","autoconduck-budget","autoconduck-expensive"]
        target = self.config_paths()[0]
        for p in self.config_paths():
            if p.exists():
                target = p
                break
        self._patch_json(target, updater)
