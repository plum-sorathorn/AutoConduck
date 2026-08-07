from __future__ import annotations
import shutil
from pathlib import Path
from .base import BaseAdapter
from ..config import Config

class ContinueDevAdapter(BaseAdapter):
    id = "continue_dev"
    display_name = "Continue"
    def detect(self) -> bool:
        return any(p.exists() for p in self.config_paths())
    def config_paths(self) -> list[Path]:
        return [Path.home() / ".continue" / "config.json"]
    def patch(self, config: Config) -> None:
        endpoint = f"http://127.0.0.1:{config.port}/v1"
        def updater(data: dict):
            models = data.setdefault("models", [])
            # remove old autoconduck entries
            models[:] = [m for m in models if m.get("title") not in ("autoconduck","autoconduck-budget","autoconduck-expensive")]
            for mid in ["autoconduck","autoconduck-budget","autoconduck-expensive"]:
                models.append({"title": mid, "provider": "openai", "model": mid, "apiBase": endpoint})
        self._patch_json(self.config_paths()[0], updater)
