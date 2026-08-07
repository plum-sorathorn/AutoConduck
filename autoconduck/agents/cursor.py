from __future__ import annotations

from pathlib import Path

from ..config import Config
from .base import BaseAdapter


class CursorAdapter(BaseAdapter):
    id = "cursor"
    display_name = "Cursor"

    def detect(self) -> bool:
        return any(p.exists() for p in self.config_paths())

    def config_paths(self) -> list[Path]:
        return [
            Path.home() / ".cursor" / "settings.json",
            Path.home() / ".config" / "cursor" / "settings.json",
        ]

    def patch(self, config: Config) -> None:
        endpoint = f"http://127.0.0.1:{config.port}/v1"

        def updater(data: dict):
            data.setdefault("autoconduck", {})["api_base"] = endpoint
            data["autoconduck"]["models"] = [
                "autoconduck",
                "autoconduck-budget",
                "autoconduck-expensive",
            ]

        target = self.config_paths()[0]
        for p in self.config_paths():
            if p.exists():
                target = p
                break
        self._patch_json(target, updater)
