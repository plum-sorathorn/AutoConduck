from __future__ import annotations
from pathlib import Path
from .base import BaseAdapter
from ..config import Config

class KiloCodeAdapter(BaseAdapter):
    binary_name = "kilocode"
    id = "kilocode"
    display_name = "Kilo Code"
    def detect(self) -> bool:
        return any(p.exists() for p in self.config_paths())
    def config_paths(self) -> list[Path]:
        return [Path.home() / ".kilocode" / "config.json", Path.cwd() / "kilo-config.json"]
    def patch(self, config: Config) -> None:
        endpoint = f"http://127.0.0.1:{config.port}/v1"
        def updater(data: dict):
            data.setdefault("autoconduck", {})["api_base"] = endpoint
            data["autoconduck"]["models"] = ["autoconduck","autoconduck-budget","autoconduck-expensive"]
        target = self.config_paths()[0]
        for p in self.config_paths():
            if p.exists():
                target=p; break
        self._patch_json(target, updater)
