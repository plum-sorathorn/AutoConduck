from __future__ import annotations

from pathlib import Path

from .base import BaseAdapter
from ..config import Config


class KiloCodeAdapter(BaseAdapter):
    binary_name = "kilocode"
    id = "kilocode"
    display_name = "Kilo Code"

    def detect(self) -> bool:
        return any(path.exists() for path in self.config_paths())

    def config_paths(self) -> list[Path]:
        return [Path.home() / ".kilocode" / "config.json", Path.cwd() / "kilo-config.json"]

    def patch(self, config: Config, port: int | None = None) -> None:
        endpoint = f"http://127.0.0.1:{port if port is not None else config.port}/v1"

        def updater(data: dict) -> None:
            managed = data.setdefault("autoconduck", {})
            managed["api_base"] = endpoint
            managed["models"] = ["autoconduck", "autoconduck-budget", "autoconduck-expensive"]

        target = next((path for path in self.config_paths() if path.exists()), self.config_paths()[0])
        self._patch_json(target, updater)
