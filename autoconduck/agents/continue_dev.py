from __future__ import annotations

from pathlib import Path

from .base import BaseAdapter
from ..config import Config


class ContinueDevAdapter(BaseAdapter):
    binary_name = "continue"
    id = "continue_dev"
    display_name = "Continue"

    def detect(self) -> bool:
        return any(path.exists() for path in self.config_paths())

    def config_paths(self) -> list[Path]:
        return [Path.home() / ".continue" / "config.json"]

    def patch(self, config: Config, port: int | None = None) -> None:
        endpoint = f"http://127.0.0.1:{port if port is not None else config.port}/v1"

        def updater(data: dict) -> None:
            models = data.setdefault("models", [])
            if not isinstance(models, list):
                models = []
                data["models"] = models
            names = ("autoconduck", "autoconduck-budget", "autoconduck-expensive")
            models[:] = [model for model in models if not isinstance(model, dict) or model.get("title") not in names]
            for model_id in names:
                models.append({"title": model_id, "provider": "openai", "model": model_id, "apiBase": endpoint})

        self._patch_json(self.config_paths()[0], updater)
