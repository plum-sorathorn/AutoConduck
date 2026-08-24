from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import List

from .base import BaseAdapter
from ..config import Config, backups_dir


class ContinueDevAdapter(BaseAdapter):
    binary_name = "continue"
    id = "continue_dev"
    display_name = "Continue"

    def detect(self) -> bool:
        if shutil.which(self.binary_name) is not None:
            return True
        return any(path.exists() for path in self.config_paths())

    def config_paths(self) -> List[Path]:
        home = Path.home()
        return [
            home / ".continue" / "config.json",
            home / ".continue" / "config.yaml",
        ]

    def patch(self, config: Config, port: int | None = None) -> None:
        effective_port = int(port if port is not None else getattr(config, "port", 11434))
        endpoint = f"http://127.0.0.1:{effective_port}/v1"

        def updater(data: dict) -> None:
            models = data.setdefault("models", [])
            if not isinstance(models, list):
                models = []
                data["models"] = models
            names = ("autoconduck", "autoconduck-budget", "autoconduck-expensive")
            models[:] = [model for model in models if not isinstance(model, dict) or model.get("title") not in names]
            for model_id in names:
                models.append({
                    "title": model_id,
                    "provider": "openai",
                    "model": model_id,
                    "apiBase": endpoint,
                    "apiKey": "autoconduck-local",
                })
            data.setdefault("autoconduck", {})["managed_models"] = list(names)

        target = next((path for path in self.config_paths() if path.exists()), self.config_paths()[0])
        self._patch_json(target, updater)

    def revert(self) -> None:
        dest_dir = backups_dir(self.id)
        if dest_dir.exists():
            for bak in sorted(dest_dir.glob("*.bak"), reverse=True):
                meta = bak.with_suffix(".meta")
                try:
                    src_str = meta.read_text(encoding="utf-8").strip() if meta.exists() else ""
                    if src_str:
                        src = Path(src_str)
                        src.parent.mkdir(parents=True, exist_ok=True)
                        src.write_bytes(bak.read_bytes())
                        return
                except Exception:
                    continue

        names = {"autoconduck", "autoconduck-budget", "autoconduck-expensive"}
        for p in self.config_paths():
            if not p.exists() or p.suffix != ".json":
                continue
            try:
                raw = p.read_text(encoding="utf-8")
                data = json.loads(raw)
                if not isinstance(data, dict):
                    continue
                data.pop("autoconduck", None)
                models = data.get("models")
                if isinstance(models, list):
                    data["models"] = [m for m in models if not isinstance(m, dict) or m.get("title") not in names]
                p.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            except Exception:
                continue

