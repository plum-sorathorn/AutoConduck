from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import List

from .base import BaseAdapter
from ..config import Config, backups_dir


class KiloCodeAdapter(BaseAdapter):
    binary_name = "kilocode"
    id = "kilocode"
    display_name = "Kilo Code"

    def detect(self) -> bool:
        if shutil.which(self.binary_name) is not None:
            return True
        return any(path.exists() for path in self.config_paths())

    def config_paths(self) -> List[Path]:
        return [
            Path.cwd() / "kilo-config.json",
            Path.home() / ".kilocode" / "config.json",
            Path.home() / ".config" / "kilocode" / "config.json",
        ]

    def patch(self, config: Config, port: int | None = None) -> None:
        effective_port = int(port if port is not None else getattr(config, "port", 11434))
        endpoint = f"http://127.0.0.1:{effective_port}/v1"

        def updater(data: dict) -> None:
            managed = data.setdefault("autoconduck", {})
            managed["api_base"] = endpoint
            managed["models"] = ["autoconduck", "autoconduck-budget", "autoconduck-expensive"]
            data["api_base"] = endpoint
            data["api_key"] = "autoconduck-local"

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

        for p in self.config_paths():
            if not p.exists() or p.suffix != ".json":
                continue
            try:
                raw = p.read_text(encoding="utf-8")
                data = json.loads(raw)
                if not isinstance(data, dict):
                    continue
                data.pop("autoconduck", None)
                if data.get("api_key") == "autoconduck-local":
                    data.pop("api_key", None)
                if str(data.get("api_base", "")).startswith("http://127.0.0.1:"):
                    data.pop("api_base", None)
                p.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            except Exception:
                continue

