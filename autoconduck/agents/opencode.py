from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import List

from .base import BaseAdapter
from ..config import Config, backups_dir


class OpenCodeAdapter(BaseAdapter):
    binary_name = "opencode"
    id = "opencode"
    display_name = "OpenCode"

    def detect(self) -> bool:
        if shutil.which(self.binary_name) is not None:
            return True
        return any(p.exists() for p in self.config_paths())

    def config_paths(self) -> List[Path]:
        home = Path.home()
        return [
            Path.cwd() / "opencode.json",
            Path.cwd() / "opencode.config.json",
            home / ".config" / "opencode" / "config.json",
            home / ".config" / "opencode" / "opencode.json",
            home / ".opencode.json",
            home / ".opencode" / "config.json",
        ]

    def patch(self, config: Config, port: int | None = None) -> None:
        effective_port = int(port if port is not None else getattr(config, "port", 11434))
        endpoint = f"http://127.0.0.1:{effective_port}/v1"
        pseudo_model = getattr(config, "pseudo_model", "autoconduck") or "autoconduck"

        def updater(data: dict) -> None:
            marker = data.get("autoconduck") if isinstance(data.get("autoconduck"), dict) else {}
            if "model" in data and not marker:
                marker["previous_model"] = data["model"]

            # Flat providers dictionary
            providers = data.setdefault("providers", {})
            if isinstance(providers, dict):
                ac = providers.setdefault("autoconduck", {})
                if isinstance(ac, dict):
                    ac["api_base"] = endpoint
                    ac["baseURL"] = endpoint
                    ac["apiKey"] = "autoconduck-local"
                    ac["models"] = ["autoconduck", "autoconduck-budget", "autoconduck-expensive"]

            # Structured provider dictionary
            provider = data.setdefault("provider", {})
            if isinstance(provider, dict):
                provider["autoconduck"] = {
                    "type": "openai",
                    "name": "AutoConduck",
                    "options": {
                        "baseURL": endpoint,
                        "apiKey": "autoconduck-local",
                    },
                    "models": {
                        "autoconduck": {
                            "name": "AutoConduck",
                            "limit": {"context": 1000000, "output": 16384},
                        },
                        "autoconduck-budget": {
                            "name": "AutoConduck Budget",
                            "limit": {"context": 1000000, "output": 16384},
                        },
                        "autoconduck-expensive": {
                            "name": "AutoConduck Expensive",
                            "limit": {"context": 1000000, "output": 16384},
                        },
                    },
                }

            # Managed active model
            data["model"] = f"autoconduck/{pseudo_model}"
            marker["managed"] = True
            marker["managed_model"] = f"autoconduck/{pseudo_model}"
            data["autoconduck"] = marker

        target = next((p for p in self.config_paths() if p.exists()), self.config_paths()[0])
        self._patch_json(target, updater)

    def revert(self) -> None:
        """Restore previous configuration or cleanly remove AutoConduck entries."""
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
            if not p.exists():
                continue
            try:
                raw = p.read_text(encoding="utf-8")
                data = json.loads(raw)
                if not isinstance(data, dict):
                    continue
                marker = data.pop("autoconduck", None)
                if isinstance(data.get("providers"), dict):
                    data["providers"].pop("autoconduck", None)
                    if not data["providers"]:
                        data.pop("providers", None)
                if isinstance(data.get("provider"), dict):
                    data["provider"].pop("autoconduck", None)
                    if not data["provider"]:
                        data.pop("provider", None)
                if isinstance(marker, dict) and "previous_model" in marker:
                    data["model"] = marker["previous_model"]
                elif str(data.get("model", "")).startswith("autoconduck"):
                    data.pop("model", None)
                p.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            except Exception:
                continue

