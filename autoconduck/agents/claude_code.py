from __future__ import annotations
import shutil
from pathlib import Path
from .base import BaseAdapter
from ..config import Config

class ClaudeCodeAdapter(BaseAdapter):
    id = "claude_code"
    display_name = "Claude Code"

    def detect(self) -> bool:
        if shutil.which("claude") is not None:
            return True
        return any(p.exists() for p in self.config_paths())

    def config_paths(self) -> list[Path]:
        home = Path.home()
        return [
            home / ".config" / "claude" / "settings.json",
            home / ".claude.json",
            home / ".claude" / "settings.json",
        ]

    def patch(self, config: Config) -> None:
        endpoint = f"http://127.0.0.1:{config.port}/v1"
        def updater(data: dict):
            # Merge autoconduck namespace
            data.setdefault("autoconduck", {})
            data["autoconduck"]["api_base"] = endpoint
            data["autoconduck"]["models"] = ["autoconduck","autoconduck-budget","autoconduck-expensive"]
            # also set env override if present
            env = data.setdefault("env", {})
            # preserve existing
        # patch primary path
        paths = self.config_paths()
        target = paths[0]
        # if any existing path exists, patch that one; else create first
        for p in paths:
            if p.exists():
                target = p
                break
        self._patch_json(target, updater)

    def revert(self) -> None:
        super().revert()
        # also remove autoconduck key from json if stripping not enough
        for p in self.config_paths():
            if p.exists():
                try:
                    import json
                    data = json.loads(p.read_text(encoding="utf-8"))
                    if "autoconduck" in data:
                        data.pop("autoconduck", None)
                        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
                except Exception:
                    pass
