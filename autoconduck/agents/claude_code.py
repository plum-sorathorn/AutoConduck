from __future__ import annotations
import json
import shutil
from pathlib import Path
from .base import BaseAdapter
from ..config import Config


class ClaudeCodeAdapter(BaseAdapter):
    binary_name = "claude"
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
        """Leave settings.json untouched; env is injected by launcher shims.

        JSON cannot carry the BEGIN/END AUTOCONDUCK marker block required for
        managed agent configuration, so Claude Code's environment is supplied
        exclusively by the AutoConduck launcher shims.
        """

    def revert(self) -> None:
        """Remove only the legacy top-level AutoConduck namespace."""
        for p in self.config_paths():
            if p.exists():
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    if "autoconduck" in data:
                        data.pop("autoconduck", None)
                        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
                except Exception:
                    pass
