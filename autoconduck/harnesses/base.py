from __future__ import annotations

import shutil
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from ..config import Config, home_dir, backups_dir


BEGIN_MARKER = "# BEGIN AUTOCONDUCK — do not edit between these markers"
END_MARKER = "# END AUTOCONDUCK"


class BaseAdapter(ABC):
    binary_name: str | None = None
    id: str = "base"
    display_name: str = "Base"

    @abstractmethod
    def detect(self) -> bool:
        ...

    @abstractmethod
    def config_paths(self) -> List[Path]:
        ...

    def backup(self, path: Path) -> Path | None:
        if not path.exists():
            return None
        dest_dir = backups_dir(self.id)
        dest_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        # keep last 5
        dest = dest_dir / f"{ts}.bak"
        suffix = 1
        while dest.exists():
            dest = dest_dir / f"{ts}-{suffix}.bak"
            suffix += 1
        try:
            data = path.read_bytes()
            dest.write_bytes(data)
            # prune old
            files = sorted(dest_dir.glob("*.bak"))
            while len(files) > 5:
                try:
                    files[0].unlink()
                except Exception:
                    break
                files = files[1:]
            # also store mapping of which source file this backup came from
            meta = dest.with_suffix(".meta")
            meta.write_text(str(path), encoding="utf-8")
            return dest
        except Exception:
            return None

    def install_features(self) -> list[str]:
        """Check and install any agent-specific plugins/extensions/features."""
        return []

    @abstractmethod
    def patch(self, config: Config, port: int | None = None) -> None:
        ...

    def revert(self) -> None:
        # restore most recent backup per path, else remove AUTOCONDUCK block / JSON keys
        dest_dir = backups_dir(self.id)
        restored = False
        if dest_dir.exists():
            for bak in sorted(dest_dir.glob("*.bak"), reverse=True):
                meta = bak.with_suffix(".meta")
                try:
                    src_str = meta.read_text(encoding="utf-8").strip() if meta.exists() else ""
                    if src_str:
                        src = Path(src_str)
                        src.parent.mkdir(parents=True, exist_ok=True)
                        src.write_bytes(bak.read_bytes())
                        restored = True
                        break
                    # fallback: restore to first config path
                    paths = self.config_paths()
                    if paths:
                        paths[0].parent.mkdir(parents=True, exist_ok=True)
                        paths[0].write_bytes(bak.read_bytes())
                        restored = True
                        break
                except Exception:
                    continue
        if restored:
            return

        # no backup: strip blocks from text files, or clean autoconduck key from JSON files
        import json
        for p in self.config_paths():
            try:
                if not p.exists():
                    continue
                raw = p.read_text(encoding="utf-8")
                if p.suffix in (".json",):
                    try:
                        data = json.loads(raw)
                        if isinstance(data, dict):
                            data.pop("autoconduck", None)
                            if "providers" in data and isinstance(data["providers"], dict):
                                data["providers"].pop("autoconduck", None)
                            if "provider" in data and isinstance(data["provider"], dict):
                                data["provider"].pop("autoconduck", None)
                            p.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
                            continue
                    except Exception:
                        pass
                self._strip_block(p)
            except Exception:
                continue

    def validate(self) -> bool:
        # default: at least one config path exists
        return any(p.exists() for p in self.config_paths())

    # helpers for delimited blocks (text files)
    def _strip_block(self, path: Path) -> None:
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            return
        lines = text.splitlines()
        out: list[str] = []
        inside = False
        for line in lines:
            if BEGIN_MARKER in line:
                inside = True
                continue
            if END_MARKER in line:
                inside = False
                continue
            if not inside:
                out.append(line)
        path.write_text("\n".join(out) + ("\n" if out else ""), encoding="utf-8")

    def _upsert_block(self, path: Path, block_content: str) -> None:
        # ensure delimited block exists with content
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = ""
        if path.exists():
            try:
                existing = path.read_text(encoding="utf-8")
            except Exception:
                existing = ""
            # backup before write
            self.backup(path)
            # remove old block
            lines = existing.splitlines()
            out: list[str] = []
            inside = False
            for line in lines:
                if BEGIN_MARKER in line:
                    inside = True
                    continue
                if END_MARKER in line:
                    inside = False
                    continue
                if not inside:
                    out.append(line)
            existing = "\n".join(out)
        block = f"{BEGIN_MARKER}\n{block_content.rstrip()}\n{END_MARKER}\n"
        new_text = (existing.rstrip() + "\n\n" + block) if existing.strip() else block
        path.write_text(new_text, encoding="utf-8")

    def _patch_json(self, path: Path, updater) -> None:
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        if path.exists():
            try:
                raw = path.read_text(encoding="utf-8").strip()
                if raw:
                    data = json.loads(raw)
            except Exception:
                # backup corrupted
                self.backup(path)
                data = {}
            else:
                self.backup(path)
        # updater mutates dict in place, should handle autoconduck namespace
        updater(data)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
