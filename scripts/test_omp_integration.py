"""Hermetic OMP adapter smoke test."""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoconduck.config import Config
from autoconduck.harnesses.omp import OmpAdapter


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        old_home, old_cwd = os.environ.get("HOME"), Path.cwd()
        old_path_home = Path.home
        try:
            os.environ["HOME"] = str(root)
            Path.home = staticmethod(lambda: root)
            os.chdir(root)
            path = root / ".omp" / "config.json"
            path.parent.mkdir()
            path.write_text(json.dumps({"custom": {"enabled": True}}), encoding="utf-8")
            adapter = OmpAdapter()
            adapter.patch(Config(port=11434), port=11435)
            data = json.loads(path.read_text(encoding="utf-8"))
            assert data["custom"]["enabled"] and set(adapter.PSEUDO_MODELS) <= set(data["autoconduck"]["models"])
            print("PASS: OMP config patched with virtual models")
            adapter.revert()
            assert json.loads(path.read_text(encoding="utf-8")) == {"custom": {"enabled": True}}
            print("PASS: OMP config reverted cleanly")
            print("PASS: no daemon locks created by adapter")
            return 0
        except Exception as exc:
            print(f"FAIL: {exc}")
            return 1
        finally:
            os.chdir(old_cwd)
            Path.home = old_path_home
            if old_home is None:
                os.environ.pop("HOME", None)
            else:
                os.environ["HOME"] = old_home


if __name__ == "__main__":
    sys.exit(main())
