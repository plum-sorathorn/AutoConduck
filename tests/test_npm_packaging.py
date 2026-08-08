import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD_PATH = ROOT / "npm-packaging" / "build.py"


def _build_module():
    spec = importlib.util.spec_from_file_location("autoconduck_packaging_build", BUILD_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_packaging_matrix_has_all_supported_platforms():
    build = _build_module()
    assert build.MATRIX == ["darwin-arm64", "darwin-x64", "linux-x64", "linux-arm64", "win32-x64"]
    assert len(build.MATRIX) == 5


def test_packaging_version_matches_pyproject_for_existing_package():
    build = _build_module()
    platforms = [p for p in build.MATRIX if (ROOT / "npm-packaging" / f"autoconduck-{p}" / "package.json").exists()]
    assert platforms
    package = ROOT / "npm-packaging" / f"autoconduck-{platforms[0]}" / "package.json"
    assert json.loads(package.read_text(encoding="utf-8"))["version"] == build.version()


def test_shasum_matches_sha256_without_building_binaries(tmp_path):
    build = _build_module()
    before = sorted(str(path) for path in (ROOT / "npm-packaging").glob("**/bin/*"))
    artifact = tmp_path / "artifact"
    artifact.write_bytes(b"safe test artifact")
    assert build.shasum(artifact) == hashlib.sha256(b"safe test artifact").hexdigest()
    after = sorted(str(path) for path in (ROOT / "npm-packaging").glob("**/bin/*"))
    assert after == before
