import json
from pathlib import Path

from autoconduck.config import Config
from autoconduck.harnesses import all_adapters
from autoconduck.harnesses.omp import OmpAdapter


def test_omp_identity_and_detection(monkeypatch):
    adapter = OmpAdapter()
    assert adapter.id == "omp"
    assert adapter.binary_name == "omp"
    monkeypatch.setattr("autoconduck.harnesses.omp.shutil.which", lambda _: None)
    monkeypatch.setattr(adapter, "config_paths", lambda: [])
    assert adapter.detect() is False


def test_omp_config_paths_are_home_and_local(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    paths = OmpAdapter().config_paths()
    assert paths[:2] == [tmp_path / ".omp" / "config.json", tmp_path / ".omp" / "plugins"]
    assert paths[2:] == [Path.cwd() / ".omprc", Path.cwd() / ".omp.json"]


def test_omp_patch_preserves_custom_json_and_revert_restores(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    config_path = tmp_path / ".omp" / "config.json"
    config_path.parent.mkdir()
    original = {"preset": {"name": "custom", "temperature": 0.2}}
    config_path.write_text(json.dumps(original), encoding="utf-8")

    adapter = OmpAdapter()
    adapter.patch(Config(port=11434), port=12345)
    patched = json.loads(config_path.read_text(encoding="utf-8"))
    assert patched["preset"] == original["preset"]
    assert patched["autoconduck"]["base_url"].endswith(":12345/v1")
    assert patched["autoconduck"]["models"] == list(adapter.PSEUDO_MODELS)

    adapter.revert()
    assert json.loads(config_path.read_text(encoding="utf-8")) == original


def test_all_adapters_includes_omp():
    assert any(adapter.id == "omp" for adapter in all_adapters())
