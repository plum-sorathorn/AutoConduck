from pathlib import Path

import yaml

from autoconduck.config import Config
from autoconduck.harnesses import all_adapters
from autoconduck.harnesses.omp import OmpAdapter


def test_omp_identity_and_detection(monkeypatch):
    adapter = OmpAdapter()
    assert adapter.id == "omp"
    assert adapter.binary_name == "omp"
    monkeypatch.setattr("autoconduck.harnesses.omp.shutil.which", lambda _: None)
    monkeypatch.setattr(adapter, "config_paths", list)
    assert adapter.detect() is False


def test_omp_config_paths_are_home_and_local(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    paths = OmpAdapter().config_paths()
    assert paths == [
        tmp_path / ".omp" / "agent" / "models.yml",
        tmp_path / ".omp" / "agent" / "models.yaml",
        Path.cwd() / ".omp" / "config.yml",
    ]


def test_omp_patch_writes_yaml_provider_and_model_role_and_revert_restores(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("AUTOCONDUCK_HOME", str(tmp_path / "autoconduck"))
    config_path = tmp_path / ".omp" / "agent" / "models.yml"
    settings_path = tmp_path / ".omp" / "agent" / "config.yml"
    config_path.parent.mkdir(parents=True)
    models_original = {"preset": {"name": "custom", "temperature": 0.2}}
    settings_original = {"theme": "custom"}
    config_path.write_text(yaml.safe_dump(models_original), encoding="utf-8")
    settings_path.write_text(yaml.safe_dump(settings_original), encoding="utf-8")

    adapter = OmpAdapter()
    adapter.patch(Config(port=11434), port=12345)
    patched_models = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    patched_settings = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    provider = patched_models["providers"][adapter.provider_name]
    assert patched_models["preset"] == models_original["preset"]
    assert provider["baseUrl"].endswith(":12345/v1")
    assert [model["id"] for model in provider["models"]] == list(adapter.PSEUDO_MODELS)
    assert patched_settings["theme"] == settings_original["theme"]
    assert patched_settings["modelRoles"]["default"] == "autoconduck/balanced"

    adapter.revert()
    assert yaml.safe_load(config_path.read_text(encoding="utf-8")) == models_original
    assert (
        yaml.safe_load(settings_path.read_text(encoding="utf-8")) == settings_original
    )


def test_all_adapters_includes_omp():
    assert any(adapter.id == "omp" for adapter in all_adapters())
