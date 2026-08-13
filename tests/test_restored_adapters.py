import json

import pytest

from autoconduck.agents.aider import AiderAdapter
from autoconduck.agents.continue_dev import ContinueDevAdapter
from autoconduck.agents.cursor import CursorAdapter
from autoconduck.agents.generic_openai import GenericOpenAIAdapter
from autoconduck.agents.kilocode import KiloCodeAdapter
from autoconduck.config import Config


PSEUDO_MODELS = ["autoconduck", "autoconduck-budget", "autoconduck-expensive"]


@pytest.fixture()
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: tmp_path))
    monkeypatch.setenv("AUTOCONDUCK_HOME", str(tmp_path / ".autoconduck"))
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.parametrize(
    ("adapter_type", "relative_path"),
    [
        (AiderAdapter, ".aider.conf.yml"),
        (CursorAdapter, ".cursor/settings.json"),
        (KiloCodeAdapter, ".kilocode/config.json"),
        (ContinueDevAdapter, ".continue/config.json"),
    ],
)
def test_file_adapters_patch_write_expected_managed_configuration(
    isolated_home, adapter_type, relative_path
):
    target = isolated_home / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if adapter_type is AiderAdapter:
        target.write_text("user_setting: keep\n", encoding="utf-8")
    elif adapter_type is ContinueDevAdapter:
        target.write_text(
            json.dumps(
                {
                    "theme": "dark",
                    "models": [
                        {"title": "user-model", "provider": "openai"},
                        {"title": "autoconduck", "provider": "old"},
                    ],
                }
            ),
            encoding="utf-8",
        )
    else:
        target.write_text(json.dumps({"user_setting": "keep"}), encoding="utf-8")

    adapter = adapter_type()
    adapter.patch(Config(), port=4321)

    if adapter_type is AiderAdapter:
        content = target.read_text(encoding="utf-8")
        assert "# BEGIN AUTOCONDUCK" in content
        assert "openai_api_base: http://127.0.0.1:4321/v1" in content
        assert content.count("# BEGIN AUTOCONDUCK") == 1
    else:
        data = json.loads(target.read_text(encoding="utf-8"))
        if adapter_type is ContinueDevAdapter:
            assert data["theme"] == "dark"
            assert [m["title"] for m in data["models"]].count("autoconduck") == 1
            managed = data["models"][-3:]
            assert [m["title"] for m in managed] == PSEUDO_MODELS
            assert all(m["apiBase"] == "http://127.0.0.1:4321/v1" for m in managed)
        else:
            assert data["user_setting"] == "keep"
            assert data["autoconduck"]["api_base"] == "http://127.0.0.1:4321/v1"
            assert data["autoconduck"]["models"] == PSEUDO_MODELS


@pytest.mark.parametrize(
    ("adapter_type", "relative_path", "original"),
    [
        (AiderAdapter, ".aider.conf.yml", "user_setting: keep\n"),
        (CursorAdapter, ".cursor/settings.json", '{"user_setting": "keep"}'),
        (KiloCodeAdapter, ".kilocode/config.json", '{"user_setting": "keep"}'),
        (
            ContinueDevAdapter,
            ".continue/config.json",
            '{"models": [{"title": "user-model"}], "theme": "dark"}',
        ),
    ],
)
def test_file_adapters_revert_restores_user_content(
    isolated_home, adapter_type, relative_path, original
):
    target = isolated_home / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(original, encoding="utf-8")

    adapter = adapter_type()
    adapter.patch(Config(), port=4321)
    adapter.revert()

    assert target.read_text(encoding="utf-8") == original
    if adapter_type is AiderAdapter:
        assert "AUTOCONDUCK" not in target.read_text(encoding="utf-8")
    else:
        assert "autoconduck" not in target.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "adapter_type",
    [AiderAdapter, CursorAdapter, KiloCodeAdapter, ContinueDevAdapter],
)
def test_file_adapters_patch_is_idempotent(isolated_home, adapter_type):
    adapter = adapter_type()
    target = adapter.config_paths()[0]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("user: keep\n" if adapter_type is AiderAdapter else "{}", encoding="utf-8")

    adapter.patch(Config(), port=4321)
    adapter.patch(Config(), port=4321)

    content = target.read_text(encoding="utf-8")
    if adapter_type is AiderAdapter:
        assert content.count("# BEGIN AUTOCONDUCK") == 1
        assert content.count("# END AUTOCONDUCK") == 1
    elif adapter_type is ContinueDevAdapter:
        models = json.loads(content)["models"]
        assert [m["title"] for m in models].count("autoconduck") == 1
        assert len(models) == 3
    else:
        assert json.loads(content)["autoconduck"]["models"] == PSEUDO_MODELS


def test_generic_openai_is_always_detected_and_does_not_write_file(tmp_path, capsys):
    adapter = GenericOpenAIAdapter()
    adapter.patch(Config(), port=4321)

    output = capsys.readouterr().out
    assert adapter.detect() is True
    assert "OPENAI_API_BASE=http://127.0.0.1:4321/v1" in output
    assert list(tmp_path.iterdir()) == []
