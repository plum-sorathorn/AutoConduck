import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from autoconduck import main as cli
from autoconduck.agents.pi import PiAdapter
from autoconduck.config import Config


@pytest.fixture()
def pi_dir(tmp_path, monkeypatch):
    """Point Pi's agent dir at an isolated tmp path for the whole test."""
    agent_dir = tmp_path / "pi-agent"
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(agent_dir))
    return agent_dir


def test_pi_adapter_patch_writes_extension_file(pi_dir):
    adapter = PiAdapter()
    cfg = Config()

    adapter.patch(cfg)

    extension_file = pi_dir / "extensions" / "autoconduck.ts"
    assert extension_file.exists()

    extension_content = extension_file.read_text(encoding="utf-8")
    assert 'pi.registerProvider("autoconduck"' in extension_content
    assert 'baseUrl: "http://127.0.0.1:11434/v1"' in extension_content
    assert '"id": "autoconduck"' in extension_content
    assert '"id": "autoconduck-budget"' in extension_content
    assert '"id": "autoconduck-expensive"' in extension_content
    assert '"contextWindow": 1000000' in extension_content

    settings_path = pi_dir / "settings.json"
    data = json.loads(settings_path.read_text(encoding="utf-8"))

    assert data.get("defaultProvider") == "autoconduck"
    assert data.get("defaultModel") == "autoconduck"
    assert "providers" not in data or "autoconduck" not in data.get("providers", {})


def test_pi_adapter_custom_context_window(pi_dir):
    adapter = PiAdapter()
    cfg = Config()
    cfg.pi.context_window = 2000000
    adapter.patch(cfg)

    extension_file = pi_dir / "extensions" / "autoconduck.ts"
    extension_content = extension_file.read_text(encoding="utf-8")
    assert '"contextWindow": 2000000' in extension_content


def test_pi_adapter_patch_with_custom_model(pi_dir):
    adapter = PiAdapter()
    cfg = Config()
    cfg.pi.model = "autoconduck-budget"

    adapter.patch(cfg)

    extension_file = pi_dir / "extensions" / "autoconduck.ts"
    extension_content = extension_file.read_text(encoding="utf-8")
    # Extension still registers all three pseudo-models regardless of default.
    assert '"id": "autoconduck"' in extension_content
    assert '"id": "autoconduck-budget"' in extension_content
    assert '"id": "autoconduck-expensive"' in extension_content

    settings_path = pi_dir / "settings.json"
    data = json.loads(settings_path.read_text(encoding="utf-8"))

    assert data.get("defaultModel") == "autoconduck-budget"
    assert "providers" not in data or "autoconduck" not in data.get("providers", {})


def test_pi_adapter_patch_preserves_unrelated_settings(pi_dir):
    settings_path = pi_dir / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    initial_data = {
        "packages": ["python", "javascript"],
        "theme": "dark",
        "other_setting": "value",
    }
    settings_path.write_text(json.dumps(initial_data, indent=2), encoding="utf-8")

    adapter = PiAdapter()
    cfg = Config()

    adapter.patch(cfg)

    extension_file = pi_dir / "extensions" / "autoconduck.ts"
    assert extension_file.exists()

    data = json.loads(settings_path.read_text(encoding="utf-8"))

    assert data.get("packages") == ["python", "javascript"]
    assert data.get("theme") == "dark"
    assert data.get("other_setting") == "value"
    assert data.get("defaultProvider") == "autoconduck"
    assert data.get("defaultModel") == "autoconduck"
    assert "providers" not in data or "autoconduck" not in data.get("providers", {})


def test_pi_adapter_patch_with_custom_port(pi_dir):
    adapter = PiAdapter()
    cfg = Config()

    adapter.patch(cfg, port=8080)

    extension_file = pi_dir / "extensions" / "autoconduck.ts"
    extension_content = extension_file.read_text(encoding="utf-8")
    assert 'baseUrl: "http://127.0.0.1:8080/v1"' in extension_content

    settings_path = pi_dir / "settings.json"
    data = json.loads(settings_path.read_text(encoding="utf-8"))

    assert data.get("defaultProvider") == "autoconduck"
    assert data.get("defaultModel") == "autoconduck"
    assert "providers" not in data or "autoconduck" not in data.get("providers", {})


def test_pi_adapter_patch_with_pi_coding_agent_dir(pi_dir):
    adapter = PiAdapter()
    settings_path = adapter.config_paths()[0]

    cfg = Config()
    adapter.patch(cfg)

    extension_file = pi_dir / "extensions" / "autoconduck.ts"
    assert extension_file.exists()

    data = json.loads(settings_path.read_text(encoding="utf-8"))
    assert data.get("defaultProvider") == "autoconduck"


def test_pi_adapter_revert_removes_provider_block(pi_dir):
    settings_path = pi_dir / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)

    # Legacy shape a previous (dead-provider-block) version might have written.
    initial_data = {
        "packages": ["python", "javascript"],
        "theme": "dark",
        "other_setting": "value",
        "defaultProvider": "autoconduck",
        "defaultModel": "autoconduck",
        "providers": {
            "autoconduck": {
                "baseUrl": "http://127.0.0.1:11434/v1",
                "apiKey": "autoconduck-local",
                "api": "openai-completions",
                "models": [{"id": "autoconduck"}],
            }
        },
        "autoconduck": {"provider": "autoconduck", "model": "autoconduck"},
    }
    settings_path.write_text(json.dumps(initial_data, indent=2), encoding="utf-8")

    adapter = PiAdapter()

    extension_file = pi_dir / "extensions" / "autoconduck.ts"
    extension_file.parent.mkdir(parents=True, exist_ok=True)
    extension_file.write_text(
        "// AutoConduck-managed provider registration\n"
        'import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";\n'
        "\nexport default function (pi: ExtensionAPI) {\n"
        '  pi.registerProvider("autoconduck", { name: "AutoConduck" });\n'
        "}\n",
        encoding="utf-8",
    )

    adapter.revert()

    assert not extension_file.exists()

    data = json.loads(settings_path.read_text(encoding="utf-8"))

    assert "defaultProvider" not in data
    assert "defaultModel" not in data
    assert "providers" not in data or "autoconduck" not in data.get("providers", {})
    assert "autoconduck" not in data
    assert data.get("packages") == ["python", "javascript"]
    assert data.get("theme") == "dark"
    assert data.get("other_setting") == "value"


def test_pi_settings_default_provider():
    cfg = Config()
    assert cfg.pi.provider == "autoconduck"


def test_config_default_pseudo_model():
    cfg = Config()
    assert cfg.pseudo_model == "autoconduck"


def test_cli_pi_flag_parser():
    # Test that the --pi flag dispatches to pi agent (via `start` subcommand)
    with patch.object(cli, "cmd_launch_agent", return_value=0) as launch:
        with patch.object(sys, "argv", ["autoconduck", "start", "--pi"]):
            try:
                cli.main()
            except SystemExit as exc:
                assert exc.code == 0
    launch.assert_called_once_with("pi", new_terminal=None)
