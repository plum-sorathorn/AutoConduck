"""Coding agent adapters unit tests for Pi, Claude Code, OpenCode, and all supported tools."""
import json
from pathlib import Path
import pytest

from autoconduck.agents import all_adapters, binary_name_for
from autoconduck.agents.aider import AiderAdapter
from autoconduck.agents.claude_code import ClaudeCodeAdapter
from autoconduck.agents.continue_dev import ContinueDevAdapter
from autoconduck.agents.cursor import CursorAdapter
from autoconduck.agents.generic_openai import GenericOpenAIAdapter
from autoconduck.agents.kilocode import KiloCodeAdapter
from autoconduck.agents.opencode import OpenCodeAdapter
from autoconduck.agents.pi import PiAdapter
from autoconduck.cli.cli_launch import resolve_agent_ids
from autoconduck.config import Config
from autoconduck.launcher import shim_script, shim_script_win
from autoconduck.presets.model_presets import discover_models, curated_model_catalog
from autoconduck.presets.presets_data import PRESETS, PRESET_ORDER


def test_all_adapters_registered():
    adapters = all_adapters()
    ids = {a.id for a in adapters}
    expected = {
        "claude_code",
        "opencode",
        "pi",
        "aider",
        "cursor",
        "continue_dev",
        "kilocode",
        "generic_openai",
    }
    assert expected.issubset(ids)
    assert binary_name_for("claude_code") == "claude"
    assert binary_name_for("opencode") == "opencode"
    assert binary_name_for("pi") == "pi"


def test_claude_code_adapter_patch_and_revert(tmp_path, monkeypatch):
    settings_file = tmp_path / ".claude" / "settings.json"
    settings_file.parent.mkdir(parents=True, exist_ok=True)
    settings_file.write_text(
        json.dumps({
            "env": {"EXISTING": "val"},
            "permissions": {"allow": ["Notebook", "Task", "Read", "Write", "Edit"]},
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    adapter = ClaudeCodeAdapter()
    cfg = Config(port=11434)
    adapter.patch(cfg, port=11434)

    # Check that settings were patched
    data = json.loads(settings_file.read_text(encoding="utf-8"))
    assert "ANTHROPIC_BASE_URL" in data["env"]
    assert "http://127.0.0.1:11434" in data["env"]["ANTHROPIC_BASE_URL"]
    assert "autoconduck" in data.get("modelOverrides", {})

    # Revert restores original state
    adapter.revert()
    data_reverted = json.loads(settings_file.read_text(encoding="utf-8"))
    assert "ANTHROPIC_BASE_URL" not in data_reverted.get("env", {})
    assert data_reverted.get("env", {}).get("EXISTING") == "val"


def test_pi_adapter_patch_install_features_and_revert(tmp_path, monkeypatch):
    pi_dir = tmp_path / ".pi" / "agent"
    pi_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(pi_dir))

    adapter = PiAdapter()
    cfg = Config(port=11434)
    adapter.patch(cfg, port=11434)

    ext = pi_dir / "extensions" / "autoconduck.ts"
    assert ext.exists()
    assert "autoconduck" in ext.read_text(encoding="utf-8")

    # Check subagent feature installation in settings.json
    settings_file = pi_dir / "settings.json"
    assert settings_file.exists()
    settings_data = json.loads(settings_file.read_text(encoding="utf-8"))
    assert "npm:pi-subagents" in settings_data.get("packages", [])
    assert settings_data.get("defaultProvider") == "autoconduck"

    # Test install_features hook idempotent execution
    installed = adapter.install_features()
    assert isinstance(installed, list)

    adapter.revert()
    assert not ext.exists()
    data_reverted = json.loads(settings_file.read_text(encoding="utf-8"))
    assert data_reverted.get("defaultProvider") is None
    assert "npm:pi-subagents" not in data_reverted.get("packages", [])


def test_opencode_adapter_patch_and_revert(tmp_path, monkeypatch):
    opencode_cfg = tmp_path / "opencode.json"
    opencode_cfg.write_text(json.dumps({"providers": {}}), encoding="utf-8")
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    adapter = OpenCodeAdapter()
    cfg = Config(port=11434)
    adapter.patch(cfg, port=11434)

    data = json.loads(opencode_cfg.read_text(encoding="utf-8"))
    assert "autoconduck" in data.get("providers", {})
    assert "autoconduck" in data.get("provider", {})
    assert data.get("model") == "autoconduck/autoconduck"

    adapter.revert()
    data_reverted = json.loads(opencode_cfg.read_text(encoding="utf-8"))
    assert "autoconduck" not in data_reverted.get("providers", {})
    assert "autoconduck" not in data_reverted.get("provider", {})


def test_aider_adapter_patch_and_revert(tmp_path, monkeypatch):
    aider_cfg = tmp_path / ".aider.conf.yml"
    aider_cfg.write_text("verbose: true\n", encoding="utf-8")
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    adapter = AiderAdapter()
    cfg = Config(port=11434)
    adapter.patch(cfg, port=11434)

    text = aider_cfg.read_text(encoding="utf-8")
    assert "openai_api_base: http://127.0.0.1:11434/v1" in text
    assert "model: openai/autoconduck" in text

    adapter.revert()
    text_reverted = aider_cfg.read_text(encoding="utf-8")
    assert "openai_api_base" not in text_reverted
    assert "verbose: true" in text_reverted


def test_cursor_adapter_patch_and_revert(tmp_path, monkeypatch):
    cursor_dir = tmp_path / ".cursor"
    cursor_dir.mkdir(parents=True, exist_ok=True)
    cursor_cfg = cursor_dir / "settings.json"
    cursor_cfg.write_text(json.dumps({"editor.fontSize": 14}), encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    adapter = CursorAdapter()
    cfg = Config(port=11434)
    adapter.patch(cfg, port=11434)

    data = json.loads(cursor_cfg.read_text(encoding="utf-8"))
    assert "autoconduck" in data
    assert data.get("openai.apiBase") == "http://127.0.0.1:11434/v1"

    adapter.revert()
    data_reverted = json.loads(cursor_cfg.read_text(encoding="utf-8"))
    assert "autoconduck" not in data_reverted
    assert data_reverted.get("editor.fontSize") == 14


def test_continue_adapter_patch_and_revert(tmp_path, monkeypatch):
    continue_dir = tmp_path / ".continue"
    continue_dir.mkdir(parents=True, exist_ok=True)
    continue_cfg = continue_dir / "config.json"
    continue_cfg.write_text(json.dumps({"models": [{"title": "existing", "model": "gpt-4"}]}), encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    adapter = ContinueDevAdapter()
    cfg = Config(port=11434)
    adapter.patch(cfg, port=11434)

    data = json.loads(continue_cfg.read_text(encoding="utf-8"))
    titles = [m.get("title") for m in data.get("models", [])]
    assert "autoconduck" in titles
    assert "autoconduck-budget" in titles

    adapter.revert()
    data_reverted = json.loads(continue_cfg.read_text(encoding="utf-8"))
    reverted_titles = [m.get("title") for m in data_reverted.get("models", [])]
    assert "autoconduck" not in reverted_titles
    assert "existing" in reverted_titles


def test_kilocode_adapter_patch_and_revert(tmp_path, monkeypatch):
    kilo_cfg = tmp_path / "kilo-config.json"
    kilo_cfg.write_text(json.dumps({"user": "dev"}), encoding="utf-8")
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    adapter = KiloCodeAdapter()
    cfg = Config(port=11434)
    adapter.patch(cfg, port=11434)

    data = json.loads(kilo_cfg.read_text(encoding="utf-8"))
    assert data.get("api_base") == "http://127.0.0.1:11434/v1"

    adapter.revert()
    data_reverted = json.loads(kilo_cfg.read_text(encoding="utf-8"))
    assert "autoconduck" not in data_reverted
    assert data_reverted.get("user") == "dev"


def test_generic_openai_adapter():
    adapter = GenericOpenAIAdapter()
    assert adapter.detect() is True
    cfg = Config(port=11434)
    adapter.patch(cfg)
    adapter.revert()


def test_launcher_shims_generation():
    real_bin = "/usr/local/bin/agent_mock"
    for agent_id in ("claude_code", "opencode", "pi", "aider", "cursor", "continue_dev", "kilocode"):
        bash = shim_script(agent_id, real_bin)
        assert "ensure --port" in bash
        assert "release --port" in bash

        win = shim_script_win(agent_id, real_bin)
        assert "ensure --port" in win
        assert "release --port" in win


def test_agent_alias_resolution():
    assert resolve_agent_ids(["claude"]) == ["claude_code"]
    assert resolve_agent_ids(["open-code"]) == ["opencode"]
    assert resolve_agent_ids(["continue"]) == ["continue_dev"]
    assert resolve_agent_ids(["kilo-code"]) == ["kilocode"]
    assert resolve_agent_ids(["all"]) == [a.id for a in all_adapters()]


def test_all_provider_presets_resolve():
    for provider in ("openai", "anthropic", "google", "mistral", "deepseek", "groq", "openrouter", "together", "xai", "llmgateway", "devpass"):
        assert provider in PRESETS
        models = PRESETS[provider]
        assert len(models) >= 1
        for m in models:
            assert "id" in m
            assert "provider" in m
            assert "tier" in m
            assert "price_in" in m
            assert "price_out" in m

    discovered = discover_models(preset_keys=["openai", "anthropic", "google", "mistral", "deepseek", "groq", "openrouter", "together", "xai"])
    assert len(discovered) >= 40
    catalog = curated_model_catalog()
    assert len(catalog) >= 100

