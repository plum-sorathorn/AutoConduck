"""Coding agent adapters unit tests for Pi, Claude Code, OpenCode, and legacy tools."""
import json
from pathlib import Path
import pytest

from autoconduck.agents import all_adapters
from autoconduck.agents.claude_code import ClaudeCodeAdapter
from autoconduck.agents.opencode import OpenCodeAdapter
from autoconduck.agents.pi import PiAdapter
from autoconduck.config import Config


def test_all_adapters_registered():
    adapters = all_adapters()
    ids = {a.id for a in adapters}
    assert "claude_code" in ids
    assert "pi" in ids
    assert "opencode" in ids
    assert "aider" in ids
    assert "cursor" in ids


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
    adapter.patch(cfg)

    # Check that settings were patched
    data = json.loads(settings_file.read_text(encoding="utf-8"))
    assert "ANTHROPIC_BASE_URL" in data["env"]
    assert "http://127.0.0.1:11434" in data["env"]["ANTHROPIC_BASE_URL"]

    # Revert restores original state
    adapter.revert()
    data_reverted = json.loads(settings_file.read_text(encoding="utf-8"))
    assert "ANTHROPIC_BASE_URL" not in data_reverted.get("env", {})
    assert data_reverted.get("env", {}).get("EXISTING") == "val"


def test_pi_adapter_patch_and_revert(tmp_path, monkeypatch):
    pi_dir = tmp_path / ".pi" / "agent"
    pi_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(pi_dir))

    adapter = PiAdapter()
    cfg = Config(port=11434)
    adapter.patch(cfg)

    ext = pi_dir / "extensions" / "autoconduck.ts"
    assert ext.exists()
    assert "autoconduck" in ext.read_text(encoding="utf-8")

    adapter.revert()
    assert not ext.exists()


def test_opencode_adapter_patch_and_revert(tmp_path, monkeypatch):
    opencode_cfg = tmp_path / "opencode.json"
    opencode_cfg.write_text(json.dumps({"providers": {}}), encoding="utf-8")
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    adapter = OpenCodeAdapter()
    cfg = Config(port=11434)
    adapter.patch(cfg)

    data = json.loads(opencode_cfg.read_text(encoding="utf-8"))
    assert "autoconduck" in data.get("providers", {})

    adapter.revert()
    data_reverted = json.loads(opencode_cfg.read_text(encoding="utf-8"))
    assert "autoconduck" not in data_reverted.get("providers", {})
