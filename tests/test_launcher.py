from pathlib import Path
from unittest.mock import patch

from autoconduck import launcher


def test_scripts_contain_lifecycle_commands():
    posix = launcher.shim_script("a", "/opt/my agent")
    windows = launcher.shim_script_win("a", "/opt/my agent")
    for text in (posix, windows):
        assert "/opt/my agent" in text
        assert "ensure" in text and "release" in text
        assert "11434" in text


def test_server_alive_handles_failure():
    with patch("autoconduck.launcher.urlopen", side_effect=OSError):
        assert launcher.server_alive() is False


def test_path_block_is_idempotent_and_removable(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(launcher.os, "name", "posix")
    bashrc = tmp_path / ".bashrc"
    bashrc.write_text("# existing\n")
    launcher.ensure_path_entry()
    launcher.ensure_path_entry()
    assert bashrc.read_text().count("BEGIN AUTOCONDUCK PATH") == 1
    launcher.remove_path_entry()
    assert "BEGIN AUTOCONDUCK PATH" not in bashrc.read_text()


def test_claude_code_shim_sets_anthropic_env():
    posix = launcher.shim_script("claude_code", "/opt/claude")
    windows = launcher.shim_script_win("claude_code", "/opt/claude")
    for text in (posix, windows):
        assert "ANTHROPIC_BASE_URL" in text
        assert "ANTHROPIC_AUTH_TOKEN" in text
        assert "ANTHROPIC_MODEL" in text

    other_posix = launcher.shim_script("some_other_agent", "/opt/other")
    other_windows = launcher.shim_script_win("some_other_agent", "/opt/other")
    for text in (other_posix, other_windows):
        assert "ANTHROPIC_BASE_URL" not in text
