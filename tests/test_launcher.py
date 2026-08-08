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
