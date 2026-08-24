"""CLI command parsing, shortcuts, lifecycle (reset/update/uninstall), and packaging unit tests."""
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from autoconduck import __version__
from autoconduck import main as cli
from autoconduck import update
from autoconduck.config import Config


def test_cli_version_flag(capsys):
    with patch.object(sys, "argv", ["autoconduck", "--version"]):
        cli.main()
    assert capsys.readouterr().out.strip() == __version__


def test_cli_agent_shortcuts():
    with patch.object(cli, "cmd_launch_agent", return_value=0) as launch:
        with patch.object(sys, "argv", ["autoconduck", "start", "--claude"]):
            try:
                cli.main()
            except SystemExit as exc:
                assert exc.code == 0
        launch.assert_called_once_with("claude_code", new_terminal=None)

    with patch.object(cli, "cmd_launch_agent", return_value=0) as launch:
        with patch.object(sys, "argv", ["autoconduck", "start", "--pi"]):
            try:
                cli.main()
            except SystemExit as exc:
                assert exc.code == 0
        launch.assert_called_once_with("pi", new_terminal=None)

    with patch.object(cli, "cmd_launch_agent", return_value=0) as launch:
        with patch.object(sys, "argv", ["autoconduck", "start", "--opencode"]):
            try:
                cli.main()
            except SystemExit as exc:
                assert exc.code == 0
        launch.assert_called_once_with("opencode", new_terminal=None)


def test_cli_reset_command(capsys, monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "home_dir", lambda: tmp_path / ".autoconduck")
    args = MagicMock(force=True)
    with patch("autoconduck.harnesses.all_adapters", return_value=[]):
        cli.cmd_reset(args)
    captured = capsys.readouterr().out
    assert "Coding agents reverted:" in captured
    assert "AutoConduck state purged" in captured


def test_update_upgrade_commands():
    assert update.upgrade_command("uv-tool") == "uv tool upgrade --reinstall autoconduck"
    assert update.upgrade_command("npm") == "npm install -g autoconduck@latest"
    assert update.upgrade_command("pip") == "pip install --force-reinstall --upgrade autoconduck"


def test_packaging_matrix():
    root = Path(__file__).resolve().parents[1]
    build_path = root / "npm-packaging" / "build.py"
    if build_path.exists():
        spec = importlib.util.spec_from_file_location("autoconduck_packaging_build", build_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert len(module.MATRIX) == 5
