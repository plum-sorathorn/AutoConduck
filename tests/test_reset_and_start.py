import sys
from unittest.mock import patch, MagicMock
from autoconduck import main as cli
from autoconduck.tui.onboarding_models import upsert_custom_models


def test_cmd_reset_reverts_and_reports(capsys, monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "home_dir", lambda: tmp_path / ".autoconduck")
    args = MagicMock(force=True)
    with patch("autoconduck.agents.all_adapters", return_value=[]):
        cli.cmd_reset(args)
    captured = capsys.readouterr().out
    assert "Coding agents reverted:" in captured
    assert "AutoConduck state purged" in captured


def test_start_claude_flag(monkeypatch):
    called = []
    monkeypatch.setattr(cli, "cmd_launch_agent", lambda agent, new_terminal=None: called.append(agent) or 0)
    with patch.object(sys, "argv", ["autoconduck", "start", "--claude"]):
        try:
            cli.main()
        except SystemExit as exc:
            assert exc.code == 0
    assert called == ["claude_code"]


def test_upsert_custom_models_supports_anthropic_base_url():
    result = upsert_custom_models(
        [],
        provider="my-custom-endpoint",
        base_url="http://localhost:8000/v1",
        api_key_env="sk-test",
        model_ids=["custom-model"],
        anthropic_base_url="http://localhost:8000/anthropic",
    )
    assert len(result) == 1
    assert result[0]["anthropic_base_url"] == "http://localhost:8000/anthropic"
