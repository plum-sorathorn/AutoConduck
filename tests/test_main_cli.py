import sys
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from autoconduck import __version__
from autoconduck.config import Config
from autoconduck import main as cli


def test_version_prints_version_and_exits_zero(capsys):
    with patch.object(sys, "argv", ["autoconduck", "--version"]):
        assert cli.main() is None
    assert capsys.readouterr().out.strip() == __version__


def test_start_headless_port_and_host_dispatch_without_binding():
    cfg = Config(port=11434, log_level="warning")
    args = MagicMock(port=12345, host="0.0.0.0", headless=True)
    with patch.object(cli, "load_config", return_value=cfg), \
         patch.object(cli, "_find_free_port", side_effect=lambda port: port), \
         patch.object(cli, "_run_proxy") as run_proxy:
        with patch.object(sys, "argv", ["autoconduck", "start", "--headless", "--port", "12345", "--host", "0.0.0.0"]):
            cli.main()
    run_proxy.assert_called_once_with(12345, "warning", "0.0.0.0")


def test_edit_and_uninstall_force_dispatch():
    with patch.object(cli, "cmd_edit") as edit, patch.object(sys, "argv", ["autoconduck", "edit"]):
        cli.main()
    edit.assert_called_once()

    with patch.object(cli, "cmd_uninstall") as uninstall, patch.object(sys, "argv", ["autoconduck", "uninstall", "--force"]):
        cli.main()
    assert uninstall.call_args.args[0].force is True


def test_no_args_falls_back_to_headless_when_tui_is_unavailable():
    cfg = Config(port=11434, models=[])
    with patch.object(cli, "home_dir", return_value=MagicMock(exists=lambda: False)), \
         patch.object(cli, "load_config", return_value=cfg), \
         patch.object(cli, "_find_free_port", return_value=12000), \
         patch.object(cli, "_run_proxy") as run_proxy, \
         patch.dict(sys.modules, {"autoconduck.tui": None}):
        with patch.object(sys, "argv", ["autoconduck"]):
            cli.main()
    run_proxy.assert_called_once_with(12000, cfg.log_level)


def test_claude_flag_dispatches_to_claude_code():
    with patch.object(cli, "cmd_launch_agent", return_value=0) as launch:
        with patch.object(sys, "argv", ["autoconduck", "--claude"]):
            with patch("builtins.print"):
                try:
                    cli.main()
                except SystemExit as exc:
                    assert exc.code == 0
    launch.assert_called_once_with("claude_code")


def test_opencode_flag_dispatches_to_opencode():
    with patch.object(cli, "cmd_launch_agent", return_value=0) as launch:
        with patch.object(sys, "argv", ["autoconduck", "--opencode"]):
            try:
                cli.main()
            except SystemExit as exc:
                assert exc.code == 0
    launch.assert_called_once_with("opencode")


def test_claude_and_opencode_flags_exit_two():
    with patch.object(sys, "argv", ["autoconduck", "--claude", "--opencode"]):
        with patch("builtins.print"):
            try:
                cli.main()
            except SystemExit as exc:
                assert exc.code == 2


def test_cmd_launch_agent_uses_real_binary_env_and_releases_in_order():
    cfg = Config(port=11434, pseudo_model="autoconduck")
    adapter = SimpleNamespace(id="claude_code", binary_name="claude", patch=lambda cfg: None)
    events = []
    result = SimpleNamespace(returncode=0)

    def ensure(port):
        events.append(("ensure", port))

    def run(command, **kwargs):
        events.append(("run", command, kwargs))
        return result

    with patch.object(cli, "load_config", return_value=cfg), \
         patch("autoconduck.agents.all_adapters", return_value=[adapter]), \
         patch("autoconduck.launcher.ensure_server", side_effect=ensure), \
         patch("autoconduck.launcher.real_binary_path", return_value="C:\\fake\\claude.exe"), \
         patch("autoconduck.launcher.release_server", side_effect=lambda port: events.append(("release", port))), \
         patch("autoconduck.launcher._claude_env", return_value={
             "ANTHROPIC_BASE_URL": "http://127.0.0.1:11434",
             "ANTHROPIC_AUTH_TOKEN": "autoconduck-local",
             "ANTHROPIC_MODEL": "autoconduck",
             "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
         }) as claude_env, \
         patch.object(cli.subprocess, "run", side_effect=run):
        assert cli.cmd_launch_agent("claude_code") == 0

    assert [event[0] for event in events] == ["ensure", "run", "release"]
    assert events[1][1] == ["C:\\fake\\claude.exe"]
    env = events[1][2]["env"]
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:11434"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "autoconduck-local"
    assert env["ANTHROPIC_MODEL"] == "autoconduck"
    assert env["CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS"] == "1"
    claude_env.assert_called_once_with(11434, "autoconduck")


def test_cmd_launch_agent_nonzero_exit_still_releases():
    cfg = Config(port=11434)
    adapter = SimpleNamespace(id="opencode", binary_name="opencode", patch=lambda cfg: None)
    with patch.object(cli, "load_config", return_value=cfg), \
         patch("autoconduck.agents.all_adapters", return_value=[adapter]), \
         patch("autoconduck.launcher.ensure_server"), \
         patch("autoconduck.launcher.real_binary_path", return_value="C:\\fake\\opencode.exe"), \
         patch("autoconduck.launcher.release_server") as release, \
         patch.object(cli.subprocess, "run", return_value=SimpleNamespace(returncode=3)):
        assert cli.cmd_launch_agent("opencode") == 3
    release.assert_called_once_with(11434)


def test_cmd_launch_agent_missing_binary_returns_one_and_releases():
    cfg = Config(port=11434)
    adapter = SimpleNamespace(id="claude_code", binary_name="claude", patch=lambda cfg: None)
    with patch.object(cli, "load_config", return_value=cfg), \
         patch("autoconduck.agents.all_adapters", return_value=[adapter]), \
         patch("autoconduck.launcher.ensure_server"), \
         patch("autoconduck.launcher.real_binary_path", return_value=None), \
         patch.object(cli.shutil, "which", return_value=None), \
         patch("autoconduck.launcher.release_server") as release:
        assert cli.cmd_launch_agent("claude_code") == 1
    release.assert_called_once_with(11434)


def test_update_dry_run_for_uv_tool(capsys):
    args = SimpleNamespace(dry_run=True)
    with patch("autoconduck.update.detect_install_method", return_value="uv-tool"), \
         patch("autoconduck.update.upgrade_command", return_value="uv tool upgrade autoconduck"):
        cli.cmd_update(args)
    output = capsys.readouterr().out
    assert "Current version:" in output
    assert "Would run: uv tool upgrade autoconduck" in output


def test_purge_home_dir_removes_state(tmp_path):
    target = tmp_path / "home"
    (target / "run").mkdir(parents=True)
    (target / "run" / "state.json").write_text("{}")
    cli.purge_home_dir(target)
    assert not target.exists()


def test_purge_home_dir_refuses_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "state.json").write_text("{}")
    cli.purge_home_dir(tmp_path)
    assert (tmp_path / "state.json").exists()
