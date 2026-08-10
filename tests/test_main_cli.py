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
         patch.object(cli, "_run_proxy") as run_proxy, \
         patch("autoconduck.agents.claude_code.ClaudeCodeAdapter.patch") as claude_patch, \
         patch("autoconduck.launcher.install_shims") as install_shims, \
         patch("autoconduck.launcher.ensure_path_entry"):
        with patch.object(sys, "argv", ["autoconduck", "start", "--headless", "--port", "12345", "--host", "0.0.0.0"]):
            cli.main()
    run_proxy.assert_called_once_with(12345, "warning", "0.0.0.0")
    claude_patch.assert_called_once()
    install_shims.assert_called_once_with(["claude_code"])


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


# Helpers for cmd_launch_agent tests (new flow uses kill+daemon+poll, not ensure_server)


_CLAUDE_ENV_MOCK = {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:11434",
    "ANTHROPIC_AUTH_TOKEN": "autoconduck-local",
    "ANTHROPIC_MODEL": "autoconduck",
    "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
}


def _build_base_patches():
    """Return a list of patch objects for common cmd_launch_agent setup."""
    p = [
        patch.object(cli.subprocess, "Popen", MagicMock()),
    ]
    return p


def test_cmd_launch_agent_uses_real_binary_env_and_releases_in_order():
    cfg = Config(port=11434, pseudo_model="autoconduck")
    adapter_cls_mock = MagicMock()
    adapter_instance = SimpleNamespace(id="claude_code", binary_name="claude", patch=lambda cfg: None)
    adapter_cls_mock.return_value = adapter_instance
    events = []
    result = SimpleNamespace(returncode=0)

    def run(command, **kwargs):
        events.append(("run", command, kwargs))
        return result

    home_mock = MagicMock()
    log_mock = MagicMock()
    home_mock.__truediv__ = lambda self, key: log_mock
    log_mock.parent.mkdir = MagicMock()

    patches = _build_base_patches()
    patches.append(patch.object(cli, "load_config", return_value=cfg))
    patches.append(patch.object(cli, "home_dir", return_value=home_mock))
    patches.append(patch("builtins.open", MagicMock()))
    patches.append(patch("autoconduck.launcher.kill_existing_on_port"))
    patches.append(patch("autoconduck.launcher.daemon_python", return_value="pythonw.exe"))
    real_path_val = "C:\\fake\\claude.exe"
    patches.append(patch("autoconduck.launcher.real_binary_path", return_value=real_path_val))
    release_var = {}

    def record_release(port):
        release_var["called"] = True
        events.append(("release", port))
    patches.append(patch("autoconduck.launcher.release_server", side_effect=record_release))
    patches.append(patch("autoconduck.launcher._claude_env", return_value=_CLAUDE_ENV_MOCK))
    patches.append(patch.object(cli.subprocess, "run", side_effect=run))

    urlopen_ctx = patch("urllib.request.urlopen")
    adapter_ctx = patch("autoconduck.agents.claude_code.ClaudeCodeAdapter", adapter_cls_mock)

    ctx_mgrs = patches + [urlopen_ctx, adapter_ctx]
    # Use contextlib.ExitStack to dynamically compose patches
    with patch("contextlib.ExitStack") as stack_mock:
        stack = MagicMock()
        stack_mock.return_value.__enter__ = MagicMock(side_effect=lambda s=stack: s)
        stack_mock.return_value.__exit__ = MagicMock(return_value=False)
        for p in ctx_mgrs:
            p.start()
        try:
            assert cli.cmd_launch_agent("claude_code") == 0
        finally:
            for p in reversed(ctx_mgrs):
                p.stop()

    assert [event[0] for event in events] == ["run", "release"]
    assert events[0][1] == ["C:\\fake\\claude.exe"]
    env = events[0][2]["env"]
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:11434"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "autoconduck-local"
    assert env["ANTHROPIC_MODEL"] == "autoconduck"
    assert env["CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS"] == "1"


def test_cmd_launch_agent_passes_file_object_streams_to_popen(tmp_path):
    cfg = Config(port=11434)
    captured = {}
    adapter_instance = SimpleNamespace(id="claude_code", binary_name="claude", patch=lambda cfg: None)

    def popen(command, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(pid=1234)

    with patch.object(cli, "load_config", return_value=cfg), \
         patch.object(cli, "home_dir", return_value=tmp_path), \
         patch("autoconduck.launcher.kill_existing_on_port"), \
         patch("autoconduck.launcher.daemon_python", return_value="pythonw.exe"), \
         patch("autoconduck.launcher.real_binary_path", return_value=None), \
         patch.object(cli.shutil, "which", return_value=None), \
         patch("autoconduck.launcher.release_server"), \
         patch.object(cli.subprocess, "Popen", side_effect=popen), \
         patch("urllib.request.urlopen"), \
         patch("autoconduck.agents.claude_code.ClaudeCodeAdapter", return_value=adapter_instance):
        assert cli.cmd_launch_agent("claude_code") == 1

    log_path = tmp_path / "run" / "server.log"
    assert captured["stdout"].name == str(log_path)
    assert captured["stderr"].name == str(log_path)
    assert hasattr(captured["stdout"], "fileno")
    assert hasattr(captured["stderr"], "fileno")
    assert not isinstance(captured["stdout"], type(log_path))
    assert not isinstance(captured["stderr"], type(log_path))


def test_cmd_launch_agent_nonzero_exit_still_releases():
    cfg = Config(port=11434)
    events = []
    result = SimpleNamespace(returncode=3)

    def run(command, **kwargs):
        events.append(("run", command, kwargs))
        return result

    home_mock = MagicMock()
    log_mock = MagicMock()
    home_mock.__truediv__ = lambda self, key: log_mock
    log_mock.parent.mkdir = MagicMock()

    real_path_val = "C:\\fake\\opencode.exe"

    with patch.object(cli, "load_config", return_value=cfg), \
         patch.object(cli, "home_dir", return_value=home_mock), \
         patch("builtins.open", MagicMock()), \
         patch("autoconduck.launcher.kill_existing_on_port"), \
         patch("autoconduck.launcher.daemon_python", return_value="pythonw.exe"), \
         patch("autoconduck.launcher.real_binary_path", return_value=real_path_val), \
         patch("autoconduck.launcher.release_server") as release, \
         patch.object(cli.subprocess, "run", side_effect=run), \
         patch.object(cli.subprocess, "Popen", MagicMock()):
        with patch("urllib.request.urlopen"):
            assert cli.cmd_launch_agent("opencode") == 3
    release.assert_called_once_with(11434)


def test_cmd_launch_agent_missing_binary_returns_one_and_releases():
    cfg = Config(port=11434)
    events = []

    home_mock = MagicMock()
    log_mock = MagicMock()
    home_mock.__truediv__ = lambda self, key: log_mock
    log_mock.parent.mkdir = MagicMock()

    with patch.object(cli, "load_config", return_value=cfg), \
         patch.object(cli, "home_dir", return_value=home_mock), \
         patch("builtins.open", MagicMock()), \
         patch("autoconduck.launcher.kill_existing_on_port"), \
         patch("autoconduck.launcher.daemon_python", return_value="pythonw.exe"), \
         patch("autoconduck.launcher.real_binary_path", return_value=None), \
         patch.object(cli.shutil, "which", return_value=None), \
         patch("autoconduck.launcher.release_server") as release, \
         patch.object(cli.subprocess, "Popen", MagicMock()):
        with patch("urllib.request.urlopen"):
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
