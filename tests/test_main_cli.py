import sys
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
