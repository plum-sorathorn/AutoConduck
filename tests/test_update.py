from unittest.mock import patch

from autoconduck import update


def test_upgrade_command_mapping():
    assert update.upgrade_command("uv-tool") == "uv tool upgrade --reinstall autoconduck"
    assert update.upgrade_command("npm") == "npm install -g autoconduck@latest"
    assert update.upgrade_command("pip") == "pip install --force-reinstall --upgrade autoconduck"
    assert update.upgrade_command("uv-tool-editable") == "uv tool install --reinstall --editable ."
    assert update.upgrade_command("pip-editable") == "pip install --force-reinstall -e ."
    assert update.upgrade_command("unknown") is None


def test_uninstall_hint_mapping():
    assert update.uninstall_hint("uv-tool") == "uv tool uninstall autoconduck"
    assert update.uninstall_hint("uv-tool-editable") == "uv tool uninstall autoconduck"
    assert update.uninstall_hint("pip") == "pip uninstall autoconduck"
    assert update.uninstall_hint("pip-editable") == "pip uninstall autoconduck"
    assert update.uninstall_hint("npm") == "npm uninstall -g autoconduck"
    assert update.uninstall_hint("unknown") is None


def _detect(*, editable, executable, npm=False, distribution=True):
    with patch.dict("os.environ", {"AUTOCONDUCK_WHEEL_DIR": "x"} if npm else {}, clear=True), \
         patch.object(update, "_is_editable", return_value=editable), \
         patch.object(update.sys, "executable", executable), \
         patch.object(update.importlib.metadata, "distribution", side_effect=None if distribution else update.importlib.metadata.PackageNotFoundError):
        return update.detect_install_method()


def test_detect_install_methods():
    uv_python = r"C:\Users\plum\AppData\Roaming\uv\tools\autoconduck\Scripts\pythonw.exe"
    regular_python = r"C:\Python\python.exe"
    assert _detect(editable=False, executable=regular_python, npm=True) == "npm"
    assert _detect(editable=False, executable=uv_python) == "uv-tool"
    assert _detect(editable=True, executable=uv_python) == "uv-tool-editable"
    assert _detect(editable=False, executable=regular_python) == "pip"
    assert _detect(editable=True, executable=regular_python) == "pip-editable"
    assert _detect(editable=False, executable=regular_python, distribution=False) == "unknown"


def test_uv_tool_editable_guidance(capsys):
    from autoconduck import main
    with patch.object(update, "detect_install_method", return_value="uv-tool-editable"):
        main.cmd_update(type("Args", (), {"dry_run": True})())
    output = capsys.readouterr().out
    assert "uv tool install --reinstall --editable ." in output
    assert "pip install -e ." not in output
