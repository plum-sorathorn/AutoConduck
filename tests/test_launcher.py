from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import threading
import os
import subprocess
import sys
import time
import pytest

from autoconduck import launcher
from autoconduck import main


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


def test_get_app_builds_and_returns_app():
    built_app = main._get_app()
    assert built_app is main.app


def test_parse_netstat_output():
    text = "  TCP    127.0.0.1:11434    0.0.0.0:0    LISTENING    4321\n"
    assert launcher._parse_netstat_output(text) == 4321


def test_parse_lsof_output():
    text = "COMMAND PID USER FD TYPE DEVICE SIZE/OFF NODE NAME\npython 987 plum 3u IPv4 0t0 TCP *:11434 (LISTEN)\n"
    assert launcher._parse_lsof_output(text) == 987


def test_parse_ss_output():
    assert launcher._parse_ss_output('users:(("python",pid=2468,fd=5))') == 2468


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


def test_windows_remove_path_entry_preserves_other_entries(monkeypatch):
    values = {"Path": (r"C:\Other;C:\AutoConduck\bin;C:\Tools", 2)}
    writes = []
    registry = object()
    key = object()
    fake = type("WinReg", (), {
        "HKEY_CURRENT_USER": object(), "KEY_READ": 1, "KEY_WRITE": 2,
        "ConnectRegistry": staticmethod(lambda *_: registry),
        "OpenKey": staticmethod(lambda *_: key),
        "QueryValueEx": staticmethod(lambda _key, name: values[name]),
        "SetValueEx": staticmethod(lambda _key, name, reserved, kind, value: (writes.append((name, kind, value)), values.__setitem__(name, (value, kind)))),
        "CloseKey": staticmethod(lambda _key: None),
    })
    monkeypatch.setitem(__import__("sys").modules, "winreg", fake)
    monkeypatch.setattr(launcher.os, "name", "nt")
    monkeypatch.setattr(launcher, "shims_dir", lambda: Path(r"C:\AutoConduck\bin"))
    launcher.remove_path_entry()
    assert values["Path"] == (r"C:\Other;C:\Tools", 2)
    assert len(writes) == 1


def test_windows_remove_path_entry_does_not_write_when_absent(monkeypatch):
    values = {"Path": (r"C:\Other;C:\Tools", 2)}
    writes = []
    fake = type("WinReg", (), {
        "HKEY_CURRENT_USER": object(), "KEY_READ": 1, "KEY_WRITE": 2,
        "ConnectRegistry": staticmethod(lambda *_: object()),
        "OpenKey": staticmethod(lambda *_: object()),
        "QueryValueEx": staticmethod(lambda _key, name: values[name]),
        "SetValueEx": staticmethod(lambda *_args: writes.append(True)),
        "CloseKey": staticmethod(lambda _key: None),
    })
    monkeypatch.setitem(__import__("sys").modules, "winreg", fake)
    monkeypatch.setattr(launcher.os, "name", "nt")
    monkeypatch.setattr(launcher, "shims_dir", lambda: Path(r"C:\AutoConduck\bin"))
    launcher.remove_path_entry()
    assert writes == []


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


def test_claude_env_contains_router_settings():
    env = launcher._claude_env(11434, "autoconduck")
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:11434"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "autoconduck-local"
    assert env["ANTHROPIC_MODEL"] == "autoconduck"
    assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    assert env["CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS"] == "1"
    assert env["CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT"] == "1"


def test_claude_shims_render_base_url():
    posix = launcher.shim_script("claude_code", "/opt/claude")
    windows = launcher.shim_script_win("claude_code", "/opt/claude")
    assert 'ANTHROPIC_BASE_URL="http://127.0.0.1:${PORT}"' in posix
    assert 'ANTHROPIC_BASE_URL=http://127.0.0.1:%PORT%' in windows


def test_pi_shim_sets_pi_agent_markers_without_claude_settings():
    posix = launcher.shim_script("pi", "/opt/pi")
    windows = launcher.shim_script_win("pi", "/opt/pi")
    for text in (posix, windows):
        assert "AI_AGENT" in text
        assert "PI_CODING_AGENT" in text
        assert "ensure" in text and "release" in text
        assert "ANTHROPIC_BASE_URL" not in text


def test_daemon_python_prefers_pythonw_when_available(monkeypatch):
    monkeypatch.setattr(launcher.os, "name", "nt")
    monkeypatch.setattr(launcher.sys, "executable", r"C:\Python\python.exe")
    monkeypatch.setattr(launcher.Path, "exists", lambda self: self.name == "pythonw.exe")
    assert launcher.daemon_python() == r"C:\Python\pythonw.exe"

    monkeypatch.setattr(launcher.Path, "exists", lambda self: False)
    assert launcher.daemon_python() == r"C:\Python\python.exe"


def test_start_daemon_passes_create_no_window_on_windows(tmp_path, monkeypatch):
    pidfile = tmp_path / "server.pid"
    claims = tmp_path / "server.claims"
    logfile = tmp_path / "server.log"
    captured = {}

    class Child:
        pid = 12345

    def popen(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return Child()

    monkeypatch.setattr(launcher, "_files", lambda: (pidfile, claims, logfile))
    monkeypatch.setattr(launcher.os, "name", "nt")
    monkeypatch.setattr(launcher.subprocess, "Popen", popen)
    launcher._start_daemon(11434)

    assert "--headless" in captured["command"]
    assert "--daemon" not in captured["command"]
    assert captured["creationflags"] & launcher.subprocess.CREATE_NO_WINDOW
    assert pidfile.read_text() == "12345"


def test_cmd_start_daemon_spawns_detached_server_without_daemon_flag(tmp_path, monkeypatch):
    captured = {}

    class Child:
        pid = 12345

    def popen(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return Child()

    monkeypatch.setattr(main, "load_config", lambda: SimpleNamespace(port=None, log_level="INFO"))
    monkeypatch.setattr(main, "home_dir", lambda: tmp_path)
    monkeypatch.setattr(main, "_check_port_available", lambda port: None)
    monkeypatch.setattr(launcher, "daemon_python", lambda: "pythonw.exe")
    monkeypatch.setattr(main.subprocess, "Popen", popen)
    monkeypatch.setattr(main.sys, "platform", "win32")
    monkeypatch.setattr(launcher, "server_alive", lambda port: True)

    main.cmd_start(SimpleNamespace(
        port=11434,
        host="127.0.0.1",
        headless=True,
        daemon=True,
    ))

    assert "--headless" in captured["command"]
    assert "--daemon" not in captured["command"]
    assert "--supervisor" in captured["command"]
    assert captured["creationflags"] & main.subprocess.DETACHED_PROCESS


@pytest.mark.skipif(os.name == "nt", reason="_pid_alive uses the Win32 query API on Windows, not os.kill")
def test_pid_alive_permission_error_is_alive(monkeypatch):
    def denied(*args):
        raise PermissionError
    monkeypatch.setattr(launcher.os, "kill", denied)
    assert launcher._pid_alive(123) is True


@pytest.mark.skipif(os.name == "nt", reason="_pid_alive uses the Win32 query API on Windows, not os.kill")
def test_pid_alive_process_lookup_error_is_dead(monkeypatch):
    def missing(*args):
        raise ProcessLookupError
    monkeypatch.setattr(launcher.os, "kill", missing)
    assert launcher._pid_alive(123) is False


def test_owner_claim_uses_explicit_supervisor_pid(tmp_path, monkeypatch):
    claims = tmp_path / "server.claims"
    monkeypatch.setattr(launcher, "_run_dir", lambda: tmp_path)
    monkeypatch.setattr(launcher, "_files", lambda: (tmp_path / "server.pid", claims, tmp_path / "server.log"))
    launcher._write_claim(False, owner=True, pid=4242)
    assert claims.read_text() == "4242 owner\n"


def test_claim_updates_are_serialized(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "_run_dir", lambda: tmp_path)
    monkeypatch.setattr(launcher, "_files", lambda: (tmp_path / "server.pid", tmp_path / "server.claims", tmp_path / "server.log"))
    barrier = threading.Barrier(2)

    def add_claim():
        barrier.wait()
        launcher._write_claim(False)

    threads = [threading.Thread(target=add_claim) for _ in range(2)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    assert len((tmp_path / "server.claims").read_text().splitlines()) == 2


def test_release_removes_only_own_shim_and_preserves_owner(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "_run_dir", lambda: tmp_path)
    pidfile = tmp_path / "server.pid"
    claims = tmp_path / "server.claims"
    monkeypatch.setattr(launcher, "_files", lambda: (pidfile, claims, tmp_path / "server.log"))
    claims.write_text("999 0\n123 owner\n")
    pidfile.write_text("456")
    monkeypatch.setattr(launcher.os, "getpid", lambda: 999)
    with patch.object(launcher, "stop_server") as stop:
        launcher.release_server()
    assert claims.read_text() == "123 owner\n"
    stop.assert_not_called()


def test_stop_server_clears_pid_claims_and_owner(tmp_path, monkeypatch):
    pidfile = tmp_path / "server.pid"
    claims = tmp_path / "server.claims"
    pidfile.write_text("999")
    claims.write_text("123 owner\n456 0\n")
    monkeypatch.setattr(launcher, "_files", lambda: (pidfile, claims, tmp_path / "server.log"))
    monkeypatch.setattr(launcher.os, "kill", lambda *args: None)
    monkeypatch.setattr(launcher.os, "name", "posix")
    assert launcher.stop_server()
    assert not pidfile.exists()
    assert not claims.exists()


def test_ensure_server_removes_dead_owner_claim(tmp_path, monkeypatch):
    pidfile = tmp_path / "server.pid"
    claims = tmp_path / "server.claims"
    log = tmp_path / "server.log"
    claims.write_text("2147483647 owner\n")
    monkeypatch.setattr(launcher, "_files", lambda: (pidfile, claims, log))
    monkeypatch.setattr(launcher, "_run_dir", lambda: tmp_path)
    monkeypatch.setattr(launcher, "server_alive", lambda port: True)
    monkeypatch.setattr(launcher, "_port", lambda port: 11434)
    launcher.ensure_server()
    assert "owner" not in claims.read_text()
    assert "stale owner" in log.read_text()


def test_ensure_server_keeps_live_owner_claim(tmp_path, monkeypatch):
    claims = tmp_path / "server.claims"
    log = tmp_path / "server.log"
    claims.write_text(f"{os.getpid()} owner\n")
    monkeypatch.setattr(launcher, "_files", lambda: (tmp_path / "server.pid", claims, log))
    monkeypatch.setattr(launcher, "_run_dir", lambda: tmp_path)
    monkeypatch.setattr(launcher, "server_alive", lambda port: True)
    monkeypatch.setattr(launcher, "_port", lambda port: 11434)
    launcher.ensure_server()
    assert "owner" in claims.read_text()


def _supervisor_process(tmp_path, child_code):
    state = tmp_path / "starts"
    code = (
        "import autoconduck.main as m; "
        "m.SUPERVISOR_INITIAL_BACKOFF = 0.01; "
        "m.SUPERVISOR_MAX_BACKOFF = 0.02; "
        "m.SUPERVISOR_FAILURE_WINDOW = 1.0; "
        "m.SUPERVISOR_MAX_RAPID_FAILURES = 3; "
        "m._run_supervisor(1, host='127.0.0.1', child_cmd=[sys.executable, '-c', "
        f"{child_code!r}])"
    )
    # The supervisor expression needs sys in its globals.
    code = "import sys; " + code
    env = os.environ.copy()
    env["AUTOCONDUCK_HOME"] = str(tmp_path)
    env["AUTOCONDUCK_TEST_STATE"] = str(state)
    kwargs = {"start_new_session": True} if os.name != "nt" else {
        "creationflags": subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW,
    }
    proc = subprocess.Popen([sys.executable, "-c", code], env=env, **kwargs)
    return proc, state


def _cleanup_supervisor(proc):
    """Terminate a supervisor and every child it spawned, even on assertion failure."""
    child_pid = None
    child_path = proc._autoconduck_state.parent / "run" / "child.pid" if hasattr(proc, "_autoconduck_state") else None
    # The caller attaches the state path below; retry briefly for async sidecar creation.
    if child_path is not None:
        deadline = time.time() + 2
        while time.time() < deadline and child_pid is None:
            try: child_pid = int(child_path.read_text().strip())
            except (OSError, ValueError): time.sleep(.05)
    try:
        if child_pid is not None:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(child_pid), "/F"], check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                try: os.kill(child_pid, __import__("signal").SIGKILL)
                except (ProcessLookupError, PermissionError): pass
            deadline = time.time() + 5
            while time.time() < deadline and launcher._pid_alive(child_pid): time.sleep(.05)
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            try: os.killpg(proc.pid, __import__("signal").SIGKILL)
            except (ProcessLookupError, PermissionError): pass
        if proc.poll() is None: proc.wait(timeout=10)
        deadline = time.time() + 5
        while time.time() < deadline and launcher._pid_alive(proc.pid): time.sleep(.05)
        child_dead = child_pid is None or not launcher._pid_alive(child_pid)
        supervisor_dead = not launcher._pid_alive(proc.pid)
        assert child_dead
        assert supervisor_dead
    finally:
        if proc.poll() is None:
            proc.kill()


def test_supervisor_restarts_then_can_be_stopped(tmp_path):
    child = """import os, pathlib, time
p = pathlib.Path(os.environ['AUTOCONDUCK_TEST_STATE'])
n = int(p.read_text()) if p.exists() else 0
p.write_text(str(n + 1))
if n == 0: raise SystemExit(1)
time.sleep(.2)
raise SystemExit(1)
"""
    supervisor, state = _supervisor_process(tmp_path, child)
    supervisor._autoconduck_state = state
    try:
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                if state.exists() and int(state.read_text()) >= 2: break
            except (OSError, ValueError): pass
            time.sleep(.05)
        assert state.exists() and int(state.read_text()) >= 2
        assert supervisor.wait(timeout=10) == 0
    finally:
        _cleanup_supervisor(supervisor)


def test_supervisor_gives_up_after_rapid_failures(tmp_path):
    supervisor, _ = _supervisor_process(tmp_path, "raise SystemExit(1)")
    supervisor._autoconduck_state = tmp_path / "starts"
    try:
        assert supervisor.wait(timeout=20) == 0
        log = (tmp_path / "run" / "server.log").read_text()
        assert "supervisor giving up" in log
    finally:
        _cleanup_supervisor(supervisor)


@pytest.mark.skipif(os.name == "nt", reason="signal semantics differ on Windows")
def test_supervisor_stop_does_not_respawn_child(tmp_path):
    child = "import time; time.sleep(60)"
    supervisor, state = _supervisor_process(tmp_path, child)
    supervisor._autoconduck_state = state
    try:
        time.sleep(.5)
        supervisor.send_signal(__import__("signal").SIGTERM)
        assert supervisor.wait(timeout=10) == 0
        assert not state.exists()
    finally:
        _cleanup_supervisor(supervisor)


def test_supervisor_backoff_doubles_and_caps(tmp_path, monkeypatch):
    class Child:
        pid = 1
        def wait(self):
            return 1
        def poll(self):
            return 1

    sleeps = []
    monkeypatch.setattr(main, "home_dir", lambda: tmp_path)
    monkeypatch.setattr(main.subprocess, "Popen", lambda *args, **kwargs: Child())
    monkeypatch.setattr(main.time, "sleep", lambda delay: sleeps.append(delay))
    main._run_supervisor(1, child_cmd=[sys.executable, "-c", "pass"])
    assert sleeps == [1, 2, 4, 8]
