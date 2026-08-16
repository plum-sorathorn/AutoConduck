"""Real-process supervisor lifecycle tests for AutoConduck launcher.

Marked as slow integration tests because they spawn real background processes and poll.
"""
import os
import subprocess
import sys
import time
import pytest
from autoconduck import launcher


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
    child_pid = None
    child_path = proc._autoconduck_state.parent / "run" / "child.pid" if hasattr(proc, "_autoconduck_state") else None
    if child_path is not None:
        deadline = time.time() + 2
        while time.time() < deadline and child_pid is None:
            try:
                child_pid = int(child_path.read_text().strip())
            except (OSError, ValueError):
                time.sleep(0.05)
    try:
        if child_pid is not None:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(child_pid), "/F"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                try:
                    os.kill(child_pid, __import__("signal").SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            deadline = time.time() + 5
            while time.time() < deadline and launcher._pid_alive(child_pid):
                time.sleep(0.05)
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            try:
                os.killpg(proc.pid, __import__("signal").SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        if proc.poll() is None:
            proc.wait(timeout=10)
        deadline = time.time() + 5
        while time.time() < deadline and launcher._pid_alive(proc.pid):
            time.sleep(0.05)
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
                if state.exists() and int(state.read_text()) >= 2:
                    break
            except (OSError, ValueError):
                pass
            time.sleep(0.05)
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
