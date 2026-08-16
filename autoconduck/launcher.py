"""Process and PATH integration for agent launcher shims."""

from __future__ import annotations
import ctypes, errno, os, re, signal, subprocess, sys, tempfile, time
from contextlib import contextmanager
from pathlib import Path
from urllib.request import urlopen
from . import config
from .launcher_procs import (
    _parse_netstat_output, _parse_lsof_output, _parse_ss_output,
    find_process_on_port, kill_process, prompt_kill_port, _pid_alive,
    _pid_alive_windows, _create_kill_on_close_job, _clear_dead_owner_claim, _read_pid,
)


@contextmanager
def _claims_lock():
    """Serialize claims read-modify-writes across shim processes."""
    lock_path = _run_dir() / "server.claims.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as stream:
        # Append while unlocked: byte zero may be locked by another process,
        # while appends are always made outside that lock region.  The lock
        # file is deliberately never replaced (the claims file is).
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as exc:
                    if exc.errno not in (errno.EACCES, errno.EDEADLK):
                        raise
                    time.sleep(0.01)
            else:
                raise TimeoutError(f"timed out acquiring claims lock: {lock_path}")
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def shims_dir() -> Path:
    return config.home_dir() / "bin"


def _run_dir() -> Path:
    return config.run_dir()


def _files():
    return (
        _run_dir() / "server.pid",
        _run_dir() / "server.claims",
        _run_dir() / "server.log",
    )


def _port(port):
    return port or config.get_config().port


def daemon_python() -> str:
    """Use the windowless Windows interpreter when it is available."""
    if os.name == "nt" and sys.executable.lower().endswith("python.exe"):
        pythonw = Path(sys.executable).with_name("pythonw.exe")
        if pythonw.exists():
            return str(pythonw)
    return sys.executable


def server_alive(port=None, timeout=0.5) -> bool:
    try:
        with urlopen(
            f"http://127.0.0.1:{_port(port)}/healthz", timeout=timeout
        ) as response:
            return response.status == 200
    except Exception:
        return False


def _parse_netstat_output(text: str, port: int | None = None) -> int | None:
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 5 and fields[0].upper() == "TCP":
            local = fields[1].rsplit(":", 1)
            if (
                len(local) == 2
                and fields[3].upper() == "LISTENING"
                and (port is None or local[1] == str(port))
            ):
                try:
                    return int(fields[4])
                except ValueError:
                    pass
    return None


def _parse_lsof_output(text: str) -> int | None:
    for line in text.splitlines()[1:]:
        fields = line.split()
        if len(fields) >= 2:
            try:
                return int(fields[1])
            except ValueError:
                pass
    return None


def _parse_ss_output(text: str) -> int | None:
    match = re.search(r"pid=(\d+)", text)
    return int(match.group(1)) if match else None



def _write_claim(owned, owner=False, pid=None):
    _, claims, _ = _files()
    claims.parent.mkdir(parents=True, exist_ok=True)
    with _claims_lock():
        try:
            lines = claims.read_text().splitlines() if claims.exists() else []
        except OSError:
            lines = []
        lines.append(
            f"{os.getpid() if pid is None else pid} {'owner' if owner else (1 if owned else 0)}"
        )
        fd, name = tempfile.mkstemp(dir=claims.parent, prefix="claims.")
        try:
            with os.fdopen(fd, "w") as f:
                f.write("\n".join(lines) + "\n")
            os.replace(name, claims)
        finally:
            try:
                os.unlink(name)
            except OSError:
                pass


def _start_daemon(port):
    pid, _, log = _files()
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        daemon_python(),
        "-m",
        "autoconduck",
        "start",
        "--headless",
        "--supervisor",
        "--port",
        str(port),
    ]
    with log.open("ab") as stream:
        if os.name == "nt":
            flags = (
                subprocess.DETACHED_PROCESS
                | subprocess.CREATE_NEW_PROCESS_GROUP
                | subprocess.CREATE_NO_WINDOW
            )
            child = subprocess.Popen(
                command,
                stdout=stream,
                stderr=stream,
                creationflags=flags,
                close_fds=True,
            )
        else:
            child = subprocess.Popen(
                command, stdout=stream, stderr=stream, start_new_session=True
            )
    pid.write_text(str(child.pid))
    return child.pid


def kill_existing_on_port(port: int) -> None:
    """Forcefully kill any process listening on *port*.

    This is called by ``cmd_launch_agent`` so every --claude launch gets a
    fresh server process instead of reusing stale ones.
    """
    pid = find_process_on_port(port)
    if pid is not None:
        kill_process(pid)
        time.sleep(0.3)  # brief grace period for OS to release the socket


def ensure_server(port=None) -> bool:
    port = _port(port)
    pid, _, _ = _files()
    _clear_dead_owner_claim()
    if server_alive(port):
        _write_claim(False)
        return False
    existing_pid = find_process_on_port(port)
    if existing_pid is not None:
        if not prompt_kill_port(port, existing_pid) or not kill_process(existing_pid):
            return False
    try:
        _start_daemon(port)
    except OSError:
        return False
    # Exponential-backoff polling avoids hammering the health endpoint while
    # allowing the first LiteLLM import to use the full cold-start budget.
    try:
        ready_budget = max(
            30.0, float(os.environ.get("AUTOCONDUCK_READY_TIMEOUT", "60.0"))
        )
    except ValueError:
        ready_budget = 60.0
    deadline = time.monotonic() + ready_budget
    attempt = 0
    while time.monotonic() < deadline:
        if server_alive(port, timeout=min(0.5, max(0.01, deadline - time.monotonic()))):
            _write_claim(True)
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(0.15 * (1.5**attempt), 0.8, remaining))
        attempt += 1
    try:
        pid.unlink()
    except OSError:
        pass
    return False



def stop_server(port=None) -> bool:
    pidfile, claims, _ = _files()
    pid = _read_pid()
    if pid is None:
        return False
    child_path = pidfile.parent / "child.pid"
    try:
        child_pid = int(child_path.read_text().strip())
    except (OSError, ValueError):
        child_pid = None
    if child_pid is not None and _pid_alive(child_pid):
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(child_pid), "/F"],
                    check=False,
                    capture_output=True,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
            else:
                os.kill(child_pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                check=False,
                capture_output=True,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        else:
            killpg = getattr(os, "killpg", None)
            if killpg is not None:
                killpg(pid, signal.SIGTERM)
            else:
                os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and (
        _pid_alive(pid) or (child_pid is not None and _pid_alive(child_pid))
    ):
        time.sleep(0.05)
    if child_pid is not None and _pid_alive(child_pid) and os.name != "nt":
        try:
            os.kill(child_pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    for path in (pidfile, claims, child_path):
        try:
            path.unlink()
        except OSError:
            pass
    return True


def release_server(port=None) -> None:
    pidfile, claims, _ = _files()
    with _claims_lock():
        try:
            lines = claims.read_text().splitlines()
        except OSError:
            return
        removed = False
        kept = []
        for line in lines:
            fields = line.split()
            if (
                not removed
                and fields
                and fields[0] == str(os.getpid())
                and len(fields) > 1
                and fields[1] != "owner"
            ):
                removed = True
            else:
                kept.append(line)
        has_owner = any(
            len(line.split()) > 1 and line.split()[1] == "owner" for line in kept
        )
        has_active_clients = any(
            len(line.split()) > 1 and line.split()[1] != "owner"
            for line in kept
        )
        if kept:
            claims.write_text("\n".join(kept) + "\n")
        else:
            try:
                claims.unlink()
            except OSError:
                pass
        if not has_active_clients and not has_owner:
            stop_server(port)



from .launcher_shims import (
    _adapter, real_binary_path, _claude_env, _claude_env_blocks, _pi_env_blocks,
    shim_script, shim_script_win, install_shims, uninstall_shims,
    ensure_path_entry, _ensure_windows_path, remove_path_entry,
)
