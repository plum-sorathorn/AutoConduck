"""Process and PATH integration for agent launcher shims."""

from __future__ import annotations
import ctypes, errno, os, re, signal, subprocess, sys, tempfile, time
from contextlib import contextmanager
from pathlib import Path
from urllib.request import urlopen
from autoconduck import config
from .launcher_procs import (
    _parse_netstat_output, _parse_lsof_output, _parse_ss_output,
    find_process_on_port, kill_process, prompt_kill_port, _pid_alive,
    _pid_alive_windows, _create_kill_on_close_job, _clear_dead_owner_claim, _read_pid,
)

def _pkg():
    return sys.modules.get("autoconduck.launcher") or sys.modules.get(__name__)


@contextmanager
def _claims_lock():
    """Serialize claims read-modify-writes across shim processes."""
    mod = _pkg()
    run_dir_fn = getattr(mod, "_run_dir", _run_dir)
    lock_path = run_dir_fn() / "server.claims.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        os_mod = getattr(mod, "os", os)
        if os_mod.name == "nt":
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
            if os_mod.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def shims_dir() -> Path:
    mod = _pkg()
    cfg = getattr(mod, "config", config)
    return cfg.home_dir() / "bin"


def _run_dir() -> Path:
    mod = _pkg()
    cfg = getattr(mod, "config", config)
    return cfg.run_dir()


def _files():
    mod = _pkg()
    files_fn = getattr(mod, "_files", None)
    if files_fn and files_fn is not _files:
        return files_fn()
    run_dir_fn = getattr(mod, "_run_dir", _run_dir)
    d = run_dir_fn()
    return (
        d / "server.pid",
        d / "server.claims",
        d / "server.log",
    )


def _port(port):
    mod = _pkg()
    cfg = getattr(mod, "config", config)
    return port or cfg.get_config().port


def daemon_python() -> str:
    """Use the windowless Windows interpreter when it is available."""
    mod = _pkg()
    os_mod = getattr(mod, "os", os)
    sys_mod = getattr(mod, "sys", sys)
    path_cls = getattr(mod, "Path", Path)
    if os_mod.name == "nt" and sys_mod.executable.lower().endswith("python.exe"):
        pythonw = path_cls(sys_mod.executable).with_name("pythonw.exe")
        if pythonw.exists():
            return str(pythonw)
    return sys_mod.executable


def server_alive(port=None, timeout=0.5) -> bool:
    mod = _pkg()
    urlopen_fn = getattr(mod, "urlopen", urlopen)
    port_fn = getattr(mod, "_port", _port)
    try:
        with urlopen_fn(
            f"http://127.0.0.1:{port_fn(port)}/healthz", timeout=timeout
        ) as response:
            return response.status == 200
    except Exception:
        return False


def _parse_netstat_output(text: str, port: int | None = None) -> int | None:
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 5 and fields[0].upper() == "TCP":
            local_addr = fields[1]
            state = fields[3]
            pid_str = fields[4]
            if state.upper() == "LISTENING" and pid_str.isdigit():
                if port is None or local_addr.endswith(f":{port}"):
                    return int(pid_str)
    return None


def _parse_lsof_output(text: str) -> int | None:
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[1].isdigit():
            return int(fields[1])
    return None


def _parse_ss_output(text: str) -> int | None:
    match = re.search(r"pid=(\d+)", text)
    return int(match.group(1)) if match else None


def _write_claim(is_owner: bool = False, owner: bool | None = None, pid: int | None = None) -> None:
    mod = _pkg()
    files_fn = getattr(mod, "_files", _files)
    claims_lock_ctx = getattr(mod, "_claims_lock", _claims_lock)
    os_mod = getattr(mod, "os", os)
    _, claims, _ = files_fn()
    claims.parent.mkdir(parents=True, exist_ok=True)
    target_pid = pid if pid is not None else os_mod.getpid()
    actual_owner = is_owner if owner is None else owner
    tag = "owner" if actual_owner else "0"
    with claims_lock_ctx():
        with claims.open("a", encoding="utf-8") as stream:
            stream.write(f"{target_pid} {tag}\n")


def _start_daemon(port: int, extra_flags: list[str] | None = None) -> int:
    mod = _pkg()
    files_fn = getattr(mod, "_files", _files)
    daemon_fn = getattr(mod, "daemon_python", daemon_python)
    subproc_mod = getattr(mod, "subprocess", subprocess)
    os_mod = getattr(mod, "os", os)
    pid, claims, log = files_fn()
    pid.parent.mkdir(parents=True, exist_ok=True)
    flags = [flag for flag in (extra_flags or []) if flag != "--daemon"]
    command = [
        daemon_fn(),
        "-m",
        "autoconduck",
        "start",
        "--headless",
        "--supervisor",
        "--port",
        str(port),
        *flags,
    ]
    with log.open("ab") as stream:
        if os_mod.name == "nt":
            child = subproc_mod.Popen(
                command,
                stdout=stream,
                stderr=stream,
                creationflags=getattr(subproc_mod, "CREATE_NO_WINDOW", 0x08000000)
                | getattr(subproc_mod, "DETACHED_PROCESS", 0x00000008),
            )
        else:
            child = subproc_mod.Popen(
                command, stdout=stream, stderr=stream, start_new_session=True
            )
    pid.write_text(str(child.pid))
    return child.pid


def kill_existing_on_port(port: int) -> None:
    mod = _pkg()
    find_proc_fn = getattr(mod, "find_process_on_port", find_process_on_port)
    kill_fn = getattr(mod, "kill_process", kill_process)
    pid = find_proc_fn(port)
    if pid is not None:
        kill_fn(pid)
        time.sleep(0.3)


def ensure_server(port=None) -> bool:
    mod = _pkg()
    port_fn = getattr(mod, "_port", _port)
    port = port_fn(port)
    files_fn = getattr(mod, "_files", _files)
    pid, _, _ = files_fn()
    clear_dead_fn = getattr(mod, "_clear_dead_owner_claim", _clear_dead_owner_claim)
    clear_dead_fn()
    alive_fn = getattr(mod, "server_alive", server_alive)
    if alive_fn(port):
        write_claim_fn = getattr(mod, "_write_claim", _write_claim)
        write_claim_fn(False)
        return False
    find_proc_fn = getattr(mod, "find_process_on_port", find_process_on_port)
    existing_pid = find_proc_fn(port)
    if existing_pid is not None:
        prompt_fn = getattr(mod, "prompt_kill_port", prompt_kill_port)
        kill_fn = getattr(mod, "kill_process", kill_process)
        if not prompt_fn(port, existing_pid) or not kill_fn(existing_pid):
            return False
    try:
        start_daemon_fn = getattr(mod, "_start_daemon", _start_daemon)
        start_daemon_fn(port)
    except OSError:
        return False
    try:
        ready_budget = max(
            30.0, float(os.environ.get("AUTOCONDUCK_READY_TIMEOUT", "60.0"))
        )
    except ValueError:
        ready_budget = 60.0
    deadline = time.monotonic() + ready_budget
    attempt = 0
    while time.monotonic() < deadline:
        if alive_fn(port, timeout=min(0.5, max(0.01, deadline - time.monotonic()))):
            write_claim_fn = getattr(mod, "_write_claim", _write_claim)
            write_claim_fn(True)
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
    mod = _pkg()
    files_fn = getattr(mod, "_files", _files)
    pidfile, claims, _ = files_fn()
    read_pid_fn = getattr(mod, "_read_pid", _read_pid)
    pid = read_pid_fn()
    child_path = pidfile.parent / "child.pid"
    try:
        child_pid = int(child_path.read_text().strip())
    except (OSError, ValueError):
        child_pid = None
    pid_alive_fn = getattr(mod, "_pid_alive", _pid_alive)
    os_mod = getattr(mod, "os", os)
    subproc_mod = getattr(mod, "subprocess", subprocess)

    # If the pidfile is missing or its PID is already dead, fall back to
    # discovering the daemon by the port it listens on.  This handles the case
    # where the supervisor writes a stale pidfile (e.g. after a crash-restart
    # cycle) and prevents the daemon from surviving a reset/uninstall.
    if pid is None or not pid_alive_fn(pid):
        port_fn = getattr(mod, "_port", _port)
        resolved_port = port_fn(port)
        find_proc_fn = getattr(mod, "find_process_on_port", find_process_on_port)
        kill_fn = getattr(mod, "kill_process", kill_process)
        port_pid = find_proc_fn(resolved_port)
        if port_pid is not None:
            kill_fn(port_pid)
            deadline2 = time.monotonic() + 5.0
            while time.monotonic() < deadline2 and pid_alive_fn(port_pid):
                time.sleep(0.05)
        # Clean up stale files regardless.
        for path in (pidfile, claims, child_path):
            try:
                path.unlink()
            except OSError:
                pass
        return port_pid is not None

    if child_pid is not None and pid_alive_fn(child_pid):
        try:
            if os_mod.name == "nt":
                subproc_mod.run(
                    ["taskkill", "/PID", str(child_pid), "/F"],
                    check=False,
                    capture_output=True,
                    creationflags=getattr(subproc_mod, "CREATE_NO_WINDOW", 0x08000000),
                )
            else:
                os_mod.kill(child_pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    try:
        if os_mod.name == "nt":
            subproc_mod.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                check=False,
                capture_output=True,
                creationflags=getattr(subproc_mod, "CREATE_NO_WINDOW", 0x08000000),
            )
        else:
            killpg = getattr(os_mod, "killpg", None)
            if killpg is not None:
                killpg(pid, signal.SIGTERM)
            else:
                os_mod.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and (
        pid_alive_fn(pid) or (child_pid is not None and pid_alive_fn(child_pid))
    ):
        time.sleep(0.05)
    if child_pid is not None and pid_alive_fn(child_pid) and os_mod.name != "nt":
        try:
            os_mod.kill(child_pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    for path in (pidfile, claims, child_path):
        try:
            path.unlink()
        except OSError:
            pass
    return True


def release_server(port=None) -> None:
    mod = _pkg()
    files_fn = getattr(mod, "_files", _files)
    pidfile, claims, _ = files_fn()
    claims_lock_ctx = getattr(mod, "_claims_lock", _claims_lock)
    os_mod = getattr(mod, "os", os)
    stop_server_fn = getattr(mod, "stop_server", stop_server)
    with claims_lock_ctx():
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
                and fields[0] == str(os_mod.getpid())
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
            stop_server_fn(port)


from .launcher_shims import (
    _adapter, real_binary_path, _claude_env, _claude_env_blocks, _pi_env_blocks,
    shim_script, shim_script_win, install_shims, uninstall_shims,
    ensure_path_entry, _ensure_windows_path, remove_path_entry,
)
