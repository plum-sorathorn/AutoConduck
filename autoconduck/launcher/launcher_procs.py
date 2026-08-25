"""Platform-specific process and platform helpers."""
import ctypes
import os
import signal
import subprocess
import sys
import time
import re

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


def find_process_on_port(port: int) -> int | None:
    """Return the PID listening on *port*, using available platform tools."""
    try:
        import psutil

        for connection in psutil.net_connections(kind="inet"):
            if connection.laddr and connection.laddr.port == port and connection.pid:
                return connection.pid
        return None
    except (ImportError, OSError):
        pass
    if os.name == "nt":
        try:
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
            result = subprocess.run(
                ["netstat", "-ano", "-p", "tcp"],
                capture_output=True,
                text=True,
                check=False,
                creationflags=flags,
            )
            return _parse_netstat_output(result.stdout, port)
        except OSError:
            return None
    for command in (("lsof", "-i", f":{port}"), ("ss", "-ltnp")):
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
        except OSError:
            continue
        pid = (
            _parse_lsof_output(result.stdout)
            if command[0] == "lsof"
            else _parse_ss_output(result.stdout)
        )
        if pid is not None:
            return pid
    return None


def kill_process(pid: int) -> bool:
    if os.name == "nt":
        try:
            result = subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
                check=False,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            return result.returncode == 0
        except OSError:
            return False
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    time.sleep(0.1)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    try:
        os.kill(pid, signal.SIGKILL)
        return True
    except OSError:
        return False


def get_parent_pid(pid: int) -> int | None:
    """Return the parent PID of a given process ID, or None if unknown."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class PROCESSENTRY32(ctypes.Structure):
            _fields_ = [
                ("dwSize", wintypes.DWORD),
                ("cntUsage", wintypes.DWORD),
                ("th32ProcessID", wintypes.DWORD),
                ("th32DefaultHeapID", ctypes.c_size_t),
                ("th32ModuleID", wintypes.DWORD),
                ("cntThreads", wintypes.DWORD),
                ("th32ParentProcessID", wintypes.DWORD),
                ("pcPriClassBase", wintypes.LONG),
                ("dwFlags", wintypes.DWORD),
                ("szExeFile", ctypes.c_char * 260),
            ]

        kernel32 = ctypes.windll.kernel32
        snapshot = kernel32.CreateToolhelp32Snapshot(2, 0)  # TH32CS_SNAPPROCESS = 2
        if not snapshot or snapshot == -1:
            return None
        entry = PROCESSENTRY32()
        entry.dwSize = ctypes.sizeof(PROCESSENTRY32)
        try:
            if kernel32.Process32First(snapshot, ctypes.byref(entry)):
                while True:
                    if entry.th32ProcessID == pid:
                        return entry.th32ParentProcessID
                    if not kernel32.Process32Next(snapshot, ctypes.byref(entry)):
                        break
        finally:
            kernel32.CloseHandle(snapshot)
        return None
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("PPid:"):
                    return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return None


def is_port_bindable(port: int, host: str = "127.0.0.1") -> bool:
    """Check if the TCP port is immediately bindable without error."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def wait_for_port_free(port: int, host: str = "127.0.0.1", timeout: float = 3.0) -> bool:
    """Poll until the TCP port is completely free and bindable."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if is_port_bindable(port, host):
            return True
        time.sleep(0.05)
    return is_port_bindable(port, host)


def prompt_kill_port(port: int, pid: int) -> bool:
    if sys.stdin is None or not hasattr(sys.stdin, "isatty") or not sys.stdin.isatty():
        return False
    answer = input(
        f"Port {port} is in use by process {pid}. Kill it and start AutoConduck? [y/N] "
    )
    return answer.strip().lower() in {"y", "yes"}

def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        return _pid_alive_windows(pid)
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except (ProcessLookupError, OSError):
        return False
    return True


def _pid_alive_windows(pid: int) -> bool:
    """Check process liveness on Windows without terminating the process.

    ``os.kill(pid, 0)`` is NOT a no-op probe on Windows the way it is on
    POSIX: CPython's ``nt_kill`` maps any signal value other than
    ``CTRL_C_EVENT``/``CTRL_BREAK_EVENT`` -- including ``0`` -- straight to
    ``TerminateProcess(handle, sig)``. Using it to "check" liveness was
    actually killing the target process on every poll, which made
    supervisor-cleanup tests race against PID reuse and report processes
    as perpetually alive. Query the exit code via the Win32 API instead so
    the check is side-effect free.
    """
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    ERROR_ACCESS_DENIED = 5
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return ctypes.get_last_error() == ERROR_ACCESS_DENIED
    try:
        exit_code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return False
        return exit_code.value == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def _create_kill_on_close_job():
    """Create a Windows job that terminates assigned children when closed."""
    if os.name != "nt":
        return None
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            return None

        class IoCounters(ctypes.Structure):
            _fields_ = [
                (name, ctypes.c_ulonglong)
                for name in (
                    "ReadOperationCount",
                    "WriteOperationCount",
                    "OtherOperationCount",
                    "ReadTransferCount",
                    "WriteTransferCount",
                    "OtherTransferCount",
                )
            ]

        class BasicLimitInformation(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", ctypes.c_ulong),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", ctypes.c_ulong),
                ("Affinity", ctypes.c_void_p),
                ("PriorityClass", ctypes.c_ulong),
                ("SchedulingClass", ctypes.c_ulong),
            ]

        class ExtendedLimitInformation(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimitInformation),
                ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        info = ExtendedLimitInformation()
        info.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            handle, 9, ctypes.byref(info), ctypes.sizeof(info)
        ):
            kernel32.CloseHandle(handle)
            return None
        return handle
    except Exception:
        return None


def _clear_dead_owner_claim() -> bool:
    """Remove an owner marker left by a crashed supervisor."""
    from autoconduck import launcher
    _, claims, log = launcher._files()
    with launcher._claims_lock():
        try:
            lines = claims.read_text().splitlines()
        except OSError:
            return False
        kept = []
        removed = False
        for line in lines:
            fields = line.split()
            if len(fields) > 1 and fields[1] == "owner":
                try:
                    alive = _pid_alive(int(fields[0]))
                except ValueError:
                    alive = False
                if not alive:
                    removed = True
                    with log.open("a", encoding="utf-8") as stream:
                        stream.write(f"cleared stale owner claim for PID {fields[0]}\n")
                    continue
            kept.append(line)
        if removed:
            if kept:
                claims.write_text("\n".join(kept) + "\n")
            else:
                try:
                    claims.unlink()
                except OSError:
                    pass
        return removed


def _read_pid():
    from autoconduck import launcher
    pid, _, _ = launcher._files()
    try:
        return int(pid.read_text().strip())
    except (OSError, ValueError):
        return None


def _parse_ss_output(text: str) -> int | None:
    import re
    match = re.search(r"pid=(\d+)", text)
    return int(match.group(1)) if match else None
