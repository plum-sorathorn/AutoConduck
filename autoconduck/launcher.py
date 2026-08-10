"""Process and PATH integration for agent launcher shims."""
from __future__ import annotations
import ctypes, errno, os, re, shutil, signal, subprocess, sys, tempfile, time
from contextlib import contextmanager
from pathlib import Path
from urllib.request import urlopen
from . import config

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

def shims_dir() -> Path: return config.home_dir() / "bin"
def _run_dir() -> Path: return config.run_dir()
def _files(): return (_run_dir()/"server.pid", _run_dir()/"server.claims", _run_dir()/"server.log")
def _port(port): return port or config.get_config().port

def daemon_python() -> str:
    """Use the windowless Windows interpreter when it is available."""
    if os.name == "nt" and sys.executable.lower().endswith("python.exe"):
        pythonw = Path(sys.executable).with_name("pythonw.exe")
        if pythonw.exists(): return str(pythonw)
    return sys.executable
def server_alive(port=None, timeout=.5) -> bool:
    try:
        with urlopen(f"http://127.0.0.1:{_port(port)}/healthz", timeout=timeout) as response:
            return response.status == 200
    except Exception: return False

def _parse_netstat_output(text: str, port: int | None = None) -> int | None:
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 5 and fields[0].upper() == "TCP":
            local = fields[1].rsplit(":", 1)
            if len(local) == 2 and fields[3].upper() == "LISTENING" and (port is None or local[1] == str(port)):
                try: return int(fields[4])
                except ValueError: pass
    return None

def _parse_lsof_output(text: str) -> int | None:
    for line in text.splitlines()[1:]:
        fields = line.split()
        if len(fields) >= 2:
            try: return int(fields[1])
            except ValueError: pass
    return None

def _parse_ss_output(text: str) -> int | None:
    match = re.search(r"pid=(\d+)", text)
    return int(match.group(1)) if match else None

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
            result = subprocess.run(["netstat", "-ano", "-p", "tcp"], capture_output=True, text=True, check=False)
            return _parse_netstat_output(result.stdout, port)
        except OSError:
            return None
    for command in (("lsof", "-i", f":{port}"), ("ss", "-ltnp")):
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False)
        except OSError:
            continue
        pid = _parse_lsof_output(result.stdout) if command[0] == "lsof" else _parse_ss_output(result.stdout)
        if pid is not None: return pid
    return None

def kill_process(pid: int) -> bool:
    if os.name == "nt":
        try:
            result = subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True, check=False, creationflags=subprocess.CREATE_NO_WINDOW)
            return result.returncode == 0
        except OSError: return False
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    time.sleep(.1)
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

def prompt_kill_port(port: int, pid: int) -> bool:
    if not sys.stdin.isatty(): return False
    answer = input(f"Port {port} is in use by process {pid}. Kill it and start AutoConduck? [y/N] ")
    return answer.strip().lower() in {"y", "yes"}

def _write_claim(owned, owner=False, pid=None):
    _, claims, _ = _files(); claims.parent.mkdir(parents=True, exist_ok=True)
    with _claims_lock():
        try: lines = claims.read_text().splitlines() if claims.exists() else []
        except OSError: lines = []
        lines.append(f"{os.getpid() if pid is None else pid} {'owner' if owner else (1 if owned else 0)}")
        fd, name = tempfile.mkstemp(dir=claims.parent, prefix="claims.")
        try:
            with os.fdopen(fd, "w") as f: f.write("\n".join(lines) + "\n")
            os.replace(name, claims)
        finally:
            try: os.unlink(name)
            except OSError: pass

def _start_daemon(port):
    pid, _, log = _files(); log.parent.mkdir(parents=True, exist_ok=True)
    command = [daemon_python(), "-m", "autoconduck", "start", "--headless", "--supervisor", "--port", str(port)]
    with log.open("ab") as stream:
        if os.name == "nt":
            flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
            child = subprocess.Popen(command, stdout=stream, stderr=stream, creationflags=flags, close_fds=True)
        else: child = subprocess.Popen(command, stdout=stream, stderr=stream, start_new_session=True)
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
    port = _port(port); pid, _, _ = _files()
    _clear_dead_owner_claim()
    if server_alive(port):
        _write_claim(False); return False
    existing_pid = find_process_on_port(port)
    if existing_pid is not None:
        if not prompt_kill_port(port, existing_pid) or not kill_process(existing_pid):
            return False
    try: _start_daemon(port)
    except OSError: return False
    # Exponential-backoff poll (max ~6 s total instead of flat 10 s).
    # The daemon only needs to import everything once; it becomes ready
    # well before all 50 iterations are needed in practice.
    try:
        ready_budget = max(0.0, float(os.environ.get("AUTOCONDUCK_READY_TIMEOUT", "30.0")))
    except ValueError:
        ready_budget = 30.0
    deadline = time.monotonic() + ready_budget
    attempt = 0
    while time.monotonic() < deadline:
        if server_alive(port, timeout=min(.5, max(.01, deadline - time.monotonic()))):
            _write_claim(True); return True
        remaining = deadline - time.monotonic()
        if remaining <= 0: break
        time.sleep(min(0.15 * (1.5 ** attempt), 0.8, remaining))
        attempt += 1
    try: pid.unlink()
    except OSError: pass
    return False

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
            _fields_ = [(name, ctypes.c_ulonglong) for name in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

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
    _, claims, log = _files()
    with _claims_lock():
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
    pid, _, _ = _files()
    try: return int(pid.read_text().strip())
    except (OSError, ValueError): return None

def stop_server(port=None) -> bool:
    pidfile, claims, _ = _files(); pid = _read_pid()
    if pid is None: return False
    child_path = pidfile.parent / "child.pid"
    try:
        child_pid = int(child_path.read_text().strip())
    except (OSError, ValueError):
        child_pid = None
    if child_pid is not None and _pid_alive(child_pid):
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(child_pid), "/F"], check=False,
                               capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                os.kill(child_pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    try:
        if os.name == "nt": subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            killpg = getattr(os, "killpg", None)
            if killpg is not None: killpg(pid, signal.SIGTERM)
            else: os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError): pass
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and (_pid_alive(pid) or (child_pid is not None and _pid_alive(child_pid))):
        time.sleep(.05)
    if child_pid is not None and _pid_alive(child_pid) and os.name != "nt":
        try: os.kill(child_pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError): pass
    for path in (pidfile, claims, child_path):
        try: path.unlink()
        except OSError: pass
    return True

def release_server(port=None) -> None:
    pidfile, claims, _ = _files()
    with _claims_lock():
        try: lines = claims.read_text().splitlines()
        except OSError: return
        removed = False; kept = []
        for line in lines:
            fields = line.split()
            if (not removed and fields and fields[0] == str(os.getpid())
                    and len(fields) > 1 and fields[1] != "owner"):
                removed = True
            else: kept.append(line)
        has_owner = any(len(line.split()) > 1 and line.split()[1] == "owner" for line in kept)
        has_active_clients = any(len(line.split()) > 1 and line.split()[1] not in ("owner", "0") for line in kept)
        if kept:
            claims.write_text("\n".join(kept) + "\n")
        else:
            try: claims.unlink()
            except OSError: pass
        if not has_active_clients and not has_owner:
            stop_server(port)

def _adapter(agent_id):
    from .agents import all_adapters
    return next((a for a in all_adapters() if a.id == agent_id), None)
def real_binary_path(agent_id):
    adapter = _adapter(agent_id); name = getattr(adapter, "binary_name", None)
    if not name: return None
    blocked = str(shims_dir()).lower()
    path = os.pathsep.join(p for p in os.environ.get("PATH", "").split(os.pathsep) if p.lower() != blocked)
    return shutil.which(name, path=path)

def _claude_env(port: int, pseudo: str = "autoconduck") -> dict[str, str]:
    return {
        "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}",
        "ANTHROPIC_AUTH_TOKEN": "autoconduck-local",
        "ANTHROPIC_MODEL": pseudo,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": pseudo,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": pseudo,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": pseudo,
        "ANTHROPIC_CUSTOM_MODEL_OPTION": pseudo,
        "ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION": "AutoConduck local router",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
        "CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT": "1",
        "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "1000000",
    }

def _claude_env_blocks(port: int, pseudo: str = "autoconduck") -> tuple[str, str]:
    """Return (bash_exports, cmd_sets) for the Claude Code agent_id only."""
    values = _claude_env(port, pseudo)
    bash = "\n".join(f'export {key}="{value}"' for key, value in values.items())
    bash = bash.replace(values["ANTHROPIC_BASE_URL"], "http://127.0.0.1:${PORT}")
    cmd = "\n".join(f'set "{key}={value}"' for key, value in values.items())
    cmd = cmd.replace(values["ANTHROPIC_BASE_URL"], "http://127.0.0.1:%PORT%")
    return bash, cmd

def _pi_env_blocks() -> tuple[str, str]:
    """Return Pi's standard agent markers for launcher-created processes.

    Pi normally adds these itself, but setting them in the shim also covers
    wrappers and older Pi releases without changing Pi's provider settings.
    """
    return (
        'export AI_AGENT="pi"\nexport PI_CODING_AGENT="true"',
        'set "AI_AGENT=pi"\nset "PI_CODING_AGENT=true"',
    )

def shim_script(agent_id, real_bin):
    import shlex
    real_bin = str(real_bin)
    cfg = config.get_config()
    lines = [
        "#!/usr/bin/env bash",
        "set -u",
        f"REAL_BIN={shlex.quote(real_bin)}",
        f"PY={shlex.quote(sys.executable)}",
        f'PORT="${{AUTOCONDUCK_PORT:-{cfg.port}}}"',
    ]
    if agent_id == "claude_code":
        bash_env, _ = _claude_env_blocks(cfg.port, cfg.pseudo_model)
        lines.append(bash_env)
    elif agent_id == "pi":
        bash_env, _ = _pi_env_blocks()
        lines.append(bash_env)
    lines.append('"$PY" -m autoconduck ensure --port "$PORT" || true')
    lines.append('"$REAL_BIN" "$@"')
    lines.append("rc=$?")
    lines.append('"$PY" -m autoconduck release --port "$PORT" || true')
    lines.append("exit $rc")
    return "\n".join(lines) + "\n"
def shim_script_win(agent_id, real_bin):
    real_bin = str(real_bin).replace("\\", "/")
    def esc(s): return s.replace("%", "%%").replace('"', '""')
    cfg = config.get_config()
    lines = [
        "@echo off",
        f'set "REAL_BIN={esc(real_bin)}"',
        f'set "PY={esc(sys.executable)}"',
        'set "PORT=%AUTOCONDUCK_PORT%"',
        "if not defined PORT set PORT=" + str(cfg.port),
    ]
    if agent_id == "claude_code":
        _, cmd_env = _claude_env_blocks(cfg.port, cfg.pseudo_model)
        lines.append(cmd_env)
    elif agent_id == "pi":
        _, cmd_env = _pi_env_blocks()
        lines.append(cmd_env)
    lines.append('"%PY%" -m autoconduck ensure --port %PORT%')
    lines.append('"%REAL_BIN%" %*')
    lines.append("set RC=%ERRORLEVEL%")
    lines.append('"%PY%" -m autoconduck release --port %PORT%')
    lines.append("exit /b %RC%")
    return "\n".join(lines) + "\n"

def install_shims(agent_ids):
    directory = shims_dir(); directory.mkdir(parents=True, exist_ok=True); result = {}
    cfg = config.get_config()
    for aid in agent_ids:
        real = real_binary_path(aid)
        adapter = _adapter(aid)
        if real is None or adapter is None: continue
        real = str(real)
        name = adapter.binary_name; path = directory / (name + (".cmd" if os.name == "nt" else ""))
        path.write_text(shim_script_win(aid, real) if os.name == "nt" else shim_script(aid, real), encoding="utf-8")
        if os.name != "nt": path.chmod(0o755)
        cfg.shims[aid] = str(path); result[aid] = path
    config.save_config(cfg); return result

def uninstall_shims(agent_ids=None):
    cfg = config.get_config(); ids = agent_ids or list(cfg.shims); deleted = []
    for aid in ids:
        path = Path(cfg.shims.get(aid, "")) if cfg.shims.get(aid) else None
        if path and path.exists(): path.unlink(); deleted.append(path)
        cfg.shims.pop(aid, None)
    config.save_config(cfg); return deleted

_BLOCK = re.compile(r"\n?# BEGIN AUTOCONDUCK PATH\n.*?\n# END AUTOCONDUCK PATH\n?", re.S)
def ensure_path_entry():
    if os.name == "nt": return _ensure_windows_path()
    changed = None; block = '# BEGIN AUTOCONDUCK PATH\nexport PATH="$HOME/.autoconduck/bin:$PATH"\n# END AUTOCONDUCK PATH\n'
    for rc in (Path.home()/".bashrc", Path.home()/".zshrc"):
        if rc == Path.home()/".zshrc" and not rc.exists(): continue
        text = rc.read_text() if rc.exists() else ""
        if "# BEGIN AUTOCONDUCK PATH" not in text: rc.write_text(text + ("\n" if text and not text.endswith("\n") else "") + block); changed = changed or rc
    return changed
def _ensure_windows_path():
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_READ|winreg.KEY_WRITE)
        try: old, _ = winreg.QueryValueEx(key, "Path")
        except OSError: old = ""
        new = str(shims_dir()) + os.pathsep + old
        if str(shims_dir()).lower() not in old.lower().split(os.pathsep): winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, new)
        winreg.CloseKey(key); return "registry"
    except OSError: return None
def remove_path_entry():
    if os.name == "nt":
        try:
            import winreg
            registry = winreg.ConnectRegistry(None, winreg.HKEY_CURRENT_USER)
            key = winreg.OpenKey(registry, "Environment", 0, winreg.KEY_READ | winreg.KEY_WRITE)
            try:
                old, value_type = winreg.QueryValueEx(key, "Path")
            except OSError:
                winreg.CloseKey(key); winreg.CloseKey(registry); return None
            target = str(shims_dir()).casefold()
            entries = str(old).split(";")
            kept = [entry for entry in entries if entry.casefold() != target]
            if len(kept) != len(entries):
                winreg.SetValueEx(key, "Path", 0, value_type, ";".join(kept))
            winreg.CloseKey(key); winreg.CloseKey(registry)
        except OSError:
            return None
        return None
    for rc in (Path.home()/".bashrc", Path.home()/".zshrc"):
        try: rc.write_text(_BLOCK.sub("\n", rc.read_text()))
        except OSError: pass
