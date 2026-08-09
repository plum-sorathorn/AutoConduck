"""Process and PATH integration for agent launcher shims."""
from __future__ import annotations
import ctypes, os, re, shutil, signal, subprocess, sys, tempfile, time
from pathlib import Path
from urllib.request import urlopen
from . import config

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
def server_alive(port=None) -> bool:
    try:
        with urlopen(f"http://127.0.0.1:{_port(port)}/healthz", timeout=.5) as response:
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

def _write_claim(owned):
    _, claims, _ = _files(); claims.parent.mkdir(parents=True, exist_ok=True)
    try: lines = claims.read_text().splitlines() if claims.exists() else []
    except OSError: lines = []
    lines.append(f"{os.getpid()} {1 if owned else 0}")
    fd, name = tempfile.mkstemp(dir=claims.parent, prefix="claims.")
    with os.fdopen(fd, "w") as f: f.write("\n".join(lines) + "\n")
    os.replace(name, claims)

def _start_daemon(port):
    pid, _, log = _files(); log.parent.mkdir(parents=True, exist_ok=True)
    command = [daemon_python(), "-m", "autoconduck", "start", "--headless", "--daemon", "--port", str(port)]
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
    for attempt in range(25):
        if server_alive(port):
            _write_claim(True); return True
        time.sleep(min(0.15 * (1.5 ** attempt), 0.8))
    try: pid.unlink()
    except OSError: pass
    return False

def _read_pid():
    pid, _, _ = _files()
    try: return int(pid.read_text().strip())
    except (OSError, ValueError): return None

def stop_server(port=None) -> bool:
    pidfile, claims, _ = _files(); pid = _read_pid()
    if pid is None: return False
    try:
        if os.name == "nt": subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
        else: os.kill(pid, signal.SIGTERM)
    except OSError: pass
    for path in (pidfile, claims):
        try: path.unlink()
        except OSError: pass
    return True

def release_server(port=None) -> None:
    pidfile, claims, _ = _files()
    try: lines = claims.read_text().splitlines()
    except OSError: return
    removed = False; kept = []
    for line in lines:
        if not removed and line.split() and line.split()[0] == str(os.getpid()): removed = True
        else: kept.append(line)
    if kept: claims.write_text("\n".join(kept) + "\n")
    else:
        try: claims.unlink()
        except OSError: pass
        if pidfile.exists(): stop_server(port)

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
    }

def _claude_env_blocks(port: int, pseudo: str = "autoconduck") -> tuple[str, str]:
    """Return (bash_exports, cmd_sets) for the Claude Code agent_id only."""
    values = _claude_env(port, pseudo)
    bash = "\n".join(f'export {key}="{value}"' for key, value in values.items())
    bash = bash.replace(values["ANTHROPIC_BASE_URL"], "http://127.0.0.1:${PORT}")
    cmd = "\n".join(f'set "{key}={value}"' for key, value in values.items())
    cmd = cmd.replace(values["ANTHROPIC_BASE_URL"], "http://127.0.0.1:%PORT%")
    return bash, cmd

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
