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
def server_alive(port=None) -> bool:
    try:
        with urlopen(f"http://127.0.0.1:{_port(port)}/healthz", timeout=.5) as response:
            return response.status == 200
    except Exception: return False

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
    command = [sys.executable, "-m", "autoconduck", "start", "--headless", "--daemon", "--port", str(port)]
    with log.open("ab") as stream:
        if os.name == "nt":
            flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
            child = subprocess.Popen(command, stdout=stream, stderr=stream, creationflags=flags, close_fds=True)
        else: child = subprocess.Popen(command, stdout=stream, stderr=stream, start_new_session=True)
    pid.write_text(str(child.pid))
    return child.pid

def ensure_server(port=None) -> bool:
    port = _port(port); pid, _, _ = _files()
    if server_alive(port):
        _write_claim(False); return False
    try: _start_daemon(port)
    except OSError: return False
    for _ in range(50):
        if server_alive(port): _write_claim(True); return True
        time.sleep(.2)
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
        if os.name == "nt": subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False, capture_output=True)
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

def shim_script(agent_id, real_bin):
    import shlex
    real_bin = str(real_bin)
    return ("#!/usr/bin/env bash\nset -u\n" f"REAL_BIN={shlex.quote(real_bin)}\nPY={shlex.quote(sys.executable)}\n"
            f'PORT="${{AUTOCONDUCK_PORT:-{config.get_config().port}}}"\n" "$PY" -m autoconduck ensure --port "$PORT" || true\n'
            '"$REAL_BIN" "$@"\nrc=$?\n"$PY" -m autoconduck release --port "$PORT" || true\nexit $rc\n')
def shim_script_win(agent_id, real_bin):
    real_bin = str(real_bin).replace("\\", "/")
    def esc(s): return s.replace("%", "%%").replace('"', '""')
    return (f'@echo off\nset "REAL_BIN={esc(real_bin)}"\nset "PY={esc(sys.executable)}"\n'
            'set "PORT=%AUTOCONDUCK_PORT%"\nif not defined PORT set PORT=' + str(config.get_config().port) + '\n'
            '"%PY%" -m autoconduck ensure --port %PORT%\n"%REAL_BIN%" %*\nset RC=%ERRORLEVEL%\n'
            '"%PY%" -m autoconduck release --port %PORT%\nexit /b %RC%\n')

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
    if os.name == "nt": return None
    for rc in (Path.home()/".bashrc", Path.home()/".zshrc"):
        try: rc.write_text(_BLOCK.sub("\n", rc.read_text()))
        except OSError: pass
