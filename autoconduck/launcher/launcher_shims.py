"""Agent launcher shims, environment integration, and PATH helpers."""
from __future__ import annotations
import os, re, shutil, sys
from pathlib import Path
from autoconduck import config

def _shims_dir() -> Path:
    # Resolve through the facade so existing monkeypatch targets keep working.
    from autoconduck import launcher
    return launcher.shims_dir()

def _adapter(agent_id):
    from autoconduck.harnesses import all_adapters

    return next((a for a in all_adapters() if a.id == agent_id), None)


def real_binary_path(agent_id):
    adapter = _adapter(agent_id)
    name = getattr(adapter, "binary_name", None)
    if not name:
        return None
    blocked = str(_shims_dir()).lower()
    path = os.pathsep.join(
        p for p in os.environ.get("PATH", "").split(os.pathsep) if p.lower() != blocked
    )
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
    """Return Pi's standard agent markers for launcher-created processes."""
    return (
        'export AI_AGENT="pi"\nexport PI_CODING_AGENT="true"',
        'set "AI_AGENT=pi"\nset "PI_CODING_AGENT=true"',
    )


def _omp_env_blocks(port: int, pseudo: str = "autoconduck") -> tuple[str, str]:
    """Return OMP's proxy endpoint and default model environment."""
    return (
        'export OMP_BASE_URL="http://127.0.0.1:${PORT}/v1"\n'
        'export OMP_API_KEY="autoconduck-local"\n'
        f'export OMP_MODEL="autoconduck/{pseudo}"',
        'set "OMP_BASE_URL=http://127.0.0.1:%PORT%/v1"\n'
        'set "OMP_API_KEY=autoconduck-local"\n'
        f'set "OMP_MODEL=autoconduck/{pseudo}"',
    )


def _opencode_env_blocks(port: int, pseudo: str = "autoconduck") -> tuple[str, str]:
    """Return environment variables for OpenCode."""
    bash = (
        'export OPENAI_BASE_URL="http://127.0.0.1:${PORT}/v1"\n'
        'export OPENAI_API_KEY="autoconduck-local"\n'
        f'export OPENCODE_MODEL="autoconduck/{pseudo}"'
    )
    cmd = (
        'set "OPENAI_BASE_URL=http://127.0.0.1:%PORT%/v1"\n'
        'set "OPENAI_API_KEY=autoconduck-local"\n'
        f'set "OPENCODE_MODEL=autoconduck/{pseudo}"'
    )
    return bash, cmd


def shim_script(agent_id, real_bin):
    import shlex

    real_bin = str(real_bin)
    cfg = config.get_config()
    pseudo = getattr(cfg, "pseudo_model", "autoconduck") or "autoconduck"
    lines = [
        "#!/usr/bin/env bash",
        "set -u",
        f"REAL_BIN={shlex.quote(real_bin)}",
        f"PY={shlex.quote(sys.executable)}",
        f'PORT="${{AUTOCONDUCK_PORT:-{cfg.port}}}"',
    ]
    if agent_id == "claude_code":
        bash_env, _ = _claude_env_blocks(cfg.port, pseudo)
        lines.append(bash_env)
    elif agent_id == "pi":
        bash_env, _ = _pi_env_blocks()
        lines.append(bash_env)
    elif agent_id == "opencode":
        bash_env, _ = _opencode_env_blocks(cfg.port, pseudo)
        lines.append(bash_env)
    elif agent_id == "omp":
        bash_env, _ = _omp_env_blocks(cfg.port, pseudo)
        lines.append(bash_env)
    lines.append('SHIM_ID="$$"')
    lines.append('"$PY" -m autoconduck ensure --port "$PORT" --client-id "$SHIM_ID" || true')
    lines.append('"$REAL_BIN" "$@"')
    lines.append("rc=$?")
    lines.append('"$PY" -m autoconduck release --port "$PORT" --client-id "$SHIM_ID" || true')
    lines.append("exit $rc")
    return "\n".join(lines) + "\n"


def shim_script_win(agent_id, real_bin):
    real_bin = str(real_bin).replace("\\", "/")

    def esc(s):
        return s.replace("%", "%%").replace('"', '""')

    cfg = config.get_config()
    pseudo = getattr(cfg, "pseudo_model", "autoconduck") or "autoconduck"
    lines = [
        "@echo off",
        f'set "REAL_BIN={esc(real_bin)}"',
        f'set "PY={esc(sys.executable)}"',
        'set "PORT=%AUTOCONDUCK_PORT%"',
        "if not defined PORT set PORT=" + str(cfg.port),
    ]
    if agent_id == "claude_code":
        _, cmd_env = _claude_env_blocks(cfg.port, pseudo)
        lines.append(cmd_env)
    elif agent_id == "pi":
        _, cmd_env = _pi_env_blocks()
        lines.append(cmd_env)
    elif agent_id == "opencode":
        _, cmd_env = _opencode_env_blocks(cfg.port, pseudo)
        lines.append(cmd_env)
    elif agent_id == "omp":
        _, cmd_env = _omp_env_blocks(cfg.port, pseudo)
        lines.append(cmd_env)
    lines.append('set "SHIM_ID=%RANDOM%%RANDOM%%RANDOM%"')
    lines.append('"%PY%" -m autoconduck ensure --port %PORT% --client-id %SHIM_ID%')
    lines.append('"%REAL_BIN%" %*')
    lines.append("set RC=%ERRORLEVEL%")
    lines.append('"%PY%" -m autoconduck release --port %PORT% --client-id %SHIM_ID%')
    lines.append("exit /b %RC%")
    return "\n".join(lines) + "\n"


def install_shims(agent_ids):
    directory = _shims_dir()
    directory.mkdir(parents=True, exist_ok=True)
    result = {}
    cfg = config.get_config()
    for aid in agent_ids:
        real = real_binary_path(aid)
        adapter = _adapter(aid)
        if real is None or adapter is None:
            continue
        real = str(real)
        name = adapter.binary_name
        if not name:
            continue
        path = directory / (name + (".cmd" if os.name == "nt" else ""))
        path.write_text(
            shim_script_win(aid, real) if os.name == "nt" else shim_script(aid, real),
            encoding="utf-8",
        )
        if os.name != "nt":
            path.chmod(0o755)
        cfg.shims[aid] = str(path)
        result[aid] = path
    config.save_config(cfg)
    return result


def uninstall_shims(agent_ids=None):
    cfg = config.get_config()
    ids = agent_ids or list(cfg.shims)
    deleted = []
    for aid in ids:
        path = Path(cfg.shims.get(aid, "")) if cfg.shims.get(aid) else None
        if path and path.exists():
            path.unlink()
            deleted.append(path)
        cfg.shims.pop(aid, None)
    config.save_config(cfg)
    return deleted


_BLOCK = re.compile(
    r"\n?# BEGIN AUTOCONDUCK PATH\n.*?\n# END AUTOCONDUCK PATH\n?", re.S
)


def ensure_path_entry():
    if os.name == "nt":
        return _ensure_windows_path()
    changed = None
    block = '# BEGIN AUTOCONDUCK PATH\nexport PATH="$HOME/.autoconduck/bin:$PATH"\n# END AUTOCONDUCK PATH\n'
    for rc in (Path.home() / ".bashrc", Path.home() / ".zshrc"):
        if rc == Path.home() / ".zshrc" and not rc.exists():
            continue
        text = rc.read_text() if rc.exists() else ""
        if "# BEGIN AUTOCONDUCK PATH" not in text:
            rc.write_text(
                text + ("\n" if text and not text.endswith("\n") else "") + block
            )
            changed = changed or rc
    return changed


def _ensure_windows_path():
    try:
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            "Environment",
            0,
            winreg.KEY_READ | winreg.KEY_WRITE,
        )
        try:
            old, _ = winreg.QueryValueEx(key, "Path")
        except OSError:
            old = ""
        new = str(_shims_dir()) + os.pathsep + old
        if str(_shims_dir()).lower() not in old.lower().split(os.pathsep):
            winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, new)
        winreg.CloseKey(key)
        return "registry"
    except OSError:
        return None


def remove_path_entry():
    if os.name == "nt":
        try:
            import winreg

            registry = winreg.ConnectRegistry(None, winreg.HKEY_CURRENT_USER)
            key = winreg.OpenKey(
                registry, "Environment", 0, winreg.KEY_READ | winreg.KEY_WRITE
            )
            try:
                old, value_type = winreg.QueryValueEx(key, "Path")
            except OSError:
                winreg.CloseKey(key)
                winreg.CloseKey(registry)
                return None
            target = str(_shims_dir()).casefold()
            entries = str(old).split(";")
            kept = [entry for entry in entries if entry.casefold() != target]
            if len(kept) != len(entries):
                winreg.SetValueEx(key, "Path", 0, value_type, ";".join(kept))
            winreg.CloseKey(key)
            winreg.CloseKey(registry)
        except OSError:
            return None
        return None
    for rc in (Path.home() / ".bashrc", Path.home() / ".zshrc"):
        try:
            rc.write_text(_BLOCK.sub("\n", rc.read_text()))
        except OSError:
            pass
