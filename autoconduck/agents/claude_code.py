from __future__ import annotations
import copy
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from .base import BaseAdapter
from ..config import Config, backups_dir
from ..launcher import _claude_env


class ClaudeCodeAdapter(BaseAdapter):
    binary_name = "claude"
    id = "claude_code"
    display_name = "Claude Code"

    def detect(self) -> bool:
        if shutil.which("claude") is not None:
            return True
        return any(p.exists() for p in self.config_paths())

    def config_paths(self) -> list[Path]:
        home = Path.home()
        return [
            home / ".config" / "claude" / "settings.json",
            home / ".claude.json",
            home / ".claude" / "settings.json",
        ]

    def patch(self, config: Config, port: int = 11434) -> None:
        path = Path.home() / ".claude" / "settings.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except (OSError, json.JSONDecodeError):
            data = {}
        if not isinstance(data, dict):
            data = {}
        first_takeover = not isinstance(data.get("autoconduck"), dict)
        if path.exists() and first_takeover:
            backup = backups_dir("claude_code")
            backup.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
            (backup / f"{stamp}.bak").write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        env = data.get("env") if isinstance(data.get("env"), dict) else {}
        values = _claude_env(port, getattr(config, "pseudo_model", "autoconduck"))
        marker = data.get("autoconduck") if isinstance(data.get("autoconduck"), dict) else {}
        previous = marker.get("previous_env", {})
        if not isinstance(previous, dict):
            previous = {}
        # Only snapshot pre-existing user env values on first-time takeover;
        # on later idempotent patches, previous_env must stay untouched so
        # revert() restores true original values instead of managed ones.
        if first_takeover:
            for key in values:
                if key not in previous and key in env:
                    previous[key] = env[key]
        for key, value in values.items():
            env[key] = value
        data["env"] = env
        claude_settings = getattr(config, "claude_code", None)
        allowed_tools = list(getattr(claude_settings, "allowed_tools", []))
        permissions = data.get("permissions")
        previous_permissions = (copy.deepcopy(marker["previous_permissions"])
                                if "previous_permissions" in marker else copy.deepcopy(permissions))
        if not isinstance(permissions, dict):
            permissions = {}
        existing_allow = permissions.get("allow")
        if not isinstance(existing_allow, list):
            existing_allow = []
        permissions["allow"] = list(dict.fromkeys(existing_allow + allowed_tools))
        default_mode = getattr(claude_settings, "default_mode", None)
        if default_mode is not None and "defaultMode" not in permissions:
            permissions["defaultMode"] = default_mode
        if getattr(claude_settings, "enable_all_project_mcp_servers", False):
            permissions["enableAllProjectMcpServers"] = True
        data["permissions"] = permissions
        contributed = marker.get("contributed_permissions", allowed_tools)
        if not isinstance(contributed, list):
            contributed = allowed_tools
        data["autoconduck"] = {
            "managed_env_keys": list(values), "previous_env": previous,
            "previous_permissions": previous_permissions,
            "contributed_permissions": list(dict.fromkeys(contributed + allowed_tools)),
        }
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    def revert(self) -> None:
        """Remove managed environment values while preserving user settings."""
        for p in self.config_paths():
            if p.exists():
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                marker = data.get("autoconduck")
                env = data.get("env") if isinstance(data.get("env"), dict) else {}
                if isinstance(marker, dict):
                    previous = marker.get("previous_env", {})
                    managed_keys = marker.get("managed_env_keys", [])
                    keys = list(managed_keys) if isinstance(managed_keys, list) else []
                    keys.extend(("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL", "ANTHROPIC_CUSTOM_MODEL_OPTION", "ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS", "CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT"))
                    for key in dict.fromkeys(keys):
                        if key in previous:
                            env[key] = previous[key]
                        else:
                            env.pop(key, None)
                    if env:
                        data["env"] = env
                    else:
                        data.pop("env", None)
                    if "previous_permissions" in marker and marker["previous_permissions"] is not None:
                        data["permissions"] = marker["previous_permissions"]
                    elif "previous_permissions" in marker:
                        permissions = data.get("permissions")
                        if isinstance(permissions, dict):
                            allow = permissions.get("allow")
                            contributed = marker.get("contributed_permissions", [])
                            if isinstance(allow, list) and isinstance(contributed, list):
                                permissions["allow"] = [item for item in allow if item not in contributed]
                                if not permissions["allow"]:
                                    permissions.pop("allow", None)
                                if not permissions:
                                    data.pop("permissions", None)
                data.pop("autoconduck", None)
                p.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
