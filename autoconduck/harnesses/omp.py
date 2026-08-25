"""Adapter for Oh My Pi (OMP)."""

from __future__ import annotations

import shutil
from pathlib import Path

from ..config import Config, backups_dir
from .base import BaseAdapter


class OmpAdapter(BaseAdapter):
    """Register AutoConduck in OMP's YAML model registry."""

    binary_name: str | None = "omp"
    id = "omp"
    display_name = "Oh My Pi"
    provider_name = "autoconduck"
    PSEUDO_MODELS = (
        "fast",
        "balanced",
        "frontier",
        "smart-dag",
    )

    def detect(self) -> bool:
        return self.detect_binary() or self.detect_config()

    def detect_binary(self) -> bool:
        binary_name = self.binary_name
        return binary_name is not None and shutil.which(binary_name) is not None

    def detect_config(self) -> bool:
        return any(path.is_file() for path in self.config_paths())

    def config_paths(self) -> list[Path]:
        home = Path.home()
        return [
            home / ".omp" / "agent" / "models.yml",
            home / ".omp" / "agent" / "models.yaml",
            Path.cwd() / ".omp" / "config.yml",
        ]

    def settings_paths(self) -> list[Path]:
        home = Path.home() / ".omp" / "agent"
        return [home / "config.yml", home / "config.yaml"]

    def patch(self, config: Config, port: int | None = None) -> None:
        import yaml

        try:
            effective_port = int(
                port if port is not None else getattr(config, "port", 11434)
            )
        except (TypeError, ValueError):
            effective_port = 11434
        target = next(
            (path for path in self.config_paths() if path.exists()),
            self.config_paths()[0],
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        existing = {}
        if target.exists():
            self.backup(target)
            try:
                loaded = yaml.safe_load(target.read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError):
                loaded = {}
            if isinstance(loaded, dict):
                existing = loaded

        providers = existing.setdefault("providers", {})
        if not isinstance(providers, dict):
            providers = {}
            existing["providers"] = providers
        providers[self.provider_name] = {
            "baseUrl": f"http://127.0.0.1:{effective_port}/v1",
            "api": "openai-completions",
            "apiKey": "autoconduck-local",
            "auth": "none",
            "models": [
                {
                    "id": model,
                    "name": f"AutoConduck {model.replace('-', ' ').title()}",
                    "contextWindow": 1000000,
                    "maxTokens": 16384,
                }
                for model in self.PSEUDO_MODELS
            ],
        }
        target.write_text(yaml.safe_dump(existing, sort_keys=False), encoding="utf-8")

        configured_pseudo_model = getattr(config, "pseudo_model", None)
        pseudo_model = str(
            configured_pseudo_model
            if configured_pseudo_model in self.PSEUDO_MODELS
            else "balanced"
        )
        settings_target = next(
            (path for path in self.settings_paths() if path.exists()),
            self.settings_paths()[0],
        )
        settings_target.parent.mkdir(parents=True, exist_ok=True)
        settings = {}
        if settings_target.exists():
            self.backup(settings_target)
            try:
                loaded = yaml.safe_load(settings_target.read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError):
                loaded = {}
            if isinstance(loaded, dict):
                settings = loaded
        model_roles = settings.setdefault("modelRoles", {})
        if not isinstance(model_roles, dict):
            model_roles = {}
            settings["modelRoles"] = model_roles
        model_roles["default"] = f"autoconduck/{pseudo_model}"
        settings_target.write_text(
            yaml.safe_dump(settings, sort_keys=False), encoding="utf-8"
        )

    def revert(self) -> None:
        """Restore the latest backup for OMP, or remove only our provider."""
        bak_dir = backups_dir(self.id)
        restored_sources: set[Path] = set()
        if bak_dir.exists():
            for backup in sorted(bak_dir.glob("*.bak"), reverse=True):
                meta = backup.with_suffix(".meta")
                try:
                    source = Path(meta.read_text(encoding="utf-8").strip())
                except (OSError, ValueError):
                    continue
                if (
                    source.name
                    in {"models.yml", "models.yaml", "config.yml", "config.yaml"}
                    and source not in restored_sources
                ):
                    try:
                        source.parent.mkdir(parents=True, exist_ok=True)
                        source.write_bytes(backup.read_bytes())
                        restored_sources.add(source)
                    except OSError:
                        continue

        import yaml

        for path in [*self.config_paths(), *self.settings_paths()]:
            if path in restored_sources:
                continue
            if not path.exists():
                continue
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError):
                continue
            if not isinstance(data, dict):
                continue
            providers = data.get("providers")
            if isinstance(providers, dict):
                providers.pop(self.provider_name, None)
                if not providers:
                    data.pop("providers", None)
            if path in self.settings_paths():
                model_roles = data.get("modelRoles")
                if isinstance(model_roles, dict) and str(
                    model_roles.get("default", "")
                ).startswith("autoconduck/"):
                    model_roles.pop("default", None)
                    if not model_roles:
                        data.pop("modelRoles", None)
            path.write_text(
                yaml.safe_dump(data, sort_keys=False) if data else "", encoding="utf-8"
            )

    def install_features(self) -> list[str]:
        return []
