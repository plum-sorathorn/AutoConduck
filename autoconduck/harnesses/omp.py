"""Adapter for Oh My Pi (OMP)."""
from __future__ import annotations

import shutil
from pathlib import Path

from ..config import Config
from .base import BaseAdapter


class OmpAdapter(BaseAdapter):
    """Register AutoConduck as an isolated provider in OMP's config."""

    binary_name = "omp"
    id = "omp"
    display_name = "Oh My Pi"
    provider_name = "autoconduck"
    PSEUDO_MODELS = (
        "autoconduck/fast",
        "autoconduck/balanced",
        "autoconduck/frontier",
        "autoconduck/smart-dag",
    )

    def detect(self) -> bool:
        return self.detect_binary() or self.detect_config()

    def detect_binary(self) -> bool:
        return shutil.which(self.binary_name) is not None

    def detect_config(self) -> bool:
        return any(path.is_file() for path in self.config_paths())

    def config_paths(self) -> list[Path]:
        home = Path.home()
        return [
            home / ".omp" / "config.json",
            home / ".omp" / "plugins",
            Path.cwd() / ".omprc",
            Path.cwd() / ".omp.json",
        ]

    def patch(self, config: Config, port: int | None = None) -> None:
        effective_port = int(port if port is not None else getattr(config, "port", 11434))
        path = self.config_paths()[0]
        endpoint = f"http://127.0.0.1:{effective_port}/v1"

        def update(data: dict) -> None:
            # Keep all managed values in one namespace so user presets and
            # provider settings remain untouched.
            data["autoconduck"] = {
                "provider": self.provider_name,
                "base_url": endpoint,
                "api_key": "autoconduck-local",
                "models": list(self.PSEUDO_MODELS),
            }

        self._patch_json(path, update)

    def install_features(self) -> list[str]:
        return []
