"""Adapter for the Pi coding agent (https://github.com/badlogic/pi-mono)."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from .base import BaseAdapter
from ..config import Config, normalize_api_base


class PiAdapter(BaseAdapter):
    """Register AutoConduck as a custom provider for Pi.

    Pi (>=0.84.1) does not read a ``providers`` block from
    ``settings.json``.  Custom providers are instead registered by a
    TypeScript extension auto-discovered from
    ``<agent_dir>/extensions/*.ts`` that calls
    ``pi.registerProvider(name, config)``.  ``settings.json`` is only
    used to select the default provider/model.
    """

    binary_name = "pi"
    id = "pi"
    display_name = "Pi"

    provider_name = "autoconduck"

    #: Every pseudo-model the LiteLLM proxy serves.  Registered together so
    #: the extension exposes the full catalog regardless of which one is
    #: the default.
    PSEUDO_MODELS = ("autoconduck", "autoconduck-budget", "autoconduck-expensive")

    @staticmethod
    def _agent_dir() -> Path:
        """Return Pi's settings directory.

        Pi supports relocating its data directory.  Honour that setting so
        detection and installation operate on the same file when users do so.
        """
        configured = os.environ.get("PI_CODING_AGENT_DIR")
        return Path(configured).expanduser() if configured else Path.home() / ".pi" / "agent"

    def detect(self) -> bool:
        return self.detect_binary() or self.detect_config()

    def detect_binary(self) -> bool:
        """Whether the Pi executable is available on PATH."""
        return shutil.which(self.binary_name) is not None

    def detect_config(self) -> bool:
        """Whether Pi has an existing settings file."""
        return any(path.is_file() for path in self.config_paths())

    def config_paths(self) -> list[Path]:
        return [self._agent_dir() / "settings.json"]

    def _extension_path(self) -> Path:
        return self._agent_dir() / "extensions" / "autoconduck.ts"

    @staticmethod
    def _model_definition(model: str, context_window: int = 1000000) -> dict[str, object]:
        """Build the minimum model description accepted by Pi.

        Unlike the OpenAI proxy API, Pi's provider configuration expects model
        descriptors rather than a list of model names.  Keeping this small
        also lets Pi apply its normal defaults for context and token limits.
        """
        return {
            "id": model,
            "name": model,
            "reasoning": True,
            "input": ["text", "image"],
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            "contextWindow": context_window,
            "maxTokens": 16384,
        }

    def _resolve_base_url(self, pi_settings, port: int) -> str:
        base_url = getattr(pi_settings, "base_url", None) if pi_settings else None
        if base_url:
            return normalize_api_base(base_url)
        return f"http://127.0.0.1:{port}/v1"

    def _resolve_api_key(self, pi_settings) -> str:
        api_key = getattr(pi_settings, "api_key", None) if pi_settings else None
        if api_key:
            return str(api_key)
        api_key_env = getattr(pi_settings, "api_key_env", None) if pi_settings else None
        if api_key_env:
            env_value = os.environ.get(api_key_env)
            if env_value:
                return env_value
        return "autoconduck-local"

    def _render_extension(self, base_url: str, api_key: str, context_window: int = 1000000) -> str:
        models = [self._model_definition(model, context_window) for model in self.PSEUDO_MODELS]
        models_json = json.dumps(models, indent=6)
        # Re-indent so the array sits correctly inside the object literal.
        indented_models = "\n".join(
            ("      " + line if idx else line)
            for idx, line in enumerate(models_json.splitlines())
        )
        return (
            "// AutoConduck-managed provider registration. Recreate with: autoconduck install pi\n"
            'import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";\n'
            "\n"
            "export default function (pi: ExtensionAPI) {\n"
            f'  pi.registerProvider("{self.provider_name}", {{\n'
            '    name: "AutoConduck",\n'
            f'    baseUrl: "{base_url}",\n'
            f'    apiKey: "{api_key}",\n'
            '    api: "openai-completions",\n'
            '    headers: {\n'
            '      "x-agent-id": "pi",\n'
            '    },\n'
            f"    models: {indented_models},\n"
            "  });\n"
            "}\n"
        )

    def install_features(self) -> list[str]:
        """Check and install any agent-specific plugins/extensions/features for Pi."""
        return []

    def patch(self, config: Config, port: int | None = None) -> None:
        effective_port = int(port if port is not None else getattr(config, "port", 11434))
        path = self.config_paths()[0]
        pi_settings = getattr(config, "pi", None)
        pseudo_model = str(
            getattr(pi_settings, "model", None) or getattr(config, "pseudo_model", None) or "autoconduck"
        )

        base_url = self._resolve_base_url(pi_settings, effective_port)
        api_key = self._resolve_api_key(pi_settings)
        context_window = int(getattr(pi_settings, "context_window", 1000000)) if pi_settings else 1000000

        extension_file = self._extension_path()
        extension_file.parent.mkdir(parents=True, exist_ok=True)
        extension_file.write_text(self._render_extension(base_url, api_key, context_window), encoding="utf-8")

        def update(data: dict) -> None:
            data["defaultProvider"] = self.provider_name
            data["defaultModel"] = pseudo_model

        self._patch_json(path, update)

    def revert(self) -> None:
        extension_file = self._extension_path()
        if extension_file.exists():
            extension_file.unlink()

        for path in self.config_paths():
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict):
                continue

            data.pop("defaultProvider", None)
            data.pop("defaultModel", None)
            data.pop("autoconduck", None)

            # Defensive cleanup for settings files written by older,
            # dead-provider-block versions of this adapter.
            providers = data.get("providers")
            if isinstance(providers, dict):
                providers.pop(self.provider_name, None)
                if providers:
                    data["providers"] = providers
                else:
                    data.pop("providers", None)
            data.pop(self.provider_name, None)

            path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
