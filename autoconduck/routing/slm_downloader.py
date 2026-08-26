"""SLM model catalog and downloader for AutoConduck onboarding and runtime."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable

from autoconduck.config import get_config, save_config

logger = logging.getLogger(__name__)

SLM_MODELS_CATALOG: list[dict[str, Any]] = [
    {
        "id": "qwen2.5-coder-0.5b-instruct",
        "key": "qwen2.5coder0.5b",
        "name": "Qwen 2.5 Coder 0.5B Instruct (Recommended)",
        "filename": "qwen2.5-coder-0.5b-instruct-q4.onnx",
        "size_mb": 350,
        "format": "onnx",
        "recommended": True,
        "url": "https://huggingface.co/onnx-community/Qwen2.5-Coder-0.5B-Instruct/resolve/main/onnx/model_q4.onnx",
        "description": "Sub-30ms ONNX accelerated task decomposition & micro-router (Default)",
    },
    {
        "id": "qwen2.5-coder-1.5b-instruct",
        "key": "qwen2.5coder1.5b",
        "name": "Qwen 2.5 Coder 1.5B Instruct",
        "filename": "qwen2.5-coder-1.5b-instruct-q4.onnx",
        "size_mb": 920,
        "format": "onnx",
        "recommended": False,
        "url": "https://huggingface.co/onnx-community/Qwen2.5-Coder-1.5B-Instruct/resolve/main/onnx/model_q4.onnx",
        "description": "High-capacity multi-file DAG planning & ONNX reasoning",
    },
    {
        "id": "lfm2.5-1.2b-instruct",
        "key": "lfm2.51.2binstruct",
        "name": "LFM 2.5 1.2B Instruct",
        "filename": "lfm2.5-1.2b-instruct-q4.onnx",
        "size_mb": 740,
        "format": "onnx",
        "recommended": False,
        "url": "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-ONNX/resolve/main/onnx/model_q4.onnx",
        "description": "Liquid AI hybrid architecture for agentic workflows & tool planning (ONNX)",
    },
    {
        "id": "none",
        "key": "none",
        "name": "Skip / Built-in Heuristic Fallback",
        "filename": "",
        "size_mb": 0,
        "format": "none",
        "recommended": False,
        "url": "",
        "description": "Zero-overhead rule-based circuit breaker (no download required)",
    },
]


def get_default_models_dir() -> Path:
    """Return the default storage directory for local SLM models."""
    models_dir = Path.home() / ".autoconduck" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def get_slm_model_info(identifier: str) -> dict[str, Any] | None:
    """Lookup SLM model metadata by ID, key, or filename."""
    norm = identifier.lower().strip()
    for m in SLM_MODELS_CATALOG:
        if norm in (m["id"].lower(), m["key"].lower(), m["filename"].lower()):
            return m
    return None


def is_slm_model_installed(identifier: str, target_dir: Path | None = None) -> bool:
    """Check if the given SLM model file exists locally and is non-empty."""
    info = get_slm_model_info(identifier)
    if not info or not info.get("filename"):
        return False
    models_dir = target_dir or get_default_models_dir()
    model_file = models_dir / info["filename"]
    return model_file.is_file() and model_file.stat().st_size > 0


def download_slm_model(
    identifier: str,
    target_dir: Path | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    chunk_size: int = 1024 * 1024,
) -> Path | None:
    """Download the specified SLM model to the local storage directory."""
    import httpx

    info = get_slm_model_info(identifier)
    if not info or not info.get("url"):
        return None

    models_dir = target_dir or get_default_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)
    target_path = models_dir / info["filename"]
    temp_path = models_dir / f"{info['filename']}.part"

    try:
        with httpx.stream("GET", info["url"], follow_redirects=True, timeout=120.0) as response:
            response.raise_for_status()
            total_bytes = int(response.headers.get("content-length", 0))
            downloaded = 0

            with open(temp_path, "wb") as f:
                for chunk in response.iter_bytes(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback:
                            progress_callback(downloaded, total_bytes)

        if temp_path.exists():
            temp_path.replace(target_path)
        return target_path
    except Exception as exc:
        logger.error("Failed to download SLM model %s: %s", identifier, exc)
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise exc


def integrate_slm_model(identifier: str, target_dir: Path | None = None) -> str:
    """Integrate chosen SLM model into the AutoConduck configuration."""
    cfg = get_config()
    info = get_slm_model_info(identifier)

    if not info or info["id"] == "none":
        # Skip / Heuristic Fallback
        cfg.selection.slm_model_path = ""
        save_config(cfg)
        return ""

    models_dir = target_dir or get_default_models_dir()
    model_path = models_dir / info["filename"]
    path_str = str(model_path)
    cfg.selection.slm_model_path = path_str
    save_config(cfg)
    return path_str
