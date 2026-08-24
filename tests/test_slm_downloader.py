"""Tests for SLM model catalog, downloader, and onboarding integration."""
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from autoconduck.config import Config, get_config, save_config
from autoconduck.routing.slm_downloader import (
    SLM_MODELS_CATALOG,
    download_slm_model,
    get_slm_model_info,
    integrate_slm_model,
    is_slm_model_installed,
)
from autoconduck.tui.onboarding.helpers import render_slm_rows
from autoconduck.tui.onboarding.screens_slm import SLMSetupScreen


def test_slm_models_catalog_entries():
    ids = {m["id"] for m in SLM_MODELS_CATALOG}
    assert "qwen2.5-coder-0.5b-instruct" in ids
    assert "qwen2.5-coder-1.5b-instruct" in ids
    assert "lfm2.5-1.2b-instruct" in ids
    assert "none" in ids

    # Verify default recommended model
    recommended = [m for m in SLM_MODELS_CATALOG if m.get("recommended")]
    assert len(recommended) == 1
    assert recommended[0]["id"] == "qwen2.5-coder-0.5b-instruct"
    assert "(Recommended)" in recommended[0]["name"]


def test_get_slm_model_info_lookups():
    # By ID
    m1 = get_slm_model_info("qwen2.5-coder-0.5b-instruct")
    assert m1 is not None
    assert m1["filename"] == "qwen2.5-coder-0.5b-instruct-q4.onnx"

    # By short key
    m2 = get_slm_model_info("qwen2.5coder1.5b")
    assert m2 is not None
    assert m2["id"] == "qwen2.5-coder-1.5b-instruct"

    m3 = get_slm_model_info("lfm2.51.2binstruct")
    assert m3 is not None
    assert m3["id"] == "lfm2.5-1.2b-instruct"

    # Non-existent
    assert get_slm_model_info("unknown-model-xyz") is None


def test_is_slm_model_installed(tmp_path):
    assert is_slm_model_installed("qwen2.5-coder-0.5b-instruct", target_dir=tmp_path) is False

    # Create dummy model file
    model_file = tmp_path / "qwen2.5-coder-0.5b-instruct-q4.onnx"
    model_file.write_bytes(b"ONNX_MOCK_DATA")

    assert is_slm_model_installed("qwen2.5-coder-0.5b-instruct", target_dir=tmp_path) is True
    assert is_slm_model_installed("qwen2.5coder0.5b", target_dir=tmp_path) is True
    assert is_slm_model_installed("none", target_dir=tmp_path) is False


def test_render_slm_rows(tmp_path):
    rendered = render_slm_rows(
        SLM_MODELS_CATALOG,
        selected_id="qwen2.5-coder-0.5b-instruct",
        cursor=0,
        target_dir=tmp_path,
    )
    assert "> [x] Qwen 2.5 Coder 0.5B Instruct (Recommended)" in rendered
    assert "Qwen 2.5 Coder 1.5B Instruct" in rendered
    assert "LFM 2.5 1.2B Instruct" in rendered
    assert "Skip / Built-in Heuristic Fallback" in rendered

    # With model installed
    model_file = tmp_path / "qwen2.5-coder-0.5b-instruct-q4.onnx"
    model_file.write_bytes(b"ONNX_MOCK_DATA")

    rendered_installed = render_slm_rows(
        SLM_MODELS_CATALOG,
        selected_id="qwen2.5-coder-0.5b-instruct",
        cursor=0,
        target_dir=tmp_path,
    )
    assert "[Installed]" in rendered_installed


def test_integrate_slm_model(tmp_path):
    path = integrate_slm_model("qwen2.5-coder-0.5b-instruct", target_dir=tmp_path)
    cfg = get_config()
    assert str(tmp_path / "qwen2.5-coder-0.5b-instruct-q4.onnx") == path
    assert cfg.selection.slm_model_path == path

    # Integrate Skip / None
    none_path = integrate_slm_model("none", target_dir=tmp_path)
    cfg2 = get_config()
    assert none_path == ""
    assert cfg2.selection.slm_model_path == ""


def test_download_slm_model_mocked(tmp_path):
    mock_response = MagicMock()
    mock_response.headers = {"content-length": "20"}
    mock_response.iter_bytes.return_value = [b"chunk1_", b"chunk2_", b"chunk3"]
    mock_response.__enter__.return_value = mock_response

    with patch("httpx.stream", return_value=mock_response):
        progress_records = []

        def on_progress(down, tot):
            progress_records.append((down, tot))

        target = download_slm_model(
            "qwen2.5-coder-0.5b-instruct",
            target_dir=tmp_path,
            progress_callback=on_progress,
        )

        assert target is not None
        assert target.exists()
        assert target.read_bytes() == b"chunk1_chunk2_chunk3"
        assert len(progress_records) > 0


def test_slm_setup_screen_key_navigation(tmp_path):
    controller = MagicMock()
    screen = SLMSetupScreen(controller, target_dir=tmp_path)

    # Initial state
    assert screen.cursor == 0
    assert screen.selected_id == "qwen2.5-coder-0.5b-instruct"

    # Move down
    down_event = MagicMock()
    down_event.key = "down"
    screen.on_key(down_event)
    assert screen.cursor == 1

    # Select with space
    space_event = MagicMock()
    space_event.key = "space"
    screen.on_key(space_event)
    assert screen.selected_id == "qwen2.5-coder-1.5b-instruct"

    # Move down to LFM
    screen.on_key(down_event)
    screen.on_key(space_event)
    assert screen.selected_id == "lfm2.5-1.2b-instruct"

    # Back key
    left_event = MagicMock()
    left_event.key = "left"
    screen.on_key(left_event)
    controller.pop_screen.assert_called_once()
