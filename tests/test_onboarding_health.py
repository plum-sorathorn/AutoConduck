import pytest

from autoconduck.tui.onboarding.health import build_health_matrix
from autoconduck.tui.onboarding.slm_progress import download_progress


def test_download_progress_handles_unknown_length_and_completion():
    assert download_progress(12, 0).percent is None
    assert download_progress(50, 100).percent == 50
    assert download_progress(100, 100).status == "complete"


def test_health_matrix_marks_missing_credentials_without_io():
    matrix = build_health_matrix(
        [{"provider": "openai", "api_key": ""}, {"provider": "ollama", "api_key": "token"}],
        {"ollama": {"reachable": True, "latency_ms": 4.2}},
        [11434],
    )
    assert matrix["providers"][0] == {
        "name": "openai", "credential_present": False, "reachable": False,
        "latency_ms": None, "action": "configure token credential",
    }
    assert matrix["providers"][1]["latency_ms"] == 4.2
    assert matrix["local_ports"][0]["port"] == 11434
