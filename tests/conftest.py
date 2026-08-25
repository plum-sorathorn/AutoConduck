"""Global pytest fixtures for test suite isolation."""

import os
import pytest
from pathlib import Path


@pytest.fixture(autouse=True)
def isolate_autoconduck_home(tmp_path, monkeypatch):
    """Isolate all test runs to a clean temporary AUTOCONDUCK_HOME directory.
    
    This ensures that running pytest will never mutate or wipe the user's
    live ~/.autoconduck/config.yaml or auth.yaml on disk.
    """
    test_home = tmp_path / ".autoconduck"
    test_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AUTOCONDUCK_HOME", str(test_home))

    # Reset in-memory cached state in manager
    import autoconduck.config.manager as manager_mod

    manager_mod._config = None
    manager_mod._config_digest = None
    manager_mod._config_path = None
