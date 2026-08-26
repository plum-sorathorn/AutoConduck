import subprocess
import sys
import time


def test_cli_startup_has_pragmatic_ceiling():
    # Design targets are 50ms for --version and 30ms for --help; this ceiling
    # allows normal interpreter/process variance on CI.
    for option in ("--version", "--help"):
        started = time.perf_counter()
        result = subprocess.run([sys.executable, "-m", "autoconduck", option], capture_output=True, text=True)
        assert result.returncode == 0
        assert time.perf_counter() - started < 1.000
