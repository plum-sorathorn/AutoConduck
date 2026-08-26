"""Oh My Pi shim helpers."""
from __future__ import annotations


def omp_shim(real_bin: str) -> str:
    """Build the standard ref-counted launcher wrapper for OMP."""
    from .launcher_shims import shim_script, shim_script_win
    import os

    return shim_script_win("omp", real_bin) if os.name == "nt" else shim_script("omp", real_bin)
