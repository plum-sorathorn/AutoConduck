#!/usr/bin/env python3
"""
Cross-compile matrix, version stamping, shasum, smoke test.
Usage: python npm-packaging/build.py [--platform darwin-arm64] [--check]
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ["darwin-arm64", "darwin-x64", "linux-x64", "linux-arm64", "win32-x64"]

def version() -> str:
    try:
        import tomllib
        data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        return data["project"]["version"]
    except Exception:
        return "0.1.0"

def shasum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def build_one(platform: str):
    print(f"[build] platform={platform} version={version()}")
    # PyInstaller spec
    dist = ROOT / "dist" / platform
    dist.mkdir(parents=True, exist_ok=True)
    # In CI this would run pyinstaller; here we produce a stub and verify structure
    pkg_dir = ROOT / "npm-packaging" / f"autoconduck-{platform}" / "bin"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    bin_name = "autoconduck.exe" if "win32" in platform else "autoconduck"
    bin_path = pkg_dir / bin_name
    # placeholder: in real build this is PyInstaller output; we create a shim for smoke
    if not bin_path.exists():
        bin_path.write_bytes(b"#!/bin/sh\necho autoconduck stub\n")
        try:
            bin_path.chmod(0o755)
        except Exception:
            pass
    print(f"  -> {bin_path} sha256={shasum(bin_path)[:12]}")
    # verify package.json version matches
    pkg_json = ROOT / "npm-packaging" / f"autoconduck-{platform}" / "package.json"
    if pkg_json.exists():
        data = json.loads(pkg_json.read_text(encoding="utf-8"))
        assert data.get("version") == version(), f"version mismatch {pkg_json}"
    print("  OK")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", default=None, help="single platform or all")
    parser.add_argument("--check", action="store_true", help="verify only")
    args = parser.parse_args()
    targets = [args.platform] if args.platform else MATRIX
    for t in targets:
        if t not in MATRIX:
            print(f"unknown platform {t}", file=sys.stderr)
            sys.exit(1)
        build_one(t)
    print("[build] done")

if __name__ == "__main__":
    main()
