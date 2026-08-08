#!/usr/bin/env python3
"""Build and validate the npm wheel payloads."""
from __future__ import annotations
import argparse, hashlib, json, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ["darwin-arm64", "darwin-x64", "linux-x64", "linux-arm64", "win32-x64"]

def version() -> str:
    import tomllib
    with (ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)["project"]["version"]

def shasum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def package_dir(platform: str) -> Path:
    return ROOT / "npm-packaging" / f"autoconduck-{platform}"

def assert_version(platform: str) -> None:
    path = package_dir(platform) / "package.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != version():
        raise ValueError(f"version mismatch {path}")

def wheel_path() -> Path:
    wheels = sorted((ROOT / "npm-packaging" / "dist").glob("autoconduck-*-py3-none-any.whl"))
    if not wheels:
        raise FileNotFoundError("wheel missing; run build without --check")
    return wheels[-1]

def build_wheel() -> Path:
    dist = ROOT / "npm-packaging" / "dist"
    dist.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "-w", str(dist), "."], cwd=ROOT)
    return wheel_path()

def build_one(platform: str, wheel: Path, check: bool = False) -> None:
    assert_version(platform)
    target = package_dir(platform) / "python" / wheel.name
    if check:
        if not target.is_file():
            raise FileNotFoundError(f"wheel missing: {target}")
        actual = target
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(wheel, target)
        for stale in (package_dir(platform) / "bin" / "autoconduck", package_dir(platform) / "bin" / "autoconduck.exe"):
            stale.unlink(missing_ok=True)
        actual = target
    print(f"[build] {platform} {actual.name} sha256={shasum(actual)[:12]}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", choices=MATRIX)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    targets = [args.platform] if args.platform else MATRIX
    try:
        wheel = wheel_path() if args.check else build_wheel()
        for platform in targets:
            build_one(platform, wheel, args.check)
    except (OSError, ValueError, AssertionError) as exc:
        print(f"[build] error: {exc}", file=sys.stderr)
        raise SystemExit(1)

if __name__ == "__main__":
    main()
