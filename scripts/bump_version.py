#!/usr/bin/env python3
"""Single-source version bumper for AutoConduck.

Updates the authoritative version in pyproject.toml (or autoconduck/__init__.py)
and synchronizes all npm packaging files and documentation.

Usage:
    python scripts/bump_version.py 0.3.3
    python scripts/bump_version.py --patch
    python scripts/bump_version.py --minor
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
INIT_FILE = ROOT / "autoconduck" / "__init__.py"
NPM_DIR = ROOT / "npm-packaging"

PLATFORMS = ["darwin-arm64", "darwin-x64", "linux-x64", "linux-arm64", "win32-x64"]


def get_current_version() -> str:
    content = PYPROJECT.read_text(encoding="utf-8")
    m = re.search(r'version\s*=\s*"([^"]+)"', content)
    if not m:
        raise ValueError("Could not find version in pyproject.toml")
    return m.group(1)


def bump_semver(current: str, part: str) -> str:
    parts = current.split(".")
    major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2].split("-")[0])
    if part == "major":
        return f"{major + 1}.0.0"
    elif part == "minor":
        return f"{major}.{minor + 1}.0"
    elif part == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise ValueError(f"Unknown bump part: {part}")


def update_pyproject(new_version: str) -> None:
    content = PYPROJECT.read_text(encoding="utf-8")
    updated = re.sub(
        r'version\s*=\s*"[^"]+"',
        f'version = "{new_version}"',
        content,
        count=1,
    )
    PYPROJECT.write_text(updated, encoding="utf-8")
    print(f"  [OK] Updated pyproject.toml -> {new_version}")


def update_init(new_version: str) -> None:
    content = INIT_FILE.read_text(encoding="utf-8")
    updated = re.sub(
        r'__version__\s*=\s*"[^"]+"',
        f'__version__ = "{new_version}"',
        content,
        count=1,
    )
    INIT_FILE.write_text(updated, encoding="utf-8")
    print(f"  [OK] Updated autoconduck/__init__.py -> {new_version}")


def update_npm_packages(new_version: str) -> None:
    # Main package.json
    main_pkg = NPM_DIR / "autoconduck" / "package.json"
    if main_pkg.is_file():
        data = json.loads(main_pkg.read_text(encoding="utf-8"))
        data["version"] = new_version
        if "optionalDependencies" in data:
            for k in list(data["optionalDependencies"].keys()):
                data["optionalDependencies"][k] = new_version
        main_pkg.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(f"  [OK] Updated {main_pkg.relative_to(ROOT)} -> {new_version}")

    # Platform package.json files
    for platform in PLATFORMS:
        pkg_file = NPM_DIR / f"autoconduck-{platform}" / "package.json"
        if pkg_file.is_file():
            data = json.loads(pkg_file.read_text(encoding="utf-8"))
            data["version"] = new_version
            pkg_file.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            print(f"  [OK] Updated {pkg_file.relative_to(ROOT)} -> {new_version}")


def update_docs(new_version: str) -> None:
    # AGENTS.md
    agents_file = ROOT / "AGENTS.md"
    if agents_file.is_file():
        content = agents_file.read_text(encoding="utf-8")
        updated = re.sub(r'Project:\s*\*\*AutoConduck\*\*\s*\(`[^`]+`\s*in\s*`pyproject\.toml`\)', f'Project: **AutoConduck** (`{new_version}` in `pyproject.toml`)', content)
        agents_file.write_text(updated, encoding="utf-8")
        print(f"  [OK] Updated AGENTS.md -> {new_version}")

    # README.md
    readme_file = ROOT / "README.md"
    if readme_file.is_file():
        content = readme_file.read_text(encoding="utf-8")
        updated = re.sub(r'# AutoConduck\s+\d+\.\d+\.\d+(?:-[a-zA-Z0-9.]+)?', f'# AutoConduck {new_version}', content)
        updated = re.sub(r'\*\*AutoConduck\s+\d+\.\d+\.\d+(?:-[a-zA-Z0-9.]+)?\*\*', f'**AutoConduck {new_version}**', updated)
        readme_file.write_text(updated, encoding="utf-8")
        print(f"  [OK] Updated README.md -> {new_version}")

    # PROJECT.md
    project_file = ROOT / "PROJECT.md"
    if project_file.is_file():
        content = project_file.read_text(encoding="utf-8")
        updated = re.sub(r'# Project:\s*AutoConduck\s+\d+\.\d+\.\d+(?:-[a-zA-Z0-9.]+)?', f'# Project: AutoConduck {new_version}', content)
        updated = re.sub(r'AutoConduck\s+\d+\.\d+\.\d+(?:-[a-zA-Z0-9.]+)?\s+transforms', f'AutoConduck {new_version} transforms', updated)
        project_file.write_text(updated, encoding="utf-8")
        print(f"  [OK] Updated PROJECT.md -> {new_version}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Synchronize version across all AutoConduck files")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("version", nargs="?", help="Explicit new version (e.g. 0.3.4)")
    group.add_argument("--patch", action="store_true", help="Bump patch version (e.g. 0.3.3 -> 0.3.4)")
    group.add_argument("--minor", action="store_true", help="Bump minor version (e.g. 0.3.3 -> 0.4.0)")
    group.add_argument("--major", action="store_true", help="Bump major version (e.g. 0.3.3 -> 1.0.0)")
    group.add_argument("--current", action="store_true", help="Print current version and exit")
    parser.add_argument("--sync", action="store_true", help="Sync npm packages to match pyproject.toml without bumping")

    args = parser.parse_args()
    current = get_current_version()

    if args.current:
        print(f"AutoConduck version: {current}")
        return

    if args.sync:
        new_version = current
        print(f"Synchronizing all files to current version {new_version}...")
    elif args.patch:
        new_version = bump_semver(current, "patch")
    elif args.minor:
        new_version = bump_semver(current, "minor")
    elif args.major:
        new_version = bump_semver(current, "major")
    elif args.version:
        new_version = args.version
    else:
        parser.print_help()
        sys.exit(1)

    print(f"Bumping AutoConduck version: {current} -> {new_version}")
    update_pyproject(new_version)
    update_init(new_version)
    update_npm_packages(new_version)
    update_docs(new_version)
    print(f"\nSuccessfully bumped and synchronized to {new_version}!")


if __name__ == "__main__":
    main()
