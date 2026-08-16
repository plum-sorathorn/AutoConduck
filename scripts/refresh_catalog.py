#!/usr/bin/env python3
"""Regenerate the deterministic model catalog documentation snapshot."""
from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autoconduck.model_presets import curated_model_catalog  # noqa: E402
from autoconduck.tui.onboarding import format_price  # noqa: E402

MODEL_LINE = re.compile(r"^- (.+) \(\$(\S+) / (\S+) per 1M\)$")


def render_catalog(rows: list[dict]) -> str:
    providers: dict[str, list[dict]] = {}
    for row in rows:
        providers.setdefault(str(row["provider"]), []).append(row)
    lines = ["# Model Catalog", "", f"Total models: {len(rows)}", "", "## Models per provider"]
    for provider in sorted(providers):
        lines.append(f"- {provider}: {len(providers[provider])}")
    lines.extend(["", "## Models"])
    for provider in sorted(providers):
        lines.append(f"### {provider}")
        for row in sorted(providers[provider], key=lambda item: item["id"]):
            lines.append(
                f"- {row['id']} (${format_price(row.get('price_in', 0))} / "
                f"{format_price(row.get('price_out', 0))} per 1M)"
            )
    zero_prices = sum(
        1 for row in rows if not float(row.get("price_in", 0)) and not float(row.get("price_out", 0))
    )
    lines.extend(["", f"Models with unknown (zero) prices: {zero_prices}", ""])
    return "\n".join(lines)


def model_prices(text: str) -> dict[tuple[str, str], tuple[str, str]]:
    provider = ""
    found: dict[tuple[str, str], tuple[str, str]] = {}
    for line in text.splitlines():
        if line.startswith("### "):
            provider = line[4:]
        else:
            match = MODEL_LINE.match(line)
            if match:
                found[(provider, match.group(1))] = (match.group(2), match.group(3))
    return found


def print_diff(existing: str, generated: str) -> None:
    old, new = model_prices(existing), model_prices(generated)
    added = sorted(set(new) - set(old))
    removed = sorted(set(old) - set(new))
    changed = sorted(key for key in set(old) & set(new) if old[key] != new[key])
    price_changes = sum(1 for key in changed if old[key] != new[key])
    print(
        f"catalog stale: added {len(added)}, removed {len(removed)}, "
        f"changed {len(changed)} ids, price changes {price_changes}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check the snapshot without writing it")
    args = parser.parse_args()
    destination = ROOT / "docs" / "model_catalog.md"
    generated = render_catalog(curated_model_catalog())
    existing = destination.read_text(encoding="utf-8") if destination.exists() else ""
    if args.check:
        if existing == generated:
            rows = curated_model_catalog()
            providers = {row["provider"] for row in rows}
            zero = sum(1 for row in rows if not row.get("price_in") and not row.get("price_out"))
            print(f"catalog up to date: {len(rows)} models, {len(providers)} providers, {zero} zero-price")
            return 0
        print_diff(existing, generated)
        return 1

    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".model_catalog.", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(generated)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    rows = curated_model_catalog()
    providers = {row["provider"] for row in rows}
    zero = sum(1 for row in rows if not row.get("price_in") and not row.get("price_out"))
    print(f"catalog refreshed: {len(rows)} models, {len(providers)} providers, {zero} zero-price, {len(generated.encode('utf-8'))} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
