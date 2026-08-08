"""Deterministic, bounded merge of analyst reports."""

import re


_REF = re.compile(r"\b(?:[\w./\\-]+\.(?:py|js|ts|tsx|go|rs|java|md|json|yaml|yml)):\d+\b")


def compact(outputs: list[str]) -> str:
    seen_refs: set[str] = set()
    lines: list[str] = []
    for output in outputs:
        for line in output.splitlines():
            refs = _REF.findall(line)
            if refs and all(ref in seen_refs for ref in refs):
                continue
            for ref in refs:
                seen_refs.add(ref)
            if line.strip() and line.strip() not in lines:
                lines.append(line.strip())
    # Approximate 1k-token cap while preserving complete lines and references.
    result: list[str] = []
    size = 0
    for line in lines:
        words = line.split()
        if size + len(words) > 950:
            break
        result.append(line)
        size += len(words)
    return "\n".join(result)
