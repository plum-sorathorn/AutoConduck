"""Small, dependency-tolerant helpers for extracting and repairing JSON text."""

import json
import ast
import re
from typing import Any


def _extract_json(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    return text[start:] if depth > 0 else None


def parse_json_text(text: str) -> tuple[Any | None, str | None, str]:
    preview = str(text or "")[:200]
    candidate = str(text or "").strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        candidate = "\n".join(lines[1:-1] if lines and lines[-1].strip().startswith("```") else lines[1:])
    extracted = _extract_json(candidate)
    if extracted is None:
        return None, "no JSON object found", preview
    candidate = extracted
    last_error: Exception | None = None
    try:
        return json.loads(candidate), None, preview
    except Exception as exc:
        last_error = exc
    try:
        import json_repair
        repaired = json_repair.loads(candidate)
        if repaired is not None:
            return repaired, f"json_repair: {last_error}", preview
    except Exception as exc:
        last_error = exc
    # Built-in fallback for installations without json-repair. First close an
    # interrupted string, then try every reasonable closing-bracket sequence.
    repaired_candidate = candidate
    quote_count = 0
    escaped = False
    for char in candidate:
        if char == "\\" and not escaped:
            escaped = True
            continue
        if char == '"' and not escaped:
            quote_count += 1
        escaped = False
    if quote_count % 2:
        repaired_candidate += '"'
    for total in range(1, 9):
        for arrays in range(total + 1):
            objects = total - arrays
            suffix = "]" * arrays + "}" * objects
            try:
                return json.loads(repaired_candidate + suffix), f"bracket-close: {last_error}", preview
            except Exception as exc:
                last_error = exc
    # Remove trailing commas as a final built-in compatibility fallback.
    literal_candidate = re.sub(r",\s*([}\]])", r"\1", repaired_candidate)
    try:
        value = ast.literal_eval(literal_candidate)
        if isinstance(value, (dict, list)):
            return value, f"literal-repair: {last_error}", preview
    except Exception:
        pass
    return None, str(last_error), preview
