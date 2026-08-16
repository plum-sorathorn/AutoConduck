"""Bounded, deterministic file context for the fast path."""

import asyncio
import logging
import re
from pathlib import Path


_PATH_RE = re.compile(
    r"[\w./\\-]+\.(?:py|js|ts|json|toml|md|yaml|yml|txt|ini|cfg|html|css|sh|ps1|tsx|jsx)\b",
    re.IGNORECASE,
)
_EXCLUDED = {"build", "graphify-out", ".git", "__pycache__", "node_modules"}
_log = logging.getLogger("autoconduck")


def _read_head(path: Path, size: int) -> bytes:
    with path.open("rb") as handle:
        return handle.read(size)


def _value(config, name: str, default):
    return getattr(config, name, default)


def _candidate_paths(messages: list[dict], base: Path, maximum: int) -> list[Path]:
    users = [m for m in messages if isinstance(m, dict) and m.get("role") == "user"]
    if not users:
        return []
    content = users[-1].get("content", "")
    if not isinstance(content, str):
        return []
    result = []
    seen = set()
    for token in _PATH_RE.findall(content):
        token = token.strip("`),.:;")
        path = Path(token)
        if not path.is_absolute():
            path = base / path
        try:
            resolved = path.resolve()
            parts = {part.lower() for part in resolved.relative_to(base.resolve()).parts}
            if parts & _EXCLUDED or not resolved.is_file():
                continue
        except (OSError, ValueError):
            continue
        if resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
            if len(result) >= maximum:
                break
    return result


async def _read_one(path: Path, size: int, timeout: float, lines: int):
    raw = await asyncio.wait_for(asyncio.to_thread(_read_head, path, size), timeout)
    if b"\x00" in raw:
        return None
    text = raw.decode("utf-8", errors="ignore")
    return "\n".join(text.splitlines()[:lines]).rstrip()


async def maybe_digest_messages(messages: list[dict], config=None, base_dir=None,
                                request=None) -> list[dict] | None:
    """Return a small system digest, or ``None`` for any unsafe/slow case."""
    try:
        if config is None:
            from .config import get_config
            config = get_config()
        if not _value(config, "fast_path_digest_enabled", True):
            return None
        if any(isinstance(m, dict) and (m.get("role") in ("assistant", "tool", "function")
                                       or "tool_calls" in m) for m in messages):
            return None
        if request is not None and await request.is_disconnected():
            return None
        minimum = int(_value(config, "fast_path_digest_min_files", 2))
        maximum = int(_value(config, "fast_path_digest_max_files", 4))
        base = Path(base_dir) if base_dir is not None else Path.cwd()
        paths = _candidate_paths(messages, base, maximum)
        if len(paths) < minimum:
            return None
        timeout = int(_value(config, "fast_path_digest_timeout_ms", 150)) / 1000
        coroutines = [_read_one(path, int(_value(config, "fast_path_digest_max_bytes", 8192)),
                                timeout, int(_value(config, "fast_path_digest_max_lines", 40)))
                      for path in paths]
        try:
            contents = await asyncio.wait_for(asyncio.gather(*coroutines, return_exceptions=True),
                                               timeout=timeout + 0.05)
        except Exception:
            return None
        entries = []
        for path, content in zip(paths, contents):
            if isinstance(content, str) and content:
                try:
                    name = path.relative_to(base.resolve()).as_posix()
                except ValueError:
                    name = str(path)
                entries.append([name, content])
        if len(entries) < minimum:
            return None
        limit = int(_value(config, "fast_path_digest_max_total_bytes", 12000))
        while entries:
            text = "[AutoConduck file digests]\n" + "\n\n".join(
                f"### {name}\n{content}" for name, content in entries)
            if len(text) <= limit:
                return [{"role": "system", "content": text}]
            if len(entries) > minimum:
                entries.pop()
                continue
            name, content = entries[-1]
            excess = len(text) - limit
            if excess >= len(content):
                return None
            entries[-1][1] = content[:-excess].rstrip()
        return None
    except Exception as exc:
        _log.debug("fast path digest skipped: %s", exc)
        return None
