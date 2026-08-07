from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from .config import home_dir


def cache_dir() -> Path:
    return home_dir() / "cache"


def _hash_key(pseudo: str, last_content: str) -> str:
    h = hashlib.sha256(f"{pseudo}||{last_content}".encode("utf-8")).hexdigest()
    return h[:32]


def make_key(pseudo: str, last_message: Any) -> str:
    if isinstance(last_message, dict):
        content = last_message.get("content", "")
    else:
        content = getattr(last_message, "content", str(last_message))
    if isinstance(content, list):
        content = " ".join(str(x.get("text", x) if isinstance(x, dict) else str(x)) for x in content)
    return _hash_key(pseudo, str(content))


def get(key: str) -> bytes | None:
    try:
        p = cache_dir() / f"{key}.json"
        if not p.exists():
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        # check TTL maybe? no expiry for now
        body_b64 = data.get("body")
        if body_b64 is None:
            return None
        import base64

        return base64.b64decode(body_b64.encode("ascii"))
    except Exception:
        return None


def put(key: str, body: bytes) -> None:
    try:
        cache_dir().mkdir(parents=True, exist_ok=True)
        import base64

        data = {"body": base64.b64encode(body).decode("ascii"), "ts": time.time()}
        p = cache_dir() / f"{key}.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        _enforce_cap()
    except Exception:
        pass


def _enforce_cap(limit_mb: int = 100) -> None:
    try:
        files = sorted(cache_dir().glob("*.json"), key=lambda p: p.stat().st_mtime)
        total = sum(p.stat().st_size for p in files)
        limit = limit_mb * 1024 * 1024
        idx = 0
        while total > limit and idx < len(files):
            total -= files[idx].stat().st_size
            try:
                files[idx].unlink()
            except Exception:
                pass
            idx += 1
    except Exception:
        pass


def clear() -> None:
    try:
        for p in cache_dir().glob("*.json"):
            p.unlink()
    except Exception:
        pass
