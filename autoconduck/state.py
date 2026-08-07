from __future__ import annotations

import json
import time
import hashlib
from collections import deque, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from pydantic import BaseModel

from .config import state_path, ensure_home


# ---------------------------------------------------------------------------
# TurnState
# ---------------------------------------------------------------------------

class TurnState(BaseModel):
    session_key: str
    last_T: float | None = None
    used_reasoning_tier: bool = False
    ts: float = 0.0


# ---------------------------------------------------------------------------
# In-memory stores
# ---------------------------------------------------------------------------

class SessionStore:
    """LRU 200, TTL 30m for TurnState."""

    def __init__(self, max_size: int = 200, ttl_s: int = 1800):
        self._data: OrderedDict[str, TurnState] = OrderedDict()
        self.max_size = max_size
        self.ttl_s = ttl_s

    def get(self, key: str) -> TurnState | None:
        now = time.time()
        v = self._data.get(key)
        if v is None:
            return None
        if now - v.ts > self.ttl_s:
            self._data.pop(key, None)
            return None
        # move to end (LRU)
        self._data.move_to_end(key)
        return v

    def put(self, state: TurnState) -> None:
        self._data[state.session_key] = state
        self._data.move_to_end(state.session_key)
        # prune TTL
        now = time.time()
        to_del = [k for k, v in self._data.items() if now - v.ts > self.ttl_s]
        for k in to_del:
            self._data.pop(k, None)
        while len(self._data) > self.max_size:
            self._data.popitem(last=False)

    def all(self) -> dict[str, TurnState]:
        return dict(self._data)


class EMAState:
    def __init__(self, alpha: float = 0.1, default_out: int = 800):
        self.alpha = alpha
        self.default_out = default_out
        self._values: Dict[str, float] = {}

    def get(self, intent: str) -> float:
        return self._values.get(intent, self._values.get("default", float(self.default_out)))

    def update(self, intent: str, actual_out: int) -> None:
        prev = self._values.get(intent, float(self.default_out))
        self._values[intent] = self.alpha * actual_out + (1 - self.alpha) * prev

    def to_dict(self) -> dict:
        return dict(self._values)

    def load_dict(self, d: dict) -> None:
        self._values = {k: float(v) for k, v in (d or {}).items()}


@dataclass
class ErrorWindow:
    events: deque = field(default_factory=deque)  # (ts, is_error)

    def record(self, is_error: bool, now: float | None = None) -> None:
        now = now or time.time()
        self.events.append((now, is_error))
        self.prune(now)

    def prune(self, now: float | None = None) -> None:
        now = now or time.time()
        cutoff = now - 300  # 5 min
        while self.events and self.events[0][0] < cutoff:
            self.events.popleft()

    def error_rate(self, now: float | None = None) -> float:
        now = now or time.time()
        self.prune(now)
        if not self.events:
            return 0.0
        total = len(self.events)
        errs = sum(1 for _, e in self.events if e)
        return errs / total if total else 0.0

    def to_list(self) -> list:
        return [{"ts": ts, "ok": not is_err} for ts, is_err in self.events]

    def load_list(self, lst: list) -> None:
        self.events = deque((item["ts"], not item["ok"]) for item in (lst or []))


# ---------------------------------------------------------------------------
# Global singletons
# ---------------------------------------------------------------------------
_sessions = SessionStore()
_ema = EMAState()
_error_windows: Dict[str, ErrorWindow] = {}
_record_count = 0


def get_session_store() -> SessionStore:
    return _sessions


def get_ema() -> EMAState:
    return _ema


def get_error_window(model_id: str) -> ErrorWindow:
    if model_id not in _error_windows:
        _error_windows[model_id] = ErrorWindow()
    return _error_windows[model_id]


def is_degraded(model_id: str, now: float | None = None) -> bool:
    w = get_error_window(model_id)
    if len(w.events) < 5:
        return False
    return w.error_rate(now) > 0.20


def record_usage(model_id: str, actual_in: int, actual_out: int, intent: str = "default") -> None:
    get_error_window(model_id).record(False)
    get_ema().update(intent, actual_out)
    _increment_flush_counter()


def record_error(model_id: str) -> None:
    get_error_window(model_id).record(True)
    _increment_flush_counter()


def session_key_from_request(chat, headers: dict | None = None) -> str:
    headers = headers or {}
    sid = headers.get("x-session-id") or headers.get("X-Session-Id")
    if sid:
        return sid[:64]
    # hash of system + first user
    parts: list[str] = []
    for m in getattr(chat, "messages", []):
        role = getattr(m, "role", "")
        content = getattr(m, "content", "")
        if isinstance(content, list):
            content = " ".join(str(c.get("text", c) if isinstance(c, dict) else str(c)) for c in content)
        if role == "system":
            parts.append(str(content))
            break
    for m in getattr(chat, "messages", []):
        if getattr(m, "role", "") == "user":
            c = getattr(m, "content", "")
            if isinstance(c, list):
                c = " ".join(str(x.get("text", x) if isinstance(x, dict) else str(x)) for x in c)
            parts.append(str(c))
            break
    if not parts:
        return "anon"
    h = hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()[:16]
    return h


def update_turn_state(session_key: str, T_prime: float, tier: str) -> None:
    used_reasoning = (T_prime >= 0.80) or (tier == "reasoning")
    st = TurnState(session_key=session_key, last_T=T_prime, used_reasoning_tier=used_reasoning, ts=time.time())
    _sessions.put(st)


def _increment_flush_counter() -> None:
    global _record_count
    _record_count += 1
    if _record_count % 10 == 0:
        flush()


def flush(path: Path | None = None) -> None:
    p = path or state_path()
    try:
        ensure_home()
        data = {
            "ema": _ema.to_dict(),
            "sessions": {k: v.model_dump() for k, v in _sessions.all().items()},
            "error_windows": {k: v.to_list() for k, v in _error_windows.items()},
        }
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(p)
    except Exception:
        pass  # never fail request


def load_state(path: Path | None = None) -> None:
    p = path or state_path()
    if not p.exists():
        return
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        _ema.load_dict(data.get("ema", {}))
        for k, v in (data.get("sessions") or {}).items():
            try:
                _sessions.put(TurnState.model_validate(v))
            except Exception:
                continue
        for k, lst in (data.get("error_windows") or {}).items():
            w = get_error_window(k)
            w.load_list(lst)
    except Exception:
        pass
