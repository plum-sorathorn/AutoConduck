from __future__ import annotations

import json
import time
import threading
from collections import deque
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from .config import logs_path, ensure_home


class RoutingEvent(BaseModel):
    ts: float
    request_id: str
    pseudo_model: str | None = None
    real_model: str = ""
    path: str  # fast, slow, ambiguous_resolved_fast, ambiguous_resolved_slow, passthrough, cache_hit, cancelled
    gate_reason: str | None = None
    T_i: float | None = None
    T_i_prime: float | None = None
    degraded_to_fast: bool = False
    cache_hit: bool = False
    cost_est: float | None = None
    latency_overhead_ms: float = 0.0
    latency_total_ms: float | None = None
    cancelled: bool = False
    error: str | None = None
    worker_ok: int | None = None
    worker_fail: int | None = None
    steps_ms: dict[str, float] | None = None


class Telemetry:
    def __init__(self, max_events: int = 500):
        self._ring: deque[RoutingEvent] = deque(maxlen=max_events)
        self._lock = threading.Lock()
        self._start_ts = time.time()
        self._total = 0

    def push(self, event: RoutingEvent) -> None:
        with self._lock:
            self._ring.append(event)
            self._total += 1
        # append to JSONL (debounced but sync for simplicity; never fail)
        try:
            ensure_home()
            p = logs_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(event.model_dump(), ensure_ascii=False)
            with open(p, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    def recent(self, n: int = 50) -> list[RoutingEvent]:
        with self._lock:
            return list(self._ring)[-n:]

    def all(self) -> list[RoutingEvent]:
        with self._lock:
            return list(self._ring)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            events = list(self._ring)
            total = self._total or len(events)
        now = time.time()
        uptime = now - self._start_ts
        if not events:
            return {
                "uptime_seconds": round(uptime, 2),
                "total_requests": total,
                "fast_path_ratio": 0.0,
                "cache_hit_ratio": 0.0,
                "avg_overhead_ms": 0.0,
                "recent_events": [],
                "degraded_models": [],
            }
        fast = sum(1 for e in events if e.path in ("fast", "ambiguous_resolved_fast"))
        cache_hits = sum(1 for e in events if e.cache_hit)
        avg_overhead = sum(e.latency_overhead_ms for e in events) / len(events)
        # degraded models (those with error)
        degraded = []
        # we surface last 50
        recent = [e.model_dump() for e in events[-50:]]
        return {
            "uptime_seconds": round(uptime, 2),
            "total_requests": total,
            "fast_path_ratio": round(fast / len(events), 3) if events else 0,
            "cache_hit_ratio": round(cache_hits / len(events), 3) if events else 0,
            "avg_overhead_ms": round(avg_overhead, 2),
            "recent_events": recent,
            "degraded_models": degraded,
        }

    def clear(self) -> None:
        with self._lock:
            self._ring.clear()


telemetry = Telemetry()
