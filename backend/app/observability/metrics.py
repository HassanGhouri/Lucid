"""
Lightweight in-process latency metrics for Lucid.

Per-stage timings are captured with `Timer` (a time.perf_counter context
manager), aggregated per request, and rolled into an in-process ring
buffer for P50/P95 reporting via /metrics.

Note: each Uvicorn worker has its own buffer. For multi-worker production
metrics, push to a real time-series store later. For benchmarking, hitting
a single worker (or using scripts/benchmark.py) is enough.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any

logger = logging.getLogger(__name__)

# Configure root logging only if nothing upstream has done so. Under uvicorn
# the server installs its own handlers, and we don't want a module-import
# side effect overriding them. For ad-hoc CLI use (e.g. an eval script that
# transitively imports this module without a configured logger), this keeps
# INFO-level output visible.
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

# Stage names tracked across the request path. Order matters for log output.
STAGE_NAMES = ("rewrite", "embed_query", "retrieve", "rerank", "generate", "judge")

_BUFFER_SIZE = 500

_lock = threading.Lock()
_buffer: deque[dict[str, float]] = deque(maxlen=_BUFFER_SIZE)
_cold_start_ms: float | None = None


class Timer:
    """
    Context manager that records elapsed milliseconds via time.perf_counter.

    `elapsed_ms` is readable both inside and outside the context: while
    inside, it reflects time since __enter__; after __exit__, it freezes
    at the final value.
    """

    __slots__ = ("_start", "_end")

    def __init__(self) -> None:
        self._start: float = 0.0
        self._end: float | None = None

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        self._end = None
        return self

    def __exit__(self, *exc: Any) -> None:
        self._end = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        end = self._end if self._end is not None else time.perf_counter()
        return (end - self._start) * 1000.0


def set_cold_start_ms(value: float) -> None:
    """Record the one-time cold-start duration captured in the lifespan hook."""
    global _cold_start_ms
    _cold_start_ms = float(value)
    logger.info("cold_start_ms=%.1f", value)


def get_cold_start_ms() -> float | None:
    return _cold_start_ms


def record_request(timings: dict[str, float]) -> None:
    """
    Append a completed request's timings and log a structured line.

    Missing stages (e.g. judge skipped on refusal) are tolerated by the
    percentile calc, which skips records that lack the stage in question.
    """
    safe = {k: float(v) for k, v in timings.items() if v is not None}
    with _lock:
        _buffer.append(safe)
    parts = " ".join(f"{k}={safe[k]:.1f}" for k in safe)
    logger.info("request_timings %s", parts)


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = max(0, min(len(values) - 1, int(round((pct / 100.0) * (len(values) - 1)))))
    return values[idx]


def snapshot() -> dict[str, Any]:
    """Return cold-start and per-stage P50/P95 across the current ring buffer."""
    with _lock:
        records = list(_buffer)

    stages = list(STAGE_NAMES) + ["total"]
    out_stages: dict[str, dict[str, float | int | None]] = {}
    for stage in stages:
        values = [r[stage] for r in records if stage in r]
        out_stages[stage] = {
            "count": len(values),
            "p50_ms": _percentile(values, 50),
            "p95_ms": _percentile(values, 95),
        }

    return {
        "cold_start_ms": _cold_start_ms,
        "request_count": len(records),
        "buffer_size": _BUFFER_SIZE,
        "stages": out_stages,
    }