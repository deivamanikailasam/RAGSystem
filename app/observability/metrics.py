"""Minimal, dependency-free metrics registry.

Tracks counters and latency summaries in-process and exposes them via
``/metrics``. It is intentionally tiny — in production you would export these
to Prometheus/OpenTelemetry (see docs/05-evaluation-and-monitoring.md), but the
same call sites (`increment`, `observe`, `timer`) map cleanly onto those SDKs.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = {}
        self._summaries: dict[str, list[float]] = {}

    def increment(self, name: str, amount: float = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + amount

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            # Cap retained samples to bound memory; a real backend aggregates.
            samples = self._summaries.setdefault(name, [])
            samples.append(value)
            if len(samples) > 1000:
                del samples[: len(samples) - 1000]

    @contextmanager
    def timer(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.observe(name, (time.perf_counter() - start) * 1000)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            summaries = {}
            for name, samples in self._summaries.items():
                if not samples:
                    continue
                ordered = sorted(samples)
                summaries[name] = {
                    "count": len(ordered),
                    "avg": round(sum(ordered) / len(ordered), 3),
                    "p50": round(_percentile(ordered, 0.50), 3),
                    "p95": round(_percentile(ordered, 0.95), 3),
                    "p99": round(_percentile(ordered, 0.99), 3),
                }
            return {
                "counters": dict(self._counters),
                "summaries": summaries,
            }


def _percentile(ordered: list[float], q: float) -> float:
    if not ordered:
        return 0.0
    idx = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
    return ordered[idx]


# Process-wide singleton.
METRICS = Metrics()
