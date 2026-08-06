"""Timing and environment capture for automated and interactive runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import statistics
import sys
from typing import Any

import numpy as np
import psutil


@dataclass
class UpdateMetric:
    reason: str
    geometry_ms: float
    submit_ms: float
    points: int
    channels: int
    samples_per_bin: int
    synchronous_frame_ms: float | None = None


@dataclass
class MetricRecorder:
    configuration: dict[str, Any]
    renderer: dict[str, Any]
    updates: list[UpdateMetric] = field(default_factory=list)
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def add(self, metric: UpdateMetric) -> None:
        self.updates.append(metric)

    @staticmethod
    def _summary(values: list[float]) -> dict[str, float] | None:
        if not values:
            return None
        array = np.asarray(values, dtype=np.float64)
        return {
            "count": int(array.size),
            "min_ms": float(array.min()),
            "median_ms": float(np.median(array)),
            "p95_ms": float(np.percentile(array, 95)),
            "max_ms": float(array.max()),
            "mean_ms": float(statistics.fmean(values)),
        }

    def to_dict(self) -> dict[str, Any]:
        process = psutil.Process(os.getpid())
        geometry = [item.geometry_ms for item in self.updates]
        submit = [item.submit_ms for item in self.updates]
        frames = [
            item.synchronous_frame_ms
            for item in self.updates
            if item.synchronous_frame_ms is not None
        ]
        return {
            "schema_version": 1,
            "started_at": self.started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "configuration": self.configuration,
            "renderer": self.renderer,
            "environment": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "python": sys.version,
                "executable": sys.executable,
                "cpu_logical_count": psutil.cpu_count(logical=True),
                "memory_total_mib": psutil.virtual_memory().total / 1024**2,
                "process_rss_mib": process.memory_info().rss / 1024**2,
            },
            "summary": {
                "geometry": self._summary(geometry),
                "submit": self._summary(submit),
                "synchronous_frame": self._summary(frames),
            },
            "updates": [asdict(item) for item in self.updates],
        }

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path
