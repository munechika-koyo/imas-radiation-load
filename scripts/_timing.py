"""Measure component wall-clock time independently of Slurm accounting."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from time import perf_counter
from typing import Any


@dataclass
class RenderTiming:
    """Keep cumulative observer/save durations and the complete component lifetime."""

    workers: int
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    setup_seconds: float = 0.0
    rendering_seconds: float = 0.0
    save_seconds: float = 0.0
    _started: float = field(default_factory=perf_counter, repr=False)

    def finish_setup(self) -> None:
        """Include input loading, scene/camera construction and input snapshots."""
        self.setup_seconds = perf_counter() - self._started

    def snapshot(self, *, finished: bool = False) -> dict[str, Any]:
        """Record UTC dates and monotonic elapsed seconds at a checkpoint or completion."""
        updated_at = datetime.now(UTC).isoformat()
        return dict(
            started_at=self.started_at,
            updated_at=updated_at,
            finished_at=updated_at if finished else None,
            setup_seconds=self.setup_seconds,
            rendering_seconds=self.rendering_seconds,
            save_seconds=self.save_seconds,
            total_seconds=perf_counter() - self._started,
            workers=self.workers,
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        )
