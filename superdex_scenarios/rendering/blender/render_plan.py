"""Pure-Python planning helpers for parallel Blender renders.

This module intentionally has no Blender dependency so the orchestration layer
and unit tests can use the same frame partitioning logic as the Blender script.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FramePartition:
    """The half-open slice assigned to one render worker."""

    total: int
    worker_index: int
    worker_count: int
    start: int
    stop: int

    @property
    def count(self) -> int:
        return self.stop - self.start


def partition_frames(total: int, worker_index: int = 0, worker_count: int = 1) -> FramePartition:
    """Split ``total`` ordered frames into balanced contiguous worker slices."""

    if total < 0:
        raise ValueError("total must be non-negative")
    if worker_count < 1:
        raise ValueError("worker_count must be positive")
    if not 0 <= worker_index < worker_count:
        raise ValueError("worker_index must be within worker_count")

    start = total * worker_index // worker_count
    stop = total * (worker_index + 1) // worker_count
    return FramePartition(total, worker_index, worker_count, start, stop)
