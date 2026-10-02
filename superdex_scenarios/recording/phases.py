"""Semantic task-phase log shared by every embodiment.

Paired robot/human episodes are aligned by *phase*, not by frame.  The log
records, for each named phase, the simulation step and time at its start and
end together with the embodiment's grasp-point pose and the object position, so
a dataset consumer can compare how two embodiments perform the same step.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

PHASE_SEQUENCE = (
    "home",
    "preshape",
    "approach",
    "pre_grasp",
    "grasp",
    "lift",
    "carry",
    "lower",
    "release",
    "retreat",
    "return_home",
)


@dataclass
class PhaseSample:
    step: int
    time_s: float
    grasp_point_world_m: list[float]
    grasp_point_quaternion_xyzw: list[float]
    object_position_m: list[float]


@dataclass
class PhaseRecord:
    name: str
    start: PhaseSample
    end: PhaseSample | None = None


@dataclass
class PhaseLog:
    """Collect phase boundaries from a sampler callable.

    ``sampler`` returns ``(step, time_s, grasp_position, grasp_quaternion_xyzw,
    object_position)`` for the current simulation state.  ``phase_sequence``
    defaults to the ball-and-bowl :data:`PHASE_SEQUENCE`; other tasks pass
    their own ordered phase names.
    """

    FORMAT = "superdex-task-phases-v1"

    embodiment: str
    sampler: Callable[[], tuple[int, float, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]]
    phase_sequence: tuple[str, ...] = PHASE_SEQUENCE
    """Ordered phase names this log enforces; tasks may supply their own."""
    records: list[PhaseRecord] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    _next_phase_index: int = field(default=0, init=False, repr=False)

    def _sample(self) -> PhaseSample:
        step, time_s, position, quaternion, object_position = self.sampler()
        return PhaseSample(
            step=int(step),
            time_s=float(time_s),
            grasp_point_world_m=np.asarray(position, dtype=float).round(6).tolist(),
            grasp_point_quaternion_xyzw=np.asarray(quaternion, dtype=float).round(6).tolist(),
            object_position_m=np.asarray(object_position, dtype=float).round(6).tolist(),
        )

    def begin(self, name: str) -> None:
        """Start ``name``; the previous phase (if open) ends at the same sample."""
        if self._next_phase_index >= len(self.phase_sequence):
            raise ValueError(
                "The phase sequence is already complete; finish the episode before "
                f"starting {name!r}."
            )
        expected = self.phase_sequence[self._next_phase_index]
        if name != expected:
            raise ValueError(
                f"Expected phase {expected!r}, got {name!r}; embodiment policies "
                "must use the shared phase sequence."
            )
        sample = self._sample()
        if self.records and self.records[-1].end is None:
            self.records[-1].end = sample
        self.records.append(PhaseRecord(name=name, start=sample))
        self._next_phase_index += 1

    def end(self) -> None:
        if self.records and self.records[-1].end is None:
            self.records[-1].end = self._sample()

    def finish_episode(self, completed: bool) -> None:
        """Close an episode and reject a successful policy that skipped phases."""
        self.end()
        phase_count = self._next_phase_index
        self._next_phase_index = 0
        if completed and phase_count != len(self.phase_sequence):
            missing = self.phase_sequence[phase_count:]
            raise RuntimeError(
                "Policy reported a completed episode before recording all task phases; "
                f"missing {missing}."
            )

    def event(self, name: str, **payload: Any) -> None:
        """Record a point event (e.g. grasp verified) at the current state."""
        sample = self._sample()
        self.events.append({"name": name, **sample.__dict__, **payload})

    @property
    def current(self) -> str | None:
        return self.records[-1].name if self.records else None

    def to_dict(self) -> dict[str, Any]:
        def sample_dict(sample: PhaseSample | None) -> dict[str, Any] | None:
            return None if sample is None else dict(sample.__dict__)

        return {
            "format": self.FORMAT,
            "embodiment": self.embodiment,
            "phase_sequence": list(self.phase_sequence),
            "phases": [
                {
                    "name": record.name,
                    "start": sample_dict(record.start),
                    "end": sample_dict(record.end),
                }
                for record in self.records
            ],
            "events": list(self.events),
        }

    def save(self, path: Path) -> None:
        self.end()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


__all__ = ["PHASE_SEQUENCE", "PhaseLog", "PhaseRecord", "PhaseSample"]
