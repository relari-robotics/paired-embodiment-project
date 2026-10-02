"""The open-loop OpenArm joint-trajectory file format.

A retargeting pipeline produces one of these files per demonstration; the
scenario runners replay it (``runner.py --replay FILE``) through the same
compliant controller, physics, recording, and success checks that drive the
scripted reference policies.  The format is deliberately plain JSON so it can
be written from any language and inspected by hand.  See ``TRAJECTORY.md`` at
the repository root for the human-readable specification.

This module imports no simulator code so it can be unit tested anywhere.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

FORMAT = "openarm-joint-trajectory-v1"
SIDES = ("right", "left")
ARM_DOF_COUNT = 7
FINGER_DOF_COUNT = 2
FINGER_JOINT_OPEN_MAGNITUDE = 0.78


def finger_sign(side: str) -> float:
    """Sign of an *open* finger joint value for ``side`` (mirrors the hardware)."""
    return -1.0 if side == "right" else 1.0


def finger_joints_from_aperture(aperture: npt.ArrayLike, side: str) -> npt.NDArray[np.float64]:
    """Map a normalized aperture (0 closed .. 1 fully open) to both finger joints."""
    aperture = np.clip(np.asarray(aperture, dtype=float).reshape(-1), 0.0, 1.0)
    value = finger_sign(side) * FINGER_JOINT_OPEN_MAGNITUDE * aperture
    return np.column_stack([value, value])


def aperture_from_finger_joints(joints: npt.ArrayLike, side: str) -> npt.NDArray[np.float64]:
    joints = np.asarray(joints, dtype=float).reshape(-1, FINGER_DOF_COUNT)
    mean = joints.mean(axis=1)
    return np.clip(mean / (finger_sign(side) * FINGER_JOINT_OPEN_MAGNITUDE), 0.0, 1.0)


@dataclass(frozen=True)
class PhaseMark:
    """A named task phase beginning at ``time_s`` on the trajectory clock."""

    name: str
    time_s: float


@dataclass
class JointTrajectory:
    """Timed joint targets for one or both OpenArm arms and grippers.

    ``time_s`` is strictly increasing and starts anywhere (the runner rebases
    it).  ``arms`` maps ``"right"``/``"left"`` to (N, 7) joint arrays in radians
    (joint1 .. joint7).  ``grippers`` maps a side to an (N, 2) array of finger
    joint values in radians; :func:`finger_joints_from_aperture` converts the
    friendlier normalized aperture.  ``phases`` are optional semantic marks.
    """

    time_s: npt.NDArray[np.float64]
    arms: dict[str, npt.NDArray[np.float64]]
    grippers: dict[str, npt.NDArray[np.float64]]
    phases: list[PhaseMark] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = ""

    # -- construction ------------------------------------------------------

    @classmethod
    def from_dict(cls, payload: dict[str, Any], source: str = "") -> JointTrajectory:
        fmt = payload.get("format")
        if fmt != FORMAT:
            raise ValueError(f"Unsupported trajectory format {fmt!r}; expected {FORMAT!r}.")
        if "time_s" not in payload:
            raise ValueError("Trajectory is missing 'time_s'.")
        time_s = np.asarray(payload["time_s"], dtype=float).reshape(-1)
        n = len(time_s)
        if n < 2:
            raise ValueError("Trajectory needs at least two samples.")
        if not np.all(np.isfinite(time_s)):
            raise ValueError("Trajectory 'time_s' contains non-finite values.")
        if np.any(np.diff(time_s) <= 0.0):
            raise ValueError("Trajectory 'time_s' must be strictly increasing.")

        arms: dict[str, npt.NDArray[np.float64]] = {}
        grippers: dict[str, npt.NDArray[np.float64]] = {}
        for side in SIDES:
            arm_key = f"{side}_arm"
            if arm_key in payload:
                arm = np.asarray(payload[arm_key], dtype=float)
                if arm.shape != (n, ARM_DOF_COUNT):
                    raise ValueError(
                        f"'{arm_key}' must have shape ({n}, {ARM_DOF_COUNT}), got {arm.shape}."
                    )
                if not np.all(np.isfinite(arm)):
                    raise ValueError(f"'{arm_key}' contains non-finite values.")
                arms[side] = arm
            aperture_key = f"{side}_gripper"
            joints_key = f"{side}_finger_joints"
            if aperture_key in payload and joints_key in payload:
                raise ValueError(f"Give either '{aperture_key}' or '{joints_key}', not both.")
            if aperture_key in payload:
                aperture = np.asarray(payload[aperture_key], dtype=float).reshape(-1)
                if len(aperture) != n:
                    raise ValueError(f"'{aperture_key}' must have {n} samples.")
                if not np.all(np.isfinite(aperture)):
                    raise ValueError(f"'{aperture_key}' contains non-finite values.")
                if np.any(aperture < -1e-9) or np.any(aperture > 1.0 + 1e-9):
                    raise ValueError(f"'{aperture_key}' values must lie in [0, 1].")
                grippers[side] = finger_joints_from_aperture(aperture, side)
            elif joints_key in payload:
                joints = np.asarray(payload[joints_key], dtype=float)
                if joints.shape != (n, FINGER_DOF_COUNT):
                    raise ValueError(
                        f"'{joints_key}' must have shape ({n}, {FINGER_DOF_COUNT}), got {joints.shape}."
                    )
                if not np.all(np.isfinite(joints)):
                    raise ValueError(f"'{joints_key}' contains non-finite values.")
                grippers[side] = joints
            if side in grippers and side not in arms:
                raise ValueError(f"'{side}' gripper given without '{side}_arm'.")
        if not arms:
            raise ValueError("Trajectory has no 'right_arm' or 'left_arm' samples.")

        phases: list[PhaseMark] = []
        for entry in payload.get("phases", []) or []:
            if not isinstance(entry, dict) or "name" not in entry or "time_s" not in entry:
                raise ValueError("Each phase needs 'name' and 'time_s'.")
            phases.append(PhaseMark(str(entry["name"]), float(entry["time_s"])))
        if phases:
            marks = np.asarray([p.time_s for p in phases], dtype=float)
            if np.any(np.diff(marks) < 0.0):
                raise ValueError("Phase times must be non-decreasing.")
            if marks[0] < time_s[0] - 1e-9 or marks[-1] > time_s[-1] + 1e-9:
                raise ValueError("Phase times must lie within the trajectory time range.")
            names = [p.name for p in phases]
            if len(set(names)) != len(names):
                raise ValueError("Phase names must be unique.")
        metadata = payload.get("metadata", {}) or {}
        if not isinstance(metadata, dict):
            raise ValueError("'metadata' must be an object.")
        return cls(time_s, arms, grippers, phases, dict(metadata), source)

    @classmethod
    def load(cls, path: str | Path) -> JointTrajectory:
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(payload, source=str(path))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": FORMAT,
            "time_s": np.round(self.time_s, 6).tolist(),
        }
        for side, arm in self.arms.items():
            payload[f"{side}_arm"] = np.round(arm, 6).tolist()
        for side, joints in self.grippers.items():
            payload[f"{side}_finger_joints"] = np.round(joints, 6).tolist()
        if self.phases:
            payload["phases"] = [
                {"name": p.name, "time_s": round(p.time_s, 6)} for p in self.phases
            ]
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), separators=(",", ":")) + "\n", encoding="utf-8")

    # -- queries -----------------------------------------------------------

    @property
    def sides(self) -> tuple[str, ...]:
        return tuple(side for side in SIDES if side in self.arms)

    @property
    def sample_count(self) -> int:
        return int(len(self.time_s))

    @property
    def duration_s(self) -> float:
        return float(self.time_s[-1] - self.time_s[0])

    @property
    def phase_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.phases)

    def rebased(self) -> JointTrajectory:
        """Return a copy whose clock starts at zero."""
        offset = float(self.time_s[0])
        return JointTrajectory(
            self.time_s - offset,
            {k: v.copy() for k, v in self.arms.items()},
            {k: v.copy() for k, v in self.grippers.items()},
            [PhaseMark(p.name, p.time_s - offset) for p in self.phases],
            dict(self.metadata),
            self.source,
        )

    def resampled(self, rate_hz: float) -> JointTrajectory:
        """Linearly resample to a uniform rate (the runner does this per physics step)."""
        if rate_hz <= 0.0:
            raise ValueError("rate_hz must be positive.")
        count = max(2, int(np.floor(self.duration_s * rate_hz)) + 1)
        times = self.time_s[0] + np.arange(count) / rate_hz
        times = np.minimum(times, self.time_s[-1])

        def interp(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return np.column_stack(
                [np.interp(times, self.time_s, values[:, d]) for d in range(values.shape[1])]
            )

        return JointTrajectory(
            times,
            {k: interp(v) for k, v in self.arms.items()},
            {k: interp(v) for k, v in self.grippers.items()},
            list(self.phases),
            dict(self.metadata),
            self.source,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "samples": self.sample_count,
            "duration_s": round(self.duration_s, 4),
            "sides": list(self.sides),
            "grippers": list(self.grippers),
            "phases": [p.name for p in self.phases],
        }


def phase_sequence_for(trajectory: JointTrajectory, fallback: str = "replay") -> tuple[str, ...]:
    """Phase names a replay logs: the file's marks, or one generic phase."""
    names = trajectory.phase_names
    return names if names else (fallback,)


__all__ = [
    "ARM_DOF_COUNT",
    "FINGER_DOF_COUNT",
    "FINGER_JOINT_OPEN_MAGNITUDE",
    "FORMAT",
    "SIDES",
    "JointTrajectory",
    "PhaseMark",
    "aperture_from_finger_joints",
    "finger_joints_from_aperture",
    "finger_sign",
    "phase_sequence_for",
]
