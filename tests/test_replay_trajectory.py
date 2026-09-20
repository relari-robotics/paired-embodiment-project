"""Simulator-free tests of the open-loop trajectory file format."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from superdex_scenarios.replay.trajectory import (  # noqa: E402
    FINGER_JOINT_OPEN_MAGNITUDE,
    FORMAT,
    JointTrajectory,
    aperture_from_finger_joints,
    finger_joints_from_aperture,
    phase_sequence_for,
)


def _payload(samples: int = 11, **extra: object) -> dict[str, object]:
    times = np.linspace(0.0, 2.0, samples)
    payload: dict[str, object] = {
        "format": FORMAT,
        "time_s": times.tolist(),
        "right_arm": np.zeros((samples, 7)).tolist(),
        "right_gripper": np.linspace(1.0, 0.0, samples).tolist(),
    }
    payload.update(extra)
    return payload


class ApertureTest(unittest.TestCase):
    def test_aperture_maps_to_mirrored_finger_joints(self) -> None:
        right = finger_joints_from_aperture([0.0, 0.5, 1.0], "right")
        left = finger_joints_from_aperture([0.0, 0.5, 1.0], "left")
        self.assertEqual(right.shape, (3, 2))
        np.testing.assert_allclose(right[:, 0], [0.0, -0.5 * FINGER_JOINT_OPEN_MAGNITUDE, -FINGER_JOINT_OPEN_MAGNITUDE])
        np.testing.assert_allclose(left, -right)
        np.testing.assert_allclose(aperture_from_finger_joints(right, "right"), [0.0, 0.5, 1.0])
        np.testing.assert_allclose(aperture_from_finger_joints(left, "left"), [0.0, 0.5, 1.0])

    def test_aperture_is_clipped(self) -> None:
        np.testing.assert_allclose(finger_joints_from_aperture([-1.0, 2.0], "right")[:, 0], [0.0, -FINGER_JOINT_OPEN_MAGNITUDE])


class JointTrajectoryTest(unittest.TestCase):
    def test_parses_single_arm_file(self) -> None:
        trajectory = JointTrajectory.from_dict(_payload())
        self.assertEqual(trajectory.sides, ("right",))
        self.assertEqual(trajectory.sample_count, 11)
        self.assertAlmostEqual(trajectory.duration_s, 2.0)
        self.assertEqual(trajectory.grippers["right"].shape, (11, 2))
        self.assertEqual(phase_sequence_for(trajectory), ("replay",))

    def test_parses_bimanual_file_with_raw_finger_joints(self) -> None:
        payload = _payload(
            left_arm=np.ones((11, 7)).tolist(),
            left_finger_joints=np.full((11, 2), 0.3).tolist(),
            phases=[{"name": "a", "time_s": 0.0}, {"name": "b", "time_s": 1.0}],
            metadata={"note": "test"},
        )
        trajectory = JointTrajectory.from_dict(payload)
        self.assertEqual(trajectory.sides, ("right", "left"))
        np.testing.assert_allclose(trajectory.grippers["left"], 0.3)
        self.assertEqual(phase_sequence_for(trajectory), ("a", "b"))
        self.assertEqual(trajectory.metadata["note"], "test")

    def test_round_trips_through_json(self) -> None:
        trajectory = JointTrajectory.from_dict(_payload(phases=[{"name": "go", "time_s": 0.5}]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.json"
            trajectory.save(path)
            loaded = JointTrajectory.load(path)
        np.testing.assert_allclose(loaded.time_s, trajectory.time_s)
        np.testing.assert_allclose(loaded.grippers["right"], trajectory.grippers["right"])
        self.assertEqual(loaded.phase_names, ("go",))
        self.assertEqual(json.loads(json.dumps(loaded.to_dict()))["format"], FORMAT)

    def test_rebased_and_resampled(self) -> None:
        payload = _payload()
        payload["time_s"] = (np.asarray(payload["time_s"]) + 5.0).tolist()
        payload["phases"] = [{"name": "p", "time_s": 6.0}]
        trajectory = JointTrajectory.from_dict(payload).rebased()
        self.assertAlmostEqual(float(trajectory.time_s[0]), 0.0)
        self.assertAlmostEqual(trajectory.phases[0].time_s, 1.0)
        resampled = trajectory.resampled(100.0)
        self.assertEqual(resampled.sample_count, 201)
        self.assertAlmostEqual(float(resampled.time_s[-1]), 2.0)
        # Linear interpolation of a linear aperture ramp is exact.
        np.testing.assert_allclose(resampled.grippers["right"][100, 0], -0.5 * FINGER_JOINT_OPEN_MAGNITUDE, atol=1e-9)

    def test_rejects_malformed_files(self) -> None:
        cases = {
            "format": {**_payload(), "format": "other"},
            "time order": _payload(time_s=np.linspace(2.0, 0.0, 11).tolist()),
            "arm shape": _payload(right_arm=np.zeros((11, 6)).tolist()),
            "aperture range": _payload(right_gripper=np.full(11, 1.5).tolist()),
            "both gripper forms": _payload(right_finger_joints=np.zeros((11, 2)).tolist()),
            "gripper without arm": _payload(left_gripper=np.zeros(11).tolist()),
            "phase outside range": _payload(phases=[{"name": "late", "time_s": 9.0}]),
            "duplicate phase": _payload(phases=[{"name": "a", "time_s": 0.0}, {"name": "a", "time_s": 1.0}]),
            "nan": _payload(right_arm=np.full((11, 7), np.nan).tolist()),
        }
        for name, payload in cases.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                JointTrajectory.from_dict(payload)

        payload = _payload()
        del payload["right_arm"]
        with self.assertRaises(ValueError):
            JointTrajectory.from_dict(payload)


if __name__ == "__main__":
    unittest.main()
