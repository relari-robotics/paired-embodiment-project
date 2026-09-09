# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Simulator-free tests for the sponge-and-plate scenario's pure-NumPy parts."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scenarios.sponge_plate.cleanliness import (  # noqa: E402
    CLEAN_MIN_NORMAL_FORCE,
    CLEAN_MIN_SLIDING_SPEED,
    DIRT_CELL,
    DIRT_RADIUS,
    CleanlinessMap,
)
from scenarios.sponge_plate.episode import TASK_PHASES, EpisodePolicy, PolicyOptions  # noqa: E402
from scenarios.sponge_plate.plate import (  # noqa: E402
    is_closed_surface,
    solidify_plate,
    top_heightfield,
)
from superdex_scenarios.recording.phases import PHASE_SEQUENCE, PhaseLog  # noqa: E402

PLATE_STL = REPOSITORY_ROOT / "scenarios" / "sponge_plate" / "assets" / "ycb_029_plate" / "nontextured.stl"


class _Specification:
    plate_center_xy = (-0.05, 0.0)


class CleanlinessMapTest(unittest.TestCase):
    def test_cells_cover_the_dish_floor_disc(self) -> None:
        cleanliness = CleanlinessMap(_Specification())
        radii = np.hypot(cleanliness.cells[:, 0], cleanliness.cells[:, 1])
        self.assertTrue(np.all(radii <= DIRT_RADIUS))
        expected = np.pi * DIRT_RADIUS**2 / DIRT_CELL**2
        self.assertAlmostEqual(len(cleanliness.cells) / expected, 1.0, delta=0.08)
        self.assertEqual(cleanliness.coverage, 0.0)

    def test_only_pressed_sliding_samples_clean_cells(self) -> None:
        cleanliness = CleanlinessMap(_Specification())
        centre = np.array(_Specification.plate_center_xy)
        world = np.array([[centre[0] + 0.005, centre[1] + 0.005, 0.39]] * 3)
        forces = np.array([CLEAN_MIN_NORMAL_FORCE, 0.0, CLEAN_MIN_NORMAL_FORCE])
        speeds = np.array([0.0, 1.0, CLEAN_MIN_SLIDING_SPEED])
        cleanliness.update(1, world[:2], forces[:2], speeds[:2])
        self.assertEqual(cleanliness.coverage, 0.0)
        cleanliness.update(2, world[2:], forces[2:], speeds[2:])
        self.assertEqual(int(cleanliness.cleaned.sum()), 1)
        self.assertEqual(int(cleanliness.first_cleaned_step[cleanliness.cleaned][0]), 2)

    def test_samples_outside_the_map_are_ignored(self) -> None:
        cleanliness = CleanlinessMap(_Specification())
        far = np.array([[1.0, 1.0, 0.39], [-0.05 + DIRT_RADIUS + 0.05, 0.0, 0.39]])
        cleanliness.update(1, far, np.ones(2), np.ones(2))
        self.assertEqual(cleanliness.coverage, 0.0)
        payload = cleanliness.to_dict()
        self.assertEqual(payload["coverage"], 0.0)
        self.assertEqual(len(payload["cleaned"]), len(cleanliness.cells))

    def test_a_sweep_across_the_disc_cleans_most_of_it(self) -> None:
        cleanliness = CleanlinessMap(_Specification())
        centre = np.array(_Specification.plate_center_xy)
        xs, ys = np.meshgrid(
            np.linspace(-DIRT_RADIUS, DIRT_RADIUS, 60),
            np.linspace(-DIRT_RADIUS, DIRT_RADIUS, 60),
        )
        world = np.column_stack([xs.ravel() + centre[0], ys.ravel() + centre[1], np.full(xs.size, 0.39)])
        cleanliness.update(5, world, np.ones(len(world)), np.ones(len(world)))
        self.assertGreater(cleanliness.coverage, 0.98)


class SolidPlateTest(unittest.TestCase):
    def test_heightfield_of_a_unit_box(self) -> None:
        box = np.array(
            [[[0, 0, 0.02], [0.1, 0, 0.02], [0.1, 0.1, 0.02]], [[0, 0, 0.02], [0.1, 0.1, 0.02], [0, 0.1, 0.02]]]
        )
        xs, ys, heights = top_heightfield(box, 0.01)
        self.assertTrue(np.all(np.isfinite(heights)))
        self.assertTrue(np.allclose(heights, 0.02))
        self.assertEqual(len(xs), 11)
        self.assertEqual(len(ys), 11)

    @unittest.skipUnless(PLATE_STL.exists(), "plate scan not checked out")
    def test_solidified_scan_is_closed_and_keeps_the_dish_profile(self) -> None:
        plate = solidify_plate(PLATE_STL)
        self.assertTrue(is_closed_surface(plate.faces))
        self.assertGreater(plate.outer_radius, 0.12)
        self.assertLess(plate.outer_radius, 0.14)
        centre = plate.center_xy
        floor = plate.surface_height(float(centre[0]), float(centre[1]))
        rim = plate.surface_height(float(centre[0]) + 0.12, float(centre[1]))
        self.assertLess(floor, 0.006)
        self.assertGreater(rim - floor, 0.015)
        self.assertTrue(np.isnan(plate.surface_height(1.0, 1.0)))
        self.assertLess(plate.base_z, 0.0)


class SpongePlateContractTest(unittest.TestCase):
    def test_phase_sequence_is_distinct_from_ball_and_bowl_but_shares_its_frame(self) -> None:
        self.assertEqual(TASK_PHASES[0], "home")
        self.assertEqual(TASK_PHASES[-1], "return_home")
        self.assertIn("wipe", TASK_PHASES)
        self.assertNotEqual(tuple(TASK_PHASES), tuple(PHASE_SEQUENCE))
        self.assertEqual(len(set(TASK_PHASES)), len(TASK_PHASES))

    def test_phase_log_enforces_the_task_phase_sequence(self) -> None:
        def sampler():
            return 0, 0.0, [0, 0, 0], [0, 0, 0, 1], [0, 0, 0]

        log = PhaseLog("test", sampler, TASK_PHASES)
        with self.assertRaisesRegex(ValueError, "Expected phase 'home'"):
            log.begin("wipe")
        for phase in TASK_PHASES:
            log.begin(phase)
        log.finish_episode(completed=True)
        self.assertEqual(log.to_dict()["phase_sequence"], list(TASK_PHASES))

    def test_policy_options_are_shared_with_ball_and_bowl(self) -> None:
        from scenarios.ball_bowl.episode import EpisodePolicy as BallBowlPolicy
        from scenarios.ball_bowl.episode import PolicyOptions as BallBowlOptions

        self.assertIs(EpisodePolicy, BallBowlPolicy)
        self.assertIs(PolicyOptions, BallBowlOptions)


if __name__ == "__main__":
    unittest.main()
