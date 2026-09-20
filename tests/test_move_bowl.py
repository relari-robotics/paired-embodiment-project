"""Tests for the movable bowl and the move-the-bowl variant of ball-and-bowl."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scenarios.ball_bowl.episode import (  # noqa: E402
    MOVE_BOWL_PHASES,
    TASK_PHASES,
)
from superdex_scenarios.recording.phases import PhaseLog  # noqa: E402

HAS_SUPERDEX = importlib.util.find_spec("superdex.physics") is not None


class MoveBowlPhasesTest(unittest.TestCase):
    def test_phase_sequence_extends_the_ball_and_bowl_phases(self) -> None:
        self.assertEqual(MOVE_BOWL_PHASES[:2], ("home", "preshape"))
        bowl_phases = tuple(p for p in MOVE_BOWL_PHASES if p.startswith("bowl_"))
        self.assertEqual(
            bowl_phases,
            ("bowl_approach", "bowl_grasp", "bowl_move", "bowl_release", "bowl_retreat"),
        )
        self.assertEqual(MOVE_BOWL_PHASES[2 + len(bowl_phases) :], TASK_PHASES[2:])

    def test_phase_log_accepts_the_sequence(self) -> None:
        def sampler():
            return 0, 0.0, [0, 0, 0], [0, 0, 0, 1], [0, 0, 0]

        log = PhaseLog("test", sampler, MOVE_BOWL_PHASES)
        for phase in MOVE_BOWL_PHASES:
            log.begin(phase)
        log.finish_episode(completed=True)
        self.assertEqual(tuple(r.name for r in log.records), MOVE_BOWL_PHASES)


@unittest.skipUnless(HAS_SUPERDEX, "scenario.py imports SuperDex")
class MovableBowlSpecificationTest(unittest.TestCase):
    def setUp(self) -> None:
        from scenarios.ball_bowl import scenario

        self.task = scenario

    def test_bowl_is_movable_without_a_target_by_default(self) -> None:
        spec = self.task.ScenarioSpecification.fixed()
        self.assertIsNone(spec.bowl_target_xy)
        self.assertGreater(spec.bowl_mass_kg, 0.0)
        self.assertEqual(spec.delivery_xy, spec.bowl_xy)
        self.assertTrue(spec.to_dict()["bowl"]["dynamic"])
        self.assertNotIn("bowl_target_m", spec.to_dict()["targets"])

    def test_bowl_is_static_only_when_asked(self) -> None:
        spec_type = self.task.ScenarioSpecification
        self.assertFalse(spec_type.fixed().bowl_static)
        self.assertFalse(spec_type.fixed(move_bowl=True).bowl_static)
        self.assertFalse(self.task.sample_specification(np.random.default_rng(7), 7, 1).bowl_static)
        self.assertFalse(spec_type.from_layout({}).bowl_static)

        bolted = spec_type.fixed(bowl_static=True)
        self.assertTrue(bolted.bowl_static)
        self.assertFalse(bolted.to_dict()["bowl"]["dynamic"])
        self.assertIsNone(bolted.to_dict()["bowl"]["mass_kg"])
        # The original regression scene: the bowl sits exactly on the desk top.
        self.assertEqual(bolted.bowl_start[2], self.task.DESK_TOP_Z)
        self.assertGreater(spec_type.fixed().bowl_start[2], self.task.DESK_TOP_Z)
        self.assertTrue(spec_type.from_layout({"bowl_static": True}).bowl_static)

    def test_a_bowl_that_has_to_move_cannot_be_static(self) -> None:
        with self.assertRaises(ValueError):
            self.task.ScenarioSpecification.fixed(move_bowl=True, bowl_static=True)
        with self.assertRaises(ValueError):
            self.task.ScenarioSpecification.from_layout(
                {"bowl_target_xy": [0.0, 0.02], "bowl_static": True}
            )

    def test_move_bowl_delivers_the_ball_to_the_target(self) -> None:
        spec = self.task.ScenarioSpecification.fixed(move_bowl=True)
        self.assertEqual(spec.delivery_xy, spec.bowl_target_xy)
        self.assertNotEqual(spec.bowl_xy, spec.bowl_target_xy)
        np.testing.assert_allclose(spec.place[:2], spec.bowl_target_xy)
        self.assertIn("bowl_target_m", spec.to_dict()["targets"])

    def test_ball_is_judged_against_where_the_bowl_actually_sits(self) -> None:
        spec = self.task.ScenarioSpecification.fixed()
        desk = self.task.DESK_TOP_Z
        ball = [*spec.bowl_xy, desk + 0.04]
        self.assertTrue(spec.contains_ball(ball))
        self.assertTrue(spec.contains_ball(ball, [*spec.bowl_xy, desk]))
        pushed_away = [spec.bowl_xy[0] + 0.30, spec.bowl_xy[1], desk]
        self.assertFalse(spec.contains_ball(ball, pushed_away))

    def test_bowl_at_target_needs_position_height_and_upright(self) -> None:
        spec = self.task.ScenarioSpecification.fixed(move_bowl=True)
        desk = self.task.DESK_TOP_Z
        at_target = [*spec.bowl_target_xy, desk]
        upright = [0.0, 0.0, 0.0, 1.0]
        tipped = [np.sin(np.radians(20.0)), 0.0, 0.0, np.cos(np.radians(20.0))]
        self.assertTrue(spec.bowl_at_target(at_target, upright))
        self.assertFalse(spec.bowl_at_target(at_target, tipped))
        self.assertFalse(spec.bowl_at_target([*spec.bowl_xy, desk], upright))
        self.assertFalse(spec.bowl_at_target([*spec.bowl_target_xy, desk + 0.05], upright))
        with self.assertRaises(ValueError):
            self.task.ScenarioSpecification.fixed().bowl_at_target(at_target)

    def test_layout_with_a_target_selects_the_variant(self) -> None:
        from_layout = self.task.ScenarioSpecification.from_layout
        self.assertIsNone(from_layout({"ball_xy": [0.0, -0.2]}).bowl_target_xy)
        spec = from_layout({"bowl_target_xy": [-0.02, 0.01], "bowl_mass_kg": 0.3})
        self.assertEqual(spec.bowl_target_xy, (-0.02, 0.01))
        self.assertEqual(spec.bowl_mass_kg, 0.3)
        fixed = self.task.ScenarioSpecification.fixed(move_bowl=True)
        self.assertEqual(spec.bowl_xy, fixed.bowl_xy)

    def test_a_seed_keeps_its_layout_from_before_the_bowl_was_movable(self) -> None:
        # The bowl mass is drawn last, so positions and colours are unchanged.
        spec = self.task.sample_specification(np.random.default_rng(1234), 1234, 1)
        rng = np.random.default_rng(1234)
        rng.integers(len(self.task.BOWL_SHAPES))
        ball = (rng.uniform(*self.task.BALL_X_RANGE), rng.uniform(*self.task.BALL_Y_RANGE))
        np.testing.assert_allclose(spec.ball_xy, ball)
        self.assertIsNone(spec.bowl_target_xy)

    def test_sampled_move_bowl_layout_is_separated(self) -> None:
        for seed in range(20):
            spec = self.task.sample_specification(
                np.random.default_rng(seed), seed, 1, move_bowl=True
            )
            self.assertEqual(spec.bowl_shape.name, "measured")
            start, target = np.asarray(spec.bowl_xy), np.asarray(spec.bowl_target_xy)
            self.assertGreaterEqual(np.linalg.norm(start - target), 0.12)

    def test_only_two_armed_embodiments_move_the_bowl(self) -> None:
        from scenarios.ball_bowl.embodiments import EMBODIMENTS

        moving = self.task.ScenarioSpecification.fixed(move_bowl=True)
        still = self.task.ScenarioSpecification.fixed()
        bimanual = EMBODIMENTS["openarm_v2_bimanual"]
        self.assertNotEqual(bimanual.policy_for(moving), bimanual.policy_for(still))
        for embodiment_id in ("openarm_v2", "human_right_hand"):
            with self.assertRaises(ValueError):
                EMBODIMENTS[embodiment_id].policy_for(moving)


if __name__ == "__main__":
    unittest.main()
