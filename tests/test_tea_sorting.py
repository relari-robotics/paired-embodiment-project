"""Geometry and measured-state task checks, separate from full policy rollouts."""

import unittest
import numpy as np
from scenarios.tea_sorting.geometry import (
    ScenarioSpecification,
    packet_surface,
    PACKET_SIZE,
)


class TeaGeometryTests(unittest.TestCase):
    def test_surface_is_closed_outward_and_thin(self):
        vertices, quads, _ = packet_surface()
        faces = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
        edges = {}
        for face in faces:
            for a, b in zip(face, np.roll(face, -1)):
                edges.setdefault(tuple(sorted((a, b))), []).append((a, b))
        self.assertTrue(
            all(len(pair) == 2 and pair[0] == pair[1][::-1] for pair in edges.values())
        )
        a, b, c = (vertices[faces[:, i]] for i in range(3))
        volume = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6
        self.assertGreater(volume, 0)
        self.assertLess(volume, np.prod(PACKET_SIZE))
        self.assertLessEqual(np.ptp(vertices[:, 1]), 0.010)
        self.assertAlmostEqual(np.ptp(vertices[:, 0]), 0.062)
        self.assertAlmostEqual(np.ptp(vertices[:, 2]), 0.075)
        # Compressed heat seal is 0.7 mm, not a hidden ten-millimetre box.
        seal = vertices[np.isclose(vertices[:, 2], PACKET_SIZE[2] / 2)]
        self.assertAlmostEqual(np.ptp(seal[:, 1]), 0.0007)

    def test_detailed_render_stays_within_contact_error_budget(self):
        physical, _, _ = packet_surface(48, 56)
        for tea in range(3):
            visual, _, _ = packet_surface(48, 56, tea, detail=True)
            self.assertLess(np.max(np.linalg.norm(visual - physical, axis=1)), 0.00023)

    def test_seeded_layouts_are_valid_and_reproducible(self):
        for seed in range(100):
            spec = ScenarioSpecification.randomized(seed)
            self.assertEqual(spec, ScenarioSpecification.randomized(seed))
            spec.validate([-0.36, -0.5969, 0.338], [0.6096, 1.1938, 0.05])

    def test_bad_layouts_fail(self):
        for data in (
            {"organizer_xy": [float("nan"), 0]},
            {"part_xy": [[0, 0]]},
            {"tea_order": [0, 0, 1]},
            {"unknown": 1},
        ):
            with self.assertRaises(ValueError):
                ScenarioSpecification.from_layout(data)
        with self.assertRaises(ValueError):
            ScenarioSpecification(organizer_xy=(1, 0)).validate(
                [-0.36, -0.5969, 0.338], [0.6096, 1.1938, 0.05]
            )


try:
    from superdex import physics, robotics
    from scenarios.tea_sorting.scenario import OrganizerScenario
except ImportError:
    physics = None


@unittest.skipIf(physics is None, "SuperDex not installed")
class TeaOutcomeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        physics.initialize(num_worker_threads=0)
        cls.context = robotics.create_context()

    @classmethod
    def tearDownClass(cls):
        physics.shutdown()

    def setUp(self):
        self.scenario = OrganizerScenario.build(self.context)

    def tearDown(self):
        self.scenario.close()

    def place_all(self):
        for i, actor in enumerate(self.scenario.parts):
            actor.set_root_transform(
                physics.TransformRT(
                    translation=self.scenario.target("part", i)
                    + [0, 0, PACKET_SIZE[2] / 2]
                )
            )

    def test_initial_fails_and_actual_containment_passes(self):
        self.assertFalse(self.scenario.outcome()["success"])
        self.place_all()
        self.assertTrue(self.scenario.outcome()["success"])
        actor = self.scenario.parts[0]
        actor.set_root_transform(
            physics.TransformRT(
                translation=self.scenario.target("part", 1) + [0, 0, PACKET_SIZE[2] / 2]
            )
        )
        self.assertFalse(self.scenario.outcome()["parts"][0]["success"])

    def test_targets_follow_box_and_not_world_constants(self):
        old = self.scenario.target("part", 0)
        delta = np.array([0.01, -0.007, 0])
        pose = self.scenario.organizer.get_root_transform()
        self.scenario.organizer.set_root_transform(
            physics.TransformRT(translation=np.asarray(pose.translation) + delta)
        )
        np.testing.assert_allclose(
            self.scenario.target("part", 0), old + delta, atol=1e-7
        )
        self.place_all()
        self.assertTrue(self.scenario.outcome()["success"])


if __name__ == "__main__":
    unittest.main()
