"""Success must follow actual object geometry and the tray's actual transform."""

import unittest
import numpy as np

try:
    from superdex import physics, robotics
    from scenarios.organizer.scenario import OrganizerScenario
except ImportError:
    physics = None


@unittest.skipIf(physics is None, "SuperDex is not installed")
class OrganizerOutcomeTests(unittest.TestCase):
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
        for kind, actors in (
            ("divider", self.scenario.dividers),
            ("part", self.scenario.parts),
        ):
            for index, actor in enumerate(actors):
                actor.set_root_transform(
                    physics.TransformRT(translation=self.scenario.target(kind, index))
                )

    def test_initial_state_fails_and_correct_geometry_passes(self):
        self.assertFalse(self.scenario.outcome()["success"])
        self.place_all()
        self.assertTrue(self.scenario.outcome()["success"])

    def test_shifted_tray_changes_targets_and_success(self):
        self.place_all()
        previous = self.scenario.target("part", 0)
        transform = self.scenario.organizer.get_root_transform()
        delta = np.array([0.02, -0.015, 0])
        self.scenario.organizer.set_root_transform(
            physics.TransformRT(translation=np.asarray(transform.translation) + delta)
        )
        np.testing.assert_allclose(
            self.scenario.target("part", 0), previous + delta, atol=1e-7
        )
        self.assertFalse(self.scenario.outcome()["success"])
        self.place_all()
        self.assertTrue(self.scenario.outcome()["success"])

    def test_unseated_divider_and_part_straddling_wall_fail(self):
        self.place_all()
        divider = self.scenario.dividers[0]
        divider.set_root_transform(
            physics.TransformRT(
                translation=self.scenario.target("divider", 0) + [0, 0, 0.01]
            )
        )
        self.assertFalse(self.scenario.outcome()["dividers"][0]["success"])
        self.place_all()
        part = self.scenario.parts[0]
        part.set_root_transform(
            physics.TransformRT(
                translation=self.scenario.target("part", 0) + [0, 0.04, 0]
            )
        )
        self.assertFalse(self.scenario.outcome()["parts"][0]["success"])


if __name__ == "__main__":
    unittest.main()
