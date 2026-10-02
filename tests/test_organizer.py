"""Layout rejection, mesh validity, and reproducible physical scene generation."""

import unittest
import numpy as np

from scenarios.organizer.geometry import (
    ScenarioSpecification,
    DIVIDER_BOXES,
    union_box_mesh,
)


class OrganizerGeometryTests(unittest.TestCase):
    def test_union_has_outward_faces_closed_edges_and_correct_volume(self):
        vertices, faces = union_box_mesh(DIVIDER_BOXES)
        edges = {}
        for face in faces:
            for a, b in zip(face, np.roll(face, -1)):
                key = tuple(sorted((int(a), int(b))))
                edges.setdefault(key, []).append((int(a), int(b)))
        self.assertTrue(
            all(len(pair) == 2 and pair[0] == pair[1][::-1] for pair in edges.values())
        )
        a, b, c = (vertices[faces[:, i]] for i in range(3))
        volume = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6
        expected = 0.140 * 0.020 * 0.035 + 0.050 * 0.042 * 0.050 - 0.050 * 0.020 * 0.010
        self.assertAlmostEqual(volume, expected, places=10)

    def test_invalid_layouts_are_rejected(self):
        for data in (
            {"organizer_xy": [0, float("nan")]},
            {"part_xy": [[0, 0]]},
            {"organzier_xy": [0, 0]},
        ):
            with self.assertRaises(ValueError):
                ScenarioSpecification.from_layout(data)
        base = ScenarioSpecification.fixed()
        for data in (
            {"organizer_xy": [1, 0]},
            {"part_xy": [base.organizer_xy, *base.part_xy[1:]]},
        ):
            spec = ScenarioSpecification.from_layout(data)
            with self.assertRaises(ValueError):
                spec.validate(
                    np.array([-0.36, -0.5969, 0.338]), np.array([0.6096, 1.1938, 0.05])
                )

    def test_random_layouts_are_reproducible_and_fit_the_table(self):
        for seed in range(100):
            spec = ScenarioSpecification.randomized(seed)
            self.assertEqual(spec, ScenarioSpecification.randomized(seed))
            spec.validate(
                np.array([-0.36, -0.5969, 0.338]), np.array([0.6096, 1.1938, 0.05])
            )

    def test_layout_positions_are_preserved_without_snapping(self):
        data = {
            "organizer_xy": [-0.005, 0.025],
            "divider_xy": [[0.01, -0.24], [-0.01, 0.22]],
            "part_xy": [[-0.03, -0.39], [-0.16, -0.14], [-0.05, 0.37]],
        }
        spec = ScenarioSpecification.from_layout(data)
        for key, value in data.items():
            np.testing.assert_allclose(getattr(spec, key), value)


if __name__ == "__main__":
    unittest.main()
