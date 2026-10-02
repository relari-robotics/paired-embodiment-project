"""Geometry checks for the authored arm used by both renderers."""

import unittest

import numpy as np

from superdex_scenarios.rendering.arm_surface import cross_section, segment_mesh


class ArmSurfaceTests(unittest.TestCase):
    def test_segments_are_closed_outward_surfaces_without_degenerate_faces(self):
        for part, length in (("upper_arm", .302), ("forearm", .246)):
            with self.subTest(part=part):
                vertices, faces = segment_mesh(part, length)
                triangles = vertices[faces]
                normals = np.cross(triangles[:, 1] - triangles[:, 0],
                                   triangles[:, 2] - triangles[:, 0])
                self.assertTrue(np.all(np.linalg.norm(normals, axis=1) > 1e-10))
                self.assertGreater(np.einsum("ij,ij->", triangles[:, 0], normals), 0)
                edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
                _, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
                self.assertTrue(np.all(counts == 2), "Every edge must have two faces")

    def test_wrist_is_flattened_and_narrower_than_forearm(self):
        angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        wrist = np.ptp(cross_section("forearm", 1, angles), axis=0)
        belly = np.ptp(cross_section("forearm", .3, angles), axis=0)
        self.assertTrue(np.all(wrist < belly * .75))
        self.assertGreater(wrist[0], wrist[1] * 1.2)

    def test_elbow_sections_meet(self):
        angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        error = cross_section("upper_arm", 1, angles) - cross_section("forearm", 0, angles)
        self.assertLess(np.max(np.linalg.norm(error, axis=1)), .0011)


if __name__ == "__main__":
    unittest.main()
