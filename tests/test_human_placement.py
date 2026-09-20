"""Visual arm placement must stay attached while the body remains stationary."""
import unittest
import numpy as np
from superdex_scenarios.rendering.human_placement import placement, arm_joints, segment_matrix, UPPER_LENGTH, FORE_LENGTH

class HumanPlacementTests(unittest.TestCase):
    def test_body_anchor_stays_fixed_while_arm_reaches_targets(self):
        wrist=np.array([.263,-.418,.651]);origin,shoulder=placement(wrist)
        initial=shoulder.copy()
        for offset in ([0,0,0],[.04,.03,.04],[-.02,.02,.07]):
            target=wrist+offset;elbow=arm_joints(shoulder,target)
            np.testing.assert_allclose(shoulder,initial)
            self.assertAlmostEqual(np.linalg.norm(elbow-shoulder),UPPER_LENGTH,places=6)
            self.assertAlmostEqual(np.linalg.norm(target-elbow),FORE_LENGTH,places=6)
        self.assertAlmostEqual(origin[2],0.)

    def test_far_targets_remain_attached_and_finite(self):
        shoulder=np.array([.4,-.4,1.2]);wrist=np.array([-.8,.2,.4])
        elbow=arm_joints(shoulder,wrist)
        self.assertTrue(np.isfinite(elbow).all())
        matrix=segment_matrix(elbow,wrist,[0,1,0])
        endpoint=matrix[:3,3]+matrix[:3,0]*np.linalg.norm(wrist-elbow)
        np.testing.assert_allclose(endpoint,wrist,atol=1e-8)
        np.testing.assert_allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3),atol=1e-8)

if __name__=='__main__':unittest.main()
