import numpy as np
from scipy.spatial.transform import Rotation

from superdex_scenarios.planning.pose_ik import PoseIKOptimizer


def test_pose_ik_uses_bounded_optimization_without_a_pseudoinverse():
    def forward(configuration):
        position = configuration[:3]
        quaternion = Rotation.from_rotvec([0.0, 0.0, configuration[3]]).as_quat()
        return position, quaternion

    optimizer = PoseIKOptimizer(forward, backend="scipy")
    result = optimizer.solve(
        [0.2, -0.3, 0.4],
        Rotation.from_euler("z", 0.5).as_quat(),
        np.zeros(4),
        np.full(4, -1.0),
        np.full(4, 1.0),
    )

    assert result.backend == "scipy-slsqp"
    assert result.converged
    np.testing.assert_allclose(result.configuration, [0.2, -0.3, 0.4, 0.5], atol=3e-3)


def test_pose_ik_respects_joint_bounds():
    optimizer = PoseIKOptimizer(
        lambda configuration: (
            np.array([configuration[0], 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 1.0]),
        ),
        backend="scipy",
    )
    result = optimizer.solve(
        [2.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0],
        [-0.5],
        [0.5],
    )

    assert result.configuration[0] <= 0.5
    assert result.configuration[0] >= -0.5
