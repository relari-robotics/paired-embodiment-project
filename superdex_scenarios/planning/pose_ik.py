"""Bounded nonlinear task-space IK built on the shared SQP optimizer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from .sqp_ik import (
    IKConstraint,
    IKLinearization,
    IKTask,
    SequentialQuadraticIK,
    SequentialQuadraticIKResult,
    SequentialQuadraticIKSettings,
)

Vector = npt.NDArray[np.float64]
ForwardPose = Callable[[Vector], tuple[Vector, Vector]]
Clearance = Callable[[Vector], float]


def _unit_quaternion(value: npt.ArrayLike) -> Vector:
    quaternion = np.asarray(value, dtype=float).reshape(4)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(quaternion).all() or norm < 1.0e-9:
        raise ValueError("target quaternion must be finite and nonzero")
    quaternion = quaternion / norm
    return -quaternion if quaternion[3] < 0.0 else quaternion


def _rotation_error(target: Vector, current: Vector) -> Vector:
    return np.asarray(
        (Rotation.from_quat(target) * Rotation.from_quat(current).inv()).as_rotvec(),
        dtype=float,
    )


@dataclass(frozen=True)
class PoseIKSettings:
    max_iterations: int = 14
    finite_difference_step: float = 2.0e-4
    position_weight: float = 400.0
    orientation_weight: float = 4.0
    posture_weight: float = 0.02
    position_tolerance_m: float = 0.012
    orientation_tolerance_rad: float = 0.20
    collision_activation_m: float = 0.035
    collision_margin_m: float = 0.0005


class PoseIKOptimizer:
    """Optimize one pose with hard joint bounds and optional clearance.

    Forward kinematics remains the source of truth. Numerical derivatives only
    linearize the nonlinear pose and clearance functions for each bounded SQP
    step; no Jacobian inverse or pseudoinverse is formed.
    """

    def __init__(
        self,
        forward_pose: ForwardPose,
        clearance: Clearance | None = None,
        *,
        settings: PoseIKSettings | None = None,
        backend: str = "auto",
    ) -> None:
        self.forward_pose = forward_pose
        self.clearance = clearance
        self.settings = settings or PoseIKSettings()
        self.solver = SequentialQuadraticIK(
            SequentialQuadraticIKSettings(
                max_iterations=self.settings.max_iterations,
                initial_trust_region=0.16,
                minimum_trust_region=2.0e-4,
                maximum_trust_region=0.30,
                damping=2.0e-4,
                constraint_tolerance=1.0e-4,
                merit_constraint_weight=1.0e7,
                line_search_steps=5,
            ),
            backend=backend,
        )

    def _step_points(
        self,
        configuration: Vector,
        lower: Vector,
        upper: Vector,
        index: int,
    ) -> tuple[Vector, Vector, float]:
        step = self.settings.finite_difference_step
        plus = configuration.copy()
        minus = configuration.copy()
        plus[index] = min(plus[index] + step, upper[index])
        minus[index] = max(minus[index] - step, lower[index])
        denominator = float(plus[index] - minus[index])
        if denominator <= 0.0:
            raise ValueError("IK joint bounds contain no finite-difference interval")
        return plus, minus, denominator

    def solve(
        self,
        position_m: npt.ArrayLike,
        quaternion_xyzw: npt.ArrayLike,
        seed: npt.ArrayLike,
        lower_limits: npt.ArrayLike,
        upper_limits: npt.ArrayLike,
    ) -> SequentialQuadraticIKResult:
        target_position = np.asarray(position_m, dtype=float).reshape(3)
        target_quaternion = _unit_quaternion(quaternion_xyzw)
        seed = np.asarray(seed, dtype=float).reshape(-1)
        lower = np.asarray(lower_limits, dtype=float).reshape(-1)
        upper = np.asarray(upper_limits, dtype=float).reshape(-1)
        if not (seed.shape == lower.shape == upper.shape):
            raise ValueError("seed and IK joint limits must have matching shapes")
        if not np.isfinite(target_position).all() or not np.all(lower < upper):
            raise ValueError("IK target and joint bounds must be finite and ordered")

        def linearize(configuration: Vector) -> IKLinearization:
            position, quaternion = self.forward_pose(configuration)
            position = np.asarray(position, dtype=float).reshape(3)
            quaternion = _unit_quaternion(quaternion)
            position_jacobian = np.empty((3, len(configuration)), dtype=float)
            rotation_jacobian = np.empty((3, len(configuration)), dtype=float)
            for index in range(len(configuration)):
                plus, minus, denominator = self._step_points(
                    configuration, lower, upper, index
                )
                plus_position, plus_quaternion = self.forward_pose(plus)
                minus_position, minus_quaternion = self.forward_pose(minus)
                position_jacobian[:, index] = (
                    np.asarray(plus_position) - np.asarray(minus_position)
                ) / denominator
                rotation_jacobian[:, index] = (
                    _rotation_error(
                        _unit_quaternion(plus_quaternion),
                        _unit_quaternion(minus_quaternion),
                    )
                    / denominator
                )

            tasks = [
                IKTask(
                    "position",
                    target_position - position,
                    position_jacobian,
                    weight=self.settings.position_weight,
                    tolerance=self.settings.position_tolerance_m,
                ),
                IKTask(
                    "orientation",
                    _rotation_error(target_quaternion, quaternion),
                    rotation_jacobian,
                    weight=self.settings.orientation_weight,
                    tolerance=self.settings.orientation_tolerance_rad,
                ),
                IKTask(
                    "posture",
                    seed - configuration,
                    np.eye(len(configuration)),
                    weight=self.settings.posture_weight,
                    required=False,
                ),
            ]
            constraints = []
            if self.clearance is not None:
                clearance = float(self.clearance(configuration))
                if clearance <= self.settings.collision_activation_m:
                    gradient = np.empty((1, len(configuration)), dtype=float)
                    for index in range(len(configuration)):
                        plus, minus, denominator = self._step_points(
                            configuration, lower, upper, index
                        )
                        gradient[0, index] = (
                            self.clearance(plus) - self.clearance(minus)
                        ) / denominator
                    constraints.append(
                        IKConstraint(
                            "collision_clearance",
                            np.asarray([clearance]),
                            gradient,
                            lower=self.settings.collision_margin_m,
                        )
                    )
            return IKLinearization.from_sequences(tasks, constraints)

        return self.solver.solve(seed, lower, upper, linearize)


__all__ = ["PoseIKOptimizer", "PoseIKSettings"]
