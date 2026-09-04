"""Global spline trajectory optimization for any articulated embodiment."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.interpolate import CubicSpline

from ..embodiments.base import ArmKinematics
from .sqp_ik import (
    IKConstraint,
    IKLinearization,
    IKTask,
    SequentialQuadraticIK,
    SequentialQuadraticIKSettings,
)


def sample_natural_cubic_spline(
    path: npt.ArrayLike, samples_per_interval: int = 3
) -> npt.NDArray[np.float64]:
    """Sample the same natural cubic basis used by planning and execution."""
    path = np.asarray(path, dtype=float)
    count = (len(path) - 1) * samples_per_interval + 1
    spline = CubicSpline(
        np.linspace(0.0, 1.0, len(path)), path, axis=0, bc_type="natural"
    )
    return np.asarray(spline(np.linspace(0.0, 1.0, count)), dtype=float)


class TrajOptTrajectoryOptimizer:
    """Sequentially convex spline TrajOpt with explicit clearance constraints."""

    TRAVEL_WEIGHT = 1.0
    ACCELERATION_WEIGHT = 0.004
    REFERENCE_WEIGHT = 0.35
    COLLISION_EPSILON = 0.002
    COLLISION_MARGIN = 0.0005
    COLLISION_ACTIVATION_DISTANCE = 0.035
    SAMPLES_PER_INTERVAL = 3

    def __init__(
        self,
        kinematics: ArmKinematics,
        configuration_constraint: object | None = None,
        *,
        accept_valid_reference: bool | None = None,
    ) -> None:
        self.kinematics = kinematics
        self.configuration_constraint = configuration_constraint
        self.accept_valid_reference = (
            getattr(
                kinematics,
                "accept_collision_free_cartesian_reference",
                False,
            )
            if accept_valid_reference is None
            else accept_valid_reference
        )
        self.limits = getattr(
            kinematics.info, "arm_limits", kinematics.info.right_arm_limits
        )

    @staticmethod
    def spline_basis(
        num_knots: int, samples_per_interval: int
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        knots = np.linspace(0.0, 1.0, num_knots)
        sample_count = (num_knots - 1) * samples_per_interval + 1
        sample_times = np.linspace(0.0, 1.0, sample_count)
        cardinal = CubicSpline(knots, np.eye(num_knots), axis=0, bc_type="natural")
        return (
            np.asarray(cardinal(sample_times), dtype=float),
            np.asarray(cardinal(sample_times, 1), dtype=float),
            np.asarray(cardinal(sample_times, 2), dtype=float),
        )

    def _clearance_and_gradient(
        self, arm_pose: npt.NDArray[np.float64]
    ) -> tuple[float, npt.NDArray[np.float64]]:
        analytical = getattr(
            self.kinematics, "collision_clearance_and_gradient", None
        )
        if analytical is not None:
            return analytical(arm_pose)

        _, clearance = self.kinematics.collision_cost(arm_pose)
        gradient = np.zeros_like(arm_pose)
        for dof in range(len(arm_pose)):
            plus = arm_pose.copy()
            minus = arm_pose.copy()
            plus[dof] += self.COLLISION_EPSILON
            minus[dof] -= self.COLLISION_EPSILON
            _, plus_clearance = self.kinematics.collision_cost(plus)
            _, minus_clearance = self.kinematics.collision_cost(minus)
            gradient[dof] = (plus_clearance - minus_clearance) / (
                2.0 * self.COLLISION_EPSILON
            )
        return clearance, gradient

    def optimize(
        self, reference_path: npt.ArrayLike, max_iterations: int = 28
    ) -> npt.NDArray[np.float64]:
        reference = np.asarray(reference_path, dtype=float)
        num_knots, num_dofs = reference.shape
        if num_knots < 3:
            return reference.copy()

        basis, velocity_basis, acceleration_basis = self.spline_basis(
            num_knots, self.SAMPLES_PER_INTERVAL
        )
        sample_count = len(basis)
        interior_count = num_knots - 2
        sample_map = np.kron(basis[:, 1:-1], np.eye(num_dofs))
        velocity_map = np.kron(velocity_basis[:, 1:-1], np.eye(num_dofs))
        acceleration_map = np.kron(
            acceleration_basis[:, 1:-1], np.eye(num_dofs)
        )

        def unpack(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            path = reference.copy()
            path[1:-1] = values.reshape(num_knots - 2, num_dofs)
            return path

        def linearize(values: npt.NDArray[np.float64]) -> IKLinearization:
            path = unpack(values)
            samples = basis @ path
            velocity = velocity_basis @ path
            acceleration = acceleration_basis @ path
            tasks = (
                IKTask(
                    "joint_travel",
                    -velocity.reshape(-1),
                    velocity_map,
                    weight=self.TRAVEL_WEIGHT / sample_count,
                    required=False,
                ),
                IKTask(
                    "joint_acceleration",
                    -acceleration.reshape(-1),
                    acceleration_map,
                    weight=self.ACCELERATION_WEIGHT / sample_count,
                    required=False,
                ),
                IKTask(
                    "reference_path",
                    reference[1:-1].reshape(-1) - values,
                    np.eye(interior_count * num_dofs),
                    weight=self.REFERENCE_WEIGHT / values.size,
                    required=False,
                ),
            )
            constraints: list[IKConstraint] = [
                IKConstraint(
                    "sampled_joint_limits",
                    samples.reshape(-1),
                    sample_map,
                    lower=np.tile(self.limits[:, 0], sample_count),
                    upper=np.tile(self.limits[:, 1], sample_count),
                )
            ]
            clearance_values: list[float] = []
            clearance_rows: list[npt.NDArray[np.float64]] = []
            for sample_index, sample in enumerate(samples):
                clearance, gradient = self._clearance_and_gradient(sample)
                if clearance > self.COLLISION_ACTIVATION_DISTANCE:
                    continue
                clearance_values.append(clearance)
                clearance_rows.append(
                    np.kron(basis[sample_index, 1:-1], gradient)
                )
            if clearance_values:
                constraints.append(
                    IKConstraint(
                        "collision_clearance",
                        np.asarray(clearance_values, dtype=float),
                        np.vstack(clearance_rows),
                        lower=self.COLLISION_MARGIN,
                    )
                )
            if self.configuration_constraint is not None:
                values: list[npt.NDArray[np.float64]] = []
                rows: list[npt.NDArray[np.float64]] = []
                lowers: list[npt.NDArray[np.float64]] = []
                uppers: list[npt.NDArray[np.float64]] = []
                for sample_index, sample in enumerate(samples):
                    value, gradient, lower_bound, upper_bound = (
                        self.configuration_constraint.linearize(sample)
                    )
                    values.append(value)
                    rows.append(
                        np.kron(
                            basis[sample_index, 1:-1][None, :], gradient
                        )
                    )
                    lowers.append(lower_bound)
                    uppers.append(upper_bound)
                constraints.append(
                    IKConstraint(
                        "configuration_feasibility",
                        np.concatenate(values),
                        np.vstack(rows),
                        lower=np.concatenate(lowers),
                        upper=np.concatenate(uppers),
                    )
                )
            return IKLinearization(tasks, tuple(constraints))

        lower = np.tile(self.limits[:, 0] + 1.0e-3, num_knots - 2)
        upper = np.tile(self.limits[:, 1] - 1.0e-3, num_knots - 2)
        initial_clearance = self.minimum_clearance(reference, basis)
        initial_travel = self.joint_travel(reference, basis)
        reference_configuration_valid = True
        if self.configuration_constraint is not None:
            try:
                self.configuration_constraint.validate(
                    basis @ reference, "TrajOpt reference"
                )
            except RuntimeError:
                reference_configuration_valid = False
        if (
            self.accept_valid_reference
            and initial_clearance >= self.COLLISION_MARGIN
            and reference_configuration_valid
        ):
            print(
                f"  Cartesian TrajOpt {num_knots:2d} knots: validated reference, "
                f"clearance {initial_clearance:+.3f} m, travel "
                f"{initial_travel:.3f} m/rad"
            )
            return reference.copy()
        solver = SequentialQuadraticIK(
            SequentialQuadraticIKSettings(
                max_iterations=max_iterations,
                initial_trust_region=0.16,
                maximum_trust_region=0.30,
                damping=1.0e-6,
                constraint_tolerance=1.0e-4,
                merit_constraint_weight=1.0e7,
                stop_when_tasks_satisfied=False,
            )
        )
        result = solver.solve(
            reference[1:-1].reshape(-1),
            lower,
            upper,
            linearize,
        )
        path = unpack(result.configuration)
        final_clearance = self.minimum_clearance(path, basis)
        final_travel = self.joint_travel(path, basis)
        if not np.all(np.isfinite(path)) or final_clearance < -1.0e-4:
            raise RuntimeError(
                "TrajOpt failed to produce a finite collision-free spline: "
                f"clearance={final_clearance:+.4f} m, backend={result.backend}, "
                f"status={result.message}"
            )
        if self.configuration_constraint is not None:
            self.configuration_constraint.validate(
                basis @ path, "Optimized trajectory"
            )
        print(
            f"  SQP TrajOpt {num_knots:2d} knots ({result.backend}): clearance "
            f"{initial_clearance:+.3f} -> {final_clearance:+.3f} m, "
            f"travel {initial_travel:.3f} -> {final_travel:.3f} rad"
        )
        return path

    def minimum_clearance(
        self,
        path: npt.ArrayLike,
        basis: npt.NDArray[np.float64] | None = None,
    ) -> float:
        path = np.asarray(path, dtype=float)
        if basis is None:
            basis = self.spline_basis(len(path), self.SAMPLES_PER_INTERVAL)[0]
        samples = basis @ path
        specialized = getattr(self.kinematics, "minimum_path_clearance", None)
        if specialized is not None:
            return float(specialized(samples))
        return min(self.kinematics.collision_cost(sample)[1] for sample in samples)

    @staticmethod
    def joint_travel(path: npt.ArrayLike, basis: npt.NDArray[np.float64]) -> float:
        samples = basis @ np.asarray(path, dtype=float)
        return float(np.sum(np.linalg.norm(np.diff(samples, axis=0), axis=1)))
