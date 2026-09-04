"""Sequential quadratic inverse kinematics over simulator-native Jacobians.

The optimizer deliberately knows nothing about a particular robot or physics
engine.  An embodiment linearizes Cartesian tasks and signed-clearance
constraints at the current configuration; this module repeatedly solves the
resulting small convex QP.  Keeping model evaluation outside the optimizer lets
Mochi remain the single kinematic source of truth.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy.optimize import Bounds, LinearConstraint, minimize


Vector = npt.NDArray[np.float64]
Matrix = npt.NDArray[np.float64]


@dataclass(frozen=True)
class IKTask:
    """One locally linear Cartesian or posture objective.

    ``target_delta`` and ``jacobian`` use the convention
    ``jacobian @ configuration_delta == target_delta``.
    """

    name: str
    target_delta: Vector
    jacobian: Matrix
    weight: float | Vector = 1.0
    tolerance: float = 1.0e-3
    required: bool = True

    def __post_init__(self) -> None:
        target = np.asarray(self.target_delta, dtype=float).reshape(-1)
        jacobian = np.asarray(self.jacobian, dtype=float)
        if jacobian.ndim != 2 or jacobian.shape[0] != len(target):
            raise ValueError(
                f"Task {self.name!r} has target shape {target.shape} and "
                f"Jacobian shape {jacobian.shape}."
            )
        weight = np.asarray(self.weight, dtype=float)
        if weight.ndim > 1 or (weight.ndim == 1 and len(weight) != len(target)):
            raise ValueError(f"Task {self.name!r} has incompatible weights.")
        if np.any(weight <= 0.0):
            raise ValueError(f"Task {self.name!r} weights must be positive.")
        object.__setattr__(self, "target_delta", target)
        object.__setattr__(self, "jacobian", jacobian)

    @property
    def error_norm(self) -> float:
        return float(np.linalg.norm(self.target_delta))


@dataclass(frozen=True)
class IKConstraint:
    """Bounds on a locally linear quantity ``value + J @ delta``."""

    name: str
    value: Vector
    jacobian: Matrix
    lower: float | Vector = -np.inf
    upper: float | Vector = np.inf

    def __post_init__(self) -> None:
        value = np.asarray(self.value, dtype=float).reshape(-1)
        jacobian = np.asarray(self.jacobian, dtype=float)
        lower = np.broadcast_to(np.asarray(self.lower, dtype=float), value.shape).copy()
        upper = np.broadcast_to(np.asarray(self.upper, dtype=float), value.shape).copy()
        if jacobian.ndim != 2 or jacobian.shape[0] != len(value):
            raise ValueError(
                f"Constraint {self.name!r} has value shape {value.shape} and "
                f"Jacobian shape {jacobian.shape}."
            )
        if np.any(lower > upper):
            raise ValueError(f"Constraint {self.name!r} has lower > upper.")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "jacobian", jacobian)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def maximum_violation(self) -> float:
        below = np.maximum(np.asarray(self.lower) - self.value, 0.0)
        above = np.maximum(self.value - np.asarray(self.upper), 0.0)
        return float(max(np.max(below, initial=0.0), np.max(above, initial=0.0)))


@dataclass(frozen=True)
class IKLinearization:
    tasks: tuple[IKTask, ...]
    constraints: tuple[IKConstraint, ...] = ()

    @classmethod
    def from_sequences(
        cls,
        tasks: Sequence[IKTask],
        constraints: Sequence[IKConstraint] = (),
    ) -> IKLinearization:
        return cls(tuple(tasks), tuple(constraints))


@dataclass(frozen=True)
class SequentialQuadraticIKSettings:
    max_iterations: int = 80
    initial_trust_region: float = 0.24
    minimum_trust_region: float = 2.0e-4
    maximum_trust_region: float = 0.40
    damping: float = 2.0e-5
    constraint_tolerance: float = 5.0e-4
    step_tolerance: float = 2.0e-6
    merit_constraint_weight: float = 2.0e5
    line_search_steps: int = 7
    stop_when_tasks_satisfied: bool = True


@dataclass(frozen=True)
class SequentialQuadraticIKResult:
    configuration: Vector
    converged: bool
    iterations: int
    backend: str
    task_errors: dict[str, float] = field(default_factory=dict)
    maximum_constraint_violation: float = 0.0
    message: str = ""


LinearizeIK = Callable[[Vector], IKLinearization]
IntegrateIK = Callable[[Vector, Vector], Vector]


class SequentialQuadraticIK:
    """Warm-startable local SQP solver for small articulated IK problems."""

    def __init__(
        self,
        settings: SequentialQuadraticIKSettings | None = None,
        backend: str = "auto",
    ) -> None:
        self.settings = settings or SequentialQuadraticIKSettings()
        if backend not in {"auto", "proxsuite", "scipy"}:
            raise ValueError(f"Unknown QP backend {backend!r}.")
        self.backend = backend
        self._proxsuite = None
        if backend != "scipy":
            try:
                import proxsuite  # type: ignore[import-not-found]

                self._proxsuite = proxsuite
            except ImportError:
                if backend == "proxsuite":
                    raise RuntimeError(
                        "The ProxQP backend requires the 'proxsuite' package."
                    ) from None
        self.backend_name = "proxqp" if self._proxsuite is not None else "scipy-slsqp"

    @staticmethod
    def _weighted_task(task: IKTask) -> tuple[Matrix, Vector]:
        weight = np.asarray(task.weight, dtype=float)
        scale = np.sqrt(weight)
        if scale.ndim == 0:
            return task.jacobian * float(scale), task.target_delta * float(scale)
        return task.jacobian * scale[:, None], task.target_delta * scale

    def _qp_matrices(
        self,
        linearization: IKLinearization,
        num_variables: int,
    ) -> tuple[Matrix, Vector, Matrix, Vector, Vector]:
        hessian = self.settings.damping * np.eye(num_variables)
        gradient = np.zeros(num_variables, dtype=float)
        for task in linearization.tasks:
            jacobian, target = self._weighted_task(task)
            hessian += jacobian.T @ jacobian
            gradient -= jacobian.T @ target
        # Numerical symmetry matters to QP solvers after many outer products.
        hessian = 0.5 * (hessian + hessian.T)

        if linearization.constraints:
            matrix = np.vstack(
                [constraint.jacobian for constraint in linearization.constraints]
            )
            lower = np.concatenate(
                [
                    np.asarray(constraint.lower) - constraint.value
                    for constraint in linearization.constraints
                ]
            )
            upper = np.concatenate(
                [
                    np.asarray(constraint.upper) - constraint.value
                    for constraint in linearization.constraints
                ]
            )
        else:
            matrix = np.empty((0, num_variables), dtype=float)
            lower = np.empty(0, dtype=float)
            upper = np.empty(0, dtype=float)
        return hessian, gradient, matrix, lower, upper

    def _solve_proxqp(
        self,
        hessian: Matrix,
        gradient: Vector,
        matrix: Matrix,
        lower: Vector,
        upper: Vector,
        box_lower: Vector,
        box_upper: Vector,
    ) -> Vector:
        assert self._proxsuite is not None
        qp = self._proxsuite.proxqp.dense.QP(
            len(gradient), 0, len(lower), True
        )
        qp.settings.eps_abs = 1.0e-8
        qp.settings.max_iter = 2000
        qp.init(
            hessian,
            gradient,
            None,
            None,
            matrix if len(lower) else None,
            lower if len(lower) else None,
            upper if len(lower) else None,
            l_box=box_lower,
            u_box=box_upper,
        )
        qp.solve()
        status = str(qp.results.info.status)
        if "SOLVED" not in status:
            raise RuntimeError(f"ProxQP failed: {status}.")
        return np.asarray(qp.results.x, dtype=float)

    @staticmethod
    def _solve_scipy_qp(
        hessian: Matrix,
        gradient: Vector,
        matrix: Matrix,
        lower: Vector,
        upper: Vector,
        box_lower: Vector,
        box_upper: Vector,
    ) -> Vector:
        constraints = (
            [LinearConstraint(matrix, lower, upper)] if len(lower) else []
        )

        def objective(delta: Vector) -> tuple[float, Vector]:
            return (
                0.5 * float(delta @ hessian @ delta) + float(gradient @ delta),
                hessian @ delta + gradient,
            )

        result = minimize(
            objective,
            np.clip(np.zeros_like(gradient), box_lower, box_upper),
            method="SLSQP",
            jac=True,
            bounds=Bounds(box_lower, box_upper),
            constraints=constraints,
            options={"maxiter": 180, "ftol": 1.0e-11, "disp": False},
        )
        if not result.success:
            raise RuntimeError(f"SciPy QP failed: {result.message}.")
        return np.asarray(result.x, dtype=float)

    def _solve_qp(
        self,
        linearization: IKLinearization,
        box_lower: Vector,
        box_upper: Vector,
    ) -> Vector:
        matrices = self._qp_matrices(linearization, len(box_lower))
        if self._proxsuite is not None:
            return self._solve_proxqp(*matrices, box_lower, box_upper)
        return self._solve_scipy_qp(*matrices, box_lower, box_upper)

    def _merit(self, linearization: IKLinearization) -> float:
        merit = 0.0
        for task in linearization.tasks:
            weight = np.asarray(task.weight, dtype=float)
            merit += float(np.sum(weight * task.target_delta**2))
        for constraint in linearization.constraints:
            violation = constraint.maximum_violation()
            merit += self.settings.merit_constraint_weight * violation**2
        return merit

    def _convergence(
        self, linearization: IKLinearization
    ) -> tuple[bool, dict[str, float], float]:
        errors = {task.name: task.error_norm for task in linearization.tasks}
        tasks_ok = all(
            not task.required or task.error_norm <= task.tolerance
            for task in linearization.tasks
        )
        maximum_violation = max(
            (
                constraint.maximum_violation()
                for constraint in linearization.constraints
            ),
            default=0.0,
        )
        return (
            tasks_ok and maximum_violation <= self.settings.constraint_tolerance,
            errors,
            maximum_violation,
        )

    def solve(
        self,
        seed: npt.ArrayLike,
        lower_limits: npt.ArrayLike,
        upper_limits: npt.ArrayLike,
        linearize: LinearizeIK,
        integrate: IntegrateIK | None = None,
    ) -> SequentialQuadraticIKResult:
        """Solve from ``seed`` while respecting configuration and task limits."""
        configuration = np.asarray(seed, dtype=float).copy()
        lower_limits = np.asarray(lower_limits, dtype=float)
        upper_limits = np.asarray(upper_limits, dtype=float)
        if not (
            configuration.shape == lower_limits.shape == upper_limits.shape
            and configuration.ndim == 1
        ):
            raise ValueError("Seed and joint limits must be equal-length vectors.")
        configuration = np.clip(configuration, lower_limits, upper_limits)
        if integrate is None:
            integrate = lambda pose, delta: pose + delta
        trust_region = self.settings.initial_trust_region
        last_message = "maximum iterations reached"

        for iteration in range(self.settings.max_iterations + 1):
            current = linearize(configuration)
            converged, errors, violation = self._convergence(current)
            if converged and self.settings.stop_when_tasks_satisfied:
                return SequentialQuadraticIKResult(
                    configuration,
                    True,
                    iteration,
                    self.backend_name,
                    errors,
                    violation,
                    "converged",
                )
            if iteration == self.settings.max_iterations:
                break

            delta_lower = np.maximum(lower_limits - configuration, -trust_region)
            delta_upper = np.minimum(upper_limits - configuration, trust_region)
            try:
                delta = self._solve_qp(current, delta_lower, delta_upper)
            except RuntimeError as error:
                last_message = str(error)
                trust_region *= 0.5
                if trust_region < self.settings.minimum_trust_region:
                    break
                continue

            if np.linalg.norm(delta, ord=np.inf) <= self.settings.step_tolerance:
                last_message = "stationary QP step"
                if (
                    violation <= self.settings.constraint_tolerance
                    and (
                        not self.settings.stop_when_tasks_satisfied
                        or converged
                    )
                ):
                    return SequentialQuadraticIKResult(
                        configuration,
                        True,
                        iteration,
                        self.backend_name,
                        errors,
                        violation,
                        last_message,
                    )
                break

            current_merit = self._merit(current)
            accepted = False
            best_candidate = configuration
            best_merit = current_merit
            alpha = 1.0
            for _ in range(self.settings.line_search_steps):
                candidate = np.clip(
                    integrate(configuration, alpha * delta),
                    lower_limits,
                    upper_limits,
                )
                candidate_merit = self._merit(linearize(candidate))
                if candidate_merit < best_merit:
                    best_candidate = candidate
                    best_merit = candidate_merit
                    accepted = True
                if candidate_merit <= current_merit * (1.0 - 1.0e-4 * alpha):
                    break
                alpha *= 0.5

            if accepted:
                configuration = best_candidate
                if alpha >= 0.5:
                    trust_region = min(
                        self.settings.maximum_trust_region, 1.35 * trust_region
                    )
            else:
                trust_region *= 0.5
                last_message = "line search could not reduce the nonlinear merit"
                if trust_region < self.settings.minimum_trust_region:
                    break

        final = linearize(configuration)
        _, errors, violation = self._convergence(final)
        return SequentialQuadraticIKResult(
            configuration,
            False,
            min(iteration, self.settings.max_iterations),
            self.backend_name,
            errors,
            violation,
            last_message,
        )


__all__ = [
    "IKConstraint",
    "IKLinearization",
    "IKTask",
    "SequentialQuadraticIK",
    "SequentialQuadraticIKResult",
    "SequentialQuadraticIKSettings",
]
