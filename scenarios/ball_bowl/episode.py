"""The contract every embodiment implements to perform the ball-into-bowl task.

The runner (``runner.py``) owns everything that is the same for every
embodiment: building the scene, the compliant pose controller, physics
stepping, the viewer, transform/telemetry/phase recording, the success check,
and export.  Everything that differs between embodiments -- how the grasp is
solved, how the trajectory is planned, how the fingers or jaws are closed and
how the object is carried -- lives in an :class:`EpisodePolicy`.

An embodiment registers its policy class in ``scenario.EMBODIMENTS``; the
runner instantiates it with the built scenario and the CLI options, asks it to
plan, and then hands it an :class:`~scenarios.ball_bowl.runner.EpisodeRunner`
whose helpers (``follow``, ``hold``, ``phase``, ``check_grasp_alignment``,
``verify_physical_grasp``, ``executor``) it uses to execute the phases.

This module deliberately imports no simulator code so the contract can be unit
tested without SuperDex.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from superdex_scenarios.recording.phases import PHASE_SEQUENCE

if TYPE_CHECKING:
    from .runner import EpisodeRunner
    from .scenario import BallBowlScenario

# Semantic task phases, in order.  Both embodiments mark the same phases so
# paired episodes can be aligned phase by phase in the dataset.
TASK_PHASES = PHASE_SEQUENCE


class PlanningError(RuntimeError):
    """The sampled task cannot be planned for this embodiment.

    Raised from :meth:`EpisodePolicy.plan`.  ``BallBowlScenario.build_randomized``
    treats it as "resample the task", exactly as the OpenArm reference does
    when its IK or trajectory optimization fails.
    """


@dataclass(frozen=True)
class PolicyOptions:
    """Runner options a policy may honour."""

    optimize_trajectory: bool = True
    """Run collision-aware trajectory optimization (``--no-trajopt`` clears it)."""
    allow_failed_grasp: bool = False
    """Continue a diagnostic episode after a failed grasp instead of raising."""


@runtime_checkable
class EpisodePolicy(Protocol):
    """One embodiment's way of doing the task, from planning to execution."""

    scenario: BallBowlScenario
    options: PolicyOptions

    def plan(self) -> None:
        """Solve everything needed before physics starts.

        Grasp synthesis, inverse kinematics, trajectory optimization, or
        loading pre-authored motion all belong here.  Raise
        :class:`PlanningError` if the sampled task is infeasible for this
        embodiment so a randomized run can resample.
        """

    def trajectory_points(self) -> npt.NDArray[np.float64]:
        """(N, 3) world positions of the grasp point along the planned route.

        Used for ``--plan-only`` validation and the viewer's trajectory curve.
        """

    def home_pose(self) -> npt.NDArray[np.float64]:
        """Full articulation pose (all DOFs) the embodiment parks in."""

    def preshape_pose(self) -> npt.NDArray[np.float64]:
        """Full articulation pose at the pre-grasp standoff (``--snapshot``)."""

    def run(self, runner: EpisodeRunner) -> bool:
        """Execute the episode through ``runner``; return False if interrupted.

        Mark each phase of :data:`TASK_PHASES` with ``runner.phase(name)`` as
        it begins.  Move the object only through simulated contact.
        """


def load_policy_class(reference: str) -> type:
    """Resolve ``"package.module:ClassName"`` to the class object."""
    module_name, _, class_name = reference.partition(":")
    if not module_name or not class_name:
        raise ValueError(f"Policy reference must look like 'module:Class', got {reference!r}.")
    return getattr(importlib.import_module(module_name), class_name)


__all__ = [
    "TASK_PHASES",
    "EpisodePolicy",
    "PlanningError",
    "PolicyOptions",
    "load_policy_class",
]
