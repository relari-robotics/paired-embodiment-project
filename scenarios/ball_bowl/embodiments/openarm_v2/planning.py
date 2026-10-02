"""OpenArm v2 reference planner: one collision-aware route for the right gripper.

The route refers to the grasp point, which coincides with the ball centre while
the ball is held.  It ends above specification.delivery_xy -- the bowl's
target when the bowl is moved first, otherwise where the bowl starts.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from superdex_scenarios.embodiments.base import ArmKinematics, EmbodimentModel
from superdex_scenarios.embodiments.openarm_v2 import freeze_other_arms
from superdex_scenarios.planning import TrajOptTrajectoryOptimizer

from ...scenario import ScenarioSpecification


@dataclass
class PlannedMotion:
    """OpenArm reference trajectory produced by the included robot planner."""

    home_to_pre_pick: npt.NDArray[np.float64]
    pre_pick_to_pick: npt.NDArray[np.float64]
    pick_to_lift: npt.NDArray[np.float64]
    lift_to_place_safe: npt.NDArray[np.float64]
    place_safe_to_place: npt.NDArray[np.float64]
    place_to_retreat: npt.NDArray[np.float64]
    retreat_to_home: npt.NDArray[np.float64]

    def segments(self) -> Iterable[npt.NDArray[np.float64]]:
        return (
            self.home_to_pre_pick,
            self.pre_pick_to_pick,
            self.pick_to_lift,
            self.lift_to_place_safe,
            self.place_safe_to_place,
            self.place_to_retreat,
            self.retreat_to_home,
        )


def _cartesian_ik_reference(
    kinematics: ArmKinematics,
    start_pose: npt.ArrayLike,
    start_position: npt.ArrayLike,
    goal_position: npt.ArrayLike,
    num_knots: int,
) -> npt.NDArray[np.float64]:
    """Follow a Cartesian line with continuation IK to stay on one joint branch."""
    start_pose = np.asarray(start_pose, dtype=float)
    positions = np.linspace(start_position, goal_position, num_knots, dtype=float)
    path = [start_pose]
    seed = start_pose
    for position in positions[1:]:
        seed = kinematics.solve(position, seed)
        path.append(seed)
    return np.asarray(path, dtype=float)


def plan_motion(
    kinematics: ArmKinematics,
    reference: EmbodimentModel,
    specification: ScenarioSpecification | None = None,
    *,
    optimize_trajectory: bool = True,
) -> PlannedMotion:
    """Build and optimize the included OpenArm reference trajectory."""
    specification = specification or ScenarioSpecification.fixed()
    home = reference.default_pose[reference.right_arm_dofs]
    if (
        reference.approach_direction_world is None
        or reference.pregrasp_distance is None
    ):
        raise ValueError("The OpenArm reference requires pregrasp metadata.")
    approach = np.asarray(reference.approach_direction_world, dtype=float)
    pre_pick_position = specification.pick - reference.pregrasp_distance * approach
    pre_pick = kinematics.solve(pre_pick_position, home)
    home_to_pre_pick = np.linspace(home, pre_pick, 12, dtype=float)
    pre_pick_to_pick = _cartesian_ik_reference(
        kinematics,
        pre_pick,
        pre_pick_position,
        specification.pick,
        8,
    )
    pick_to_lift = _cartesian_ik_reference(
        kinematics,
        pre_pick_to_pick[-1],
        specification.pick,
        specification.pick_lift,
        8,
    )
    lift_to_place_safe = _cartesian_ik_reference(
        kinematics,
        pick_to_lift[-1],
        specification.pick_lift,
        specification.place_safe,
        16,
    )
    place_safe_to_place = _cartesian_ik_reference(
        kinematics,
        lift_to_place_safe[-1],
        specification.place_safe,
        specification.place,
        7,
    )
    place_to_retreat = place_safe_to_place[::-1].copy()
    retreat_to_home = np.linspace(place_to_retreat[-1], home, 12, dtype=float)
    paths = (
        home_to_pre_pick,
        pre_pick_to_pick,
        pick_to_lift,
        lift_to_place_safe,
        place_safe_to_place,
        place_to_retreat,
        retreat_to_home,
    )

    if not optimize_trajectory:
        print("Trajectory optimization disabled: using direct Cartesian references.")
        return PlannedMotion(*(freeze_other_arms(path, reference) for path in paths))

    print("TrajOpt-style collision-aware trajectory optimization:")
    optimizer = TrajOptTrajectoryOptimizer(kinematics)
    return PlannedMotion(
        *(
            freeze_other_arms(optimizer.optimize(path, max_iterations=28), reference)
            for path in paths
        )
    )


__all__ = ["PlannedMotion", "plan_motion"]
