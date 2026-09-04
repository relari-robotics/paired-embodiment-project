"""Blank human-hand policy: the project deliverable.

The scenario supplies the physical 27-DOF hand on its six-DOF carrier, the
randomized workcell, the task specification, and a neutral kinematic twin.
The runner supplies control, physics stepping, recording, the success check
and export -- the same machinery that drives the OpenArm reference.

Everything else is the project: the grasp (which digits touch the ball where,
and with what hand shape), the wrist trajectory, how the fingers close, how the
ball is held while carried, and how it is released.  Implement
:class:`HumanPolicy` below; :mod:`scenarios.ball_bowl.openarm_policy` is the
worked reference for the same contract and ``PROJECT.md`` states the goals.

Any approach is acceptable -- optimization, retargeting, motion capture,
teleoperation, learning -- provided the ball moves only through simulated
contact and the hand poses stay comparable to the gripper's at every phase.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from .episode import PolicyOptions

if TYPE_CHECKING:
    from .runner import EpisodeRunner
    from .scenario import BallBowlScenario

UNIMPLEMENTED = (
    "The human hand trajectory is intentionally unimplemented. "
    "Implement HumanPolicy in scenarios/ball_bowl/human_project.py."
)


class HumanPolicy:
    """Episode policy for the physical Meta XR right hand.

    Useful scenario handles:

    * ``scenario.bot_info`` -- the live hand (``arm_dofs`` = carrier x, y, z,
      yaw, pitch, roll; ``hand_dofs`` = 27 finger joints; ``hand_limits``;
      ``contact_groups`` per digit; ``hand_poses["home"]``; ``target_pose``).
    * ``scenario.kinematics`` -- a kinematic twin in a private scene for
      solving poses without disturbing physics (``solve``, ``grasp_point_world``,
      ``collision_cost``, ``link_actors``).
    * ``scenario.specification`` -- ``pick``, ``pre_pick``, ``pick_lift``,
      ``place_safe``, ``place``, ``ball_mass_kg``, ``contains_ball``.
    * ``scenario.workcell.ball`` -- the dynamic ball actor.
    """

    def __init__(self, scenario: BallBowlScenario, options: PolicyOptions) -> None:
        self.scenario = scenario
        self.options = options

    def plan(self) -> None:
        """Solve the grasp and plan the wrist and finger motion.

        Raise ``PlanningError`` when a randomized task is infeasible for the
        hand so the runner resamples it.
        """
        raise NotImplementedError(UNIMPLEMENTED)

    def trajectory_points(self) -> npt.NDArray[np.float64]:
        """(N, 3) world route of the hand's grasp point, for validation and display."""
        raise NotImplementedError(UNIMPLEMENTED)

    def home_pose(self) -> npt.NDArray[np.float64]:
        """Full articulation pose the hand parks in at the start and end."""
        raise NotImplementedError(UNIMPLEMENTED)

    def preshape_pose(self) -> npt.NDArray[np.float64]:
        """Full articulation pose at the pre-grasp standoff (used by ``--snapshot``)."""
        raise NotImplementedError(UNIMPLEMENTED)

    def run(self, runner: EpisodeRunner) -> bool:
        """Execute the phases through ``runner`` (see ``OpenArmPolicy.run``)."""
        raise NotImplementedError(UNIMPLEMENTED)


__all__ = ["HumanPolicy", "UNIMPLEMENTED"]
