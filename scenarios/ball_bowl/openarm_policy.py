"""OpenArm v2 reference policy: collision-aware planning and a two-jaw grasp.

This is the worked example of :class:`~scenarios.ball_bowl.episode.EpisodePolicy`.
It plans one Cartesian route for the gripper point (the ball centre while
held) with the scenario's IK and TrajOpt, then executes the phases with a
purely position-controlled jaw closure -- adequate for parallel jaws, and a
behavioural reference rather than a recipe for other embodiments.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from superdex_scenarios.planning import sample_natural_cubic_spline

from . import scenario as task
from .episode import PlanningError, PolicyOptions
from .runner import EpisodeRunner


class OpenArmPolicy:
    """Reference implementation for the OpenArm v2 right arm and gripper."""

    def __init__(self, scenario: task.BallBowlScenario, options: PolicyOptions) -> None:
        self.scenario = scenario
        self.options = options
        self.info = scenario.bot_info
        self.motion: task.PlannedMotion | None = None

    # -- planning ---------------------------------------------------------

    def plan(self) -> None:
        try:
            self.motion = task.plan_motion(
                self.scenario.kinematics,
                self.info,
                self.scenario.specification,
                optimize_trajectory=self.options.optimize_trajectory,
            )
        except RuntimeError as error:
            raise PlanningError(str(error)) from error

    def _motion(self) -> task.PlannedMotion:
        if self.motion is None:
            raise RuntimeError("OpenArmPolicy.plan() must run before execution.")
        return self.motion

    def trajectory_points(self) -> npt.NDArray[np.float64]:
        """Sample the same cubic splines that the simulator executes."""
        kinematics = self.scenario.kinematics
        motion = self._motion()
        points = []
        for segment in motion.segments():
            samples = sample_natural_cubic_spline(segment)
            points.extend(kinematics.grasp_point_world(q) for q in samples[:-1])
        points.append(kinematics.grasp_point_world(motion.retreat_to_home[-1]))
        return np.asarray(points, dtype=float)

    def home_pose(self) -> npt.NDArray[np.float64]:
        return self.info.target_pose(self._motion().home_to_pre_pick[0], "home")

    def preshape_pose(self) -> npt.NDArray[np.float64]:
        return self.info.target_pose(self._motion().pre_pick_to_pick[-1], "preshape")

    # -- execution --------------------------------------------------------

    def run(self, runner: EpisodeRunner) -> bool:
        motion = self._motion()
        home = motion.home_to_pre_pick[0]
        pregrasp = motion.pre_pick_to_pick[-1]
        pick = pregrasp
        place = motion.place_safe_to_place[-1]

        print(f"Executing {self.info.display_name} pick-and-drop episode...")
        runner.phase("home")
        if not runner.hold(home, "home", 0.20):
            return False
        runner.phase("preshape")
        if not runner.follow(np.vstack([home, home]), 0.55, "home", "preshape"):
            return False
        runner.phase("approach")
        if not runner.follow(motion.home_to_pre_pick, 1.5, "preshape"):
            return False
        runner.phase("pre_grasp")
        if not runner.follow(motion.pre_pick_to_pick, 1.2, "preshape"):
            return False
        if not runner.hold(pregrasp, "preshape", 0.65):
            return False
        runner.print_tracking_error(pregrasp)
        runner.check_grasp_alignment()

        # Parallel jaws: a position ramp closes them; the ball is held by the
        # resulting contact forces alone.
        runner.phase("grasp")
        if not runner.follow(np.vstack([pick, pick]), 0.90, "open", "closed"):
            return False
        if not runner.hold(pick, "closed", 0.35):
            return False
        grasp = self.info.hand_poses["closed"]
        runner.phase("lift")
        if not runner.follow(motion.pick_to_lift, 1.2, grasp):
            return False
        runner.verify_physical_grasp()
        runner.phase("carry")
        if not runner.follow(motion.lift_to_place_safe, 2.0, grasp):
            return False
        runner.phase("lower")
        if not runner.follow(motion.place_safe_to_place, 0.7, grasp):
            return False

        runner.phase("release")
        runner.print_release_position()
        if not runner.follow(np.vstack([place, place]), 0.55, grasp, "open"):
            return False
        if not runner.hold(place, "open", 1.15):
            return False
        runner.phase("retreat")
        if not runner.follow(motion.place_to_retreat, 0.7, "open"):
            return False
        runner.phase("return_home")
        if not runner.follow(motion.retreat_to_home, 1.5, "open", "home"):
            return False
        return runner.hold(home, "home", 0.35)


__all__ = ["OpenArmPolicy"]
