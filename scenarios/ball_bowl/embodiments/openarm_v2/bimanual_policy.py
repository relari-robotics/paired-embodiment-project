"""Bimanual OpenArm v2 reference policy: slide the bowl, then drop the ball in.

Left arm first: with the gripper pitched nose-down (so the wrist stays above
the rim), descend onto the bowl rim on the arm's own side, pinch the rim
between the jaws, drag the bowl across the desk to its target, open, and
return home.  Right arm second: the ball-and-bowl reference motion
(collision-aware planning against the bowl at its *target*) picks the ball up
and releases it above the relocated bowl.  Everything moves through simulated
contact; nothing is attached or teleported.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from superdex_scenarios.embodiments.openarm_v2 import (
    RIGHT_FINGERS_CLOSED,
    RIGHT_FINGERS_HOME,
    RIGHT_FINGERS_OPEN,
    _quaternion_xyzw,
    ee_link,
    finger_sign,
)
from superdex_scenarios.planning import sample_natural_cubic_spline

from ... import scenario as task
from ...episode import MOVE_BOWL_PHASES, PlanningError, PolicyOptions
from ...runner import EpisodeRunner
from .planning import PlannedMotion, plan_motion

RIM_INSET = 0.006
"""Grasp point this far inside the outer rim radius (the wall between the jaws)."""
RIM_GRASP_DEPTH = 0.004
"""Grasp point this far below the rim top (the pitched jaw tips reach 2 cm lower)."""
RIM_PINCH = 0.04
"""Finger joint magnitude commanded while pinching the thin rim (0 = fully closed)."""
BOWL_HOVER = 0.045
"""Pre-grasp/retreat height above the rim grasp point."""
HOME_LIFT = 0.12
"""The left gripper rises this much straight up before travelling over the bowl."""
BOWL_DRAG_LIFT = 0.0
"""Extra height while dragging; zero keeps the bowl resting on the desk."""


def _hand(info, right: float, left: float) -> npt.NDArray[np.float64]:
    """Full gripper vector [right1, right2, left1, left2] from per-side magnitudes."""
    del info
    return np.array(
        [
            finger_sign("right") * right,
            finger_sign("right") * right,
            finger_sign("left") * left,
            finger_sign("left") * left,
        ]
    )


HOME = -RIGHT_FINGERS_HOME
OPEN = -RIGHT_FINGERS_OPEN
CLOSED = -RIGHT_FINGERS_CLOSED


def _cartesian_line(kinematics, start_pose, start, goal, knots):
    positions = np.linspace(start, goal, knots)
    path = [np.asarray(start_pose, dtype=float)]
    seed = path[0]
    for position in positions[1:]:
        seed = kinematics.solve(position, seed)
        path.append(seed)
    return np.asarray(path, dtype=float)


def _slerp(q0: npt.NDArray[np.float64], q1: npt.NDArray[np.float64], t: float) -> npt.NDArray[np.float64]:
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        out = q0 + t * (q1 - q0)
        return out / np.linalg.norm(out)
    theta = np.arccos(dot)
    return (np.sin((1.0 - t) * theta) * q0 + np.sin(t * theta) * q1) / np.sin(theta)


def _pose_line(kinematics, start_pose, start, q_start, goal, q_goal, knots):
    """IK continuation along a straight line while blending the orientation."""
    positions = np.linspace(start, goal, knots)
    path = [np.asarray(start_pose, dtype=float)]
    seed = path[0]
    for index, position in enumerate(positions[1:], start=1):
        orientation = _slerp(np.asarray(q_start), np.asarray(q_goal), index / (knots - 1))
        seed = kinematics.solve_pose(position, orientation, seed)
        path.append(seed)
    return np.asarray(path, dtype=float)


class BimanualOpenArmPolicy:
    """Reference implementation for both OpenArm v2 arms."""

    phase_sequence = MOVE_BOWL_PHASES

    def __init__(self, scenario: task.BallBowlScenario, options: PolicyOptions) -> None:
        self.scenario = scenario
        self.options = options
        self.info = scenario.bot_info
        self.left_motion: dict[str, npt.NDArray[np.float64]] | None = None
        self.right_motion: PlannedMotion | None = None
        groups = self.info.dof_groups
        self.left_mask = np.isin(self.info.arm_dofs, groups["left_arm"])
        self.right_mask = np.isin(self.info.arm_dofs, groups["right_arm"])

    # -- planning ---------------------------------------------------------

    def _freeze(self, path: npt.NDArray[np.float64], mask: npt.NDArray[np.bool_], values):
        """Pin the columns in ``mask`` to ``values`` (the other arm must not move)."""
        path = np.array(path, dtype=float)
        path[:, mask] = np.asarray(values, dtype=float)[mask]
        return path

    def plan(self) -> None:
        spec = self.scenario.specification
        info = self.info
        left = self.scenario.left_kinematics
        right = self.scenario.kinematics
        if left is None:
            raise PlanningError("The bimanual reference needs a left-arm kinematic twin.")
        if spec.bowl_target_xy is None:
            raise PlanningError("The bimanual reference needs a bowl_target_xy to move the bowl to.")
        home = info.default_pose[info.arm_dofs].copy()
        try:
            radius_y = float(spec.bowl_outer_radii[1])
            grasp_z = task.DESK_TOP_Z + spec.bowl_rim_height - RIM_GRASP_DEPTH
            rim_start = np.array([spec.bowl_xy[0], spec.bowl_xy[1] + radius_y - RIM_INSET, grasp_z])
            rim_target = np.array(
                [spec.bowl_target_xy[0], spec.bowl_target_xy[1] + radius_y - RIM_INSET, grasp_z]
            )
            above_start = rim_start + [0.0, 0.0, BOWL_HOVER]
            above_target = rim_target + [0.0, 0.0, BOWL_HOVER]
            drag_lift = np.array([0.0, 0.0, BOWL_DRAG_LIFT])

            # The parked gripper sits at rim height next to the bowl, so rise
            # straight up first, travel above the rim, then descend onto it.
            home_point, home_orientation = left.grasp_point_pose(home)
            above_home = home_point + [0.0, 0.0, HOME_LIFT]
            # Rise with the parked orientation; the pitched rim orientation is
            # blended in on the way to the hover point above the rim.
            q_up = left.solve_pose(above_home, home_orientation, home)
            home_to_up = np.linspace(home, q_up, 6)
            rim_orientation = _quaternion_xyzw(left.level_rotation)
            up_to_above = _pose_line(
                left, q_up, above_home, home_orientation, above_start, rim_orientation, 10
            )
            home_to_above = np.vstack([home_to_up, up_to_above[1:]])
            above_to_rim = _cartesian_line(left, home_to_above[-1], above_start, rim_start, 6)
            rim_to_target = _cartesian_line(
                left, above_to_rim[-1], rim_start + drag_lift, rim_target + drag_lift, 14
            )
            target_to_above = _cartesian_line(left, rim_to_target[-1], rim_target, above_target, 6)
            above_to_up = _pose_line(
                left, target_to_above[-1], above_target, rim_orientation, above_home, home_orientation, 10
            )
            above_to_home = np.vstack([above_to_up, np.linspace(above_to_up[-1], home, 6)[1:]])
            self.left_motion = {
                "home_to_above": self._freeze(home_to_above, self.right_mask, home),
                "above_to_rim": self._freeze(above_to_rim, self.right_mask, home),
                "rim_to_target": self._freeze(rim_to_target, self.right_mask, home),
                "target_to_above": self._freeze(target_to_above, self.right_mask, home),
                "above_to_home": self._freeze(above_to_home, self.right_mask, home),
            }
            self.rim_start = rim_start
            self.rim_target = rim_target

            motion = plan_motion(
                right,
                info,
                spec,
                optimize_trajectory=self.options.optimize_trajectory,
            )
            frozen = [self._freeze(segment, self.left_mask, home) for segment in motion.segments()]
            self.right_motion = PlannedMotion(*frozen)
        except RuntimeError as error:
            raise PlanningError(str(error)) from error

    def _left(self) -> dict[str, npt.NDArray[np.float64]]:
        if self.left_motion is None:
            raise RuntimeError("BimanualOpenArmPolicy.plan() must run before execution.")
        return self.left_motion

    def _right(self) -> PlannedMotion:
        if self.right_motion is None:
            raise RuntimeError("BimanualOpenArmPolicy.plan() must run before execution.")
        return self.right_motion

    def trajectory_points(self) -> npt.NDArray[np.float64]:
        left = self.scenario.left_kinematics
        right = self.scenario.kinematics
        assert left is not None
        points = []
        for segment in self._left().values():
            for q in sample_natural_cubic_spline(segment)[:-1]:
                points.append(left.grasp_point_world(q))
        points.append(np.full(3, np.nan))  # route separator: left arm, then right arm
        for segment in self._right().segments():
            for q in sample_natural_cubic_spline(segment)[:-1]:
                points.append(right.grasp_point_world(q))
        points.append(right.grasp_point_world(self._right().retreat_to_home[-1]))
        return np.asarray(points, dtype=float)

    def home_pose(self) -> npt.NDArray[np.float64]:
        return self.info.target_pose(self._left()["home_to_above"][0], _hand(self.info, HOME, HOME))

    def preshape_pose(self) -> npt.NDArray[np.float64]:
        return self.info.target_pose(self._left()["above_to_rim"][-1], _hand(self.info, HOME, OPEN))

    # -- execution --------------------------------------------------------

    def run(self, runner: EpisodeRunner) -> bool:
        info = self.info
        left = self._left()
        right = self._right()
        home = left["home_to_above"][0]
        hand = lambda r, l: _hand(info, r, l)  # noqa: E731

        print(f"Executing {info.display_name} move-bowl-then-ball episode...")
        runner.phase("home")
        if not runner.hold(home, hand(HOME, HOME), 0.20):
            return False
        runner.phase("preshape")
        if not runner.follow(np.vstack([home, home]), 0.55, hand(HOME, HOME), hand(HOME, OPEN)):
            return False

        # -- left arm: bowl ---------------------------------------------------
        runner.phase("bowl_approach")
        if not runner.follow(left["home_to_above"], 2.2, hand(HOME, OPEN)):
            return False
        if not runner.follow(left["above_to_rim"], 1.0, hand(HOME, OPEN)):
            return False
        rim_pose = left["above_to_rim"][-1]
        if not runner.hold(rim_pose, hand(HOME, OPEN), 0.4):
            return False
        runner.check_bowl_grasp_alignment(self.rim_start, ee_link("left"))
        runner.phase("bowl_grasp")
        if not runner.follow(np.vstack([rim_pose, rim_pose]), 0.8, hand(HOME, OPEN), hand(HOME, RIM_PINCH)):
            return False
        if not runner.hold(rim_pose, hand(HOME, RIM_PINCH), 0.3):
            return False
        runner.phase("bowl_move")
        if not runner.follow(left["rim_to_target"], 2.5, hand(HOME, RIM_PINCH)):
            return False
        moved_pose = left["rim_to_target"][-1]
        if not runner.hold(moved_pose, hand(HOME, RIM_PINCH), 0.3):
            return False
        runner.verify_bowl_moved()
        runner.phase("bowl_release")
        if not runner.follow(np.vstack([moved_pose, moved_pose]), 0.5, hand(HOME, RIM_PINCH), hand(HOME, OPEN)):
            return False
        if not runner.hold(moved_pose, hand(HOME, OPEN), 0.3):
            return False
        runner.phase("bowl_retreat")
        if not runner.follow(left["target_to_above"], 0.8, hand(HOME, OPEN)):
            return False
        if not runner.follow(left["above_to_home"], 2.2, hand(HOME, OPEN), hand(HOME, HOME)):
            return False

        # -- right arm: ball (the ball-and-bowl reference) ----------------------
        pregrasp = right.pre_pick_to_pick[-1]
        place = right.place_safe_to_place[-1]
        runner.phase("approach")
        if not runner.follow(np.vstack([home, home]), 0.4, hand(HOME, HOME), hand(OPEN, HOME)):
            return False
        if not runner.follow(right.home_to_pre_pick, 1.5, hand(OPEN, HOME)):
            return False
        runner.phase("pre_grasp")
        if not runner.follow(right.pre_pick_to_pick, 1.2, hand(OPEN, HOME)):
            return False
        if not runner.hold(pregrasp, hand(OPEN, HOME), 0.65):
            return False
        runner.print_tracking_error(pregrasp)
        runner.check_grasp_alignment()
        runner.phase("grasp")
        if not runner.follow(np.vstack([pregrasp, pregrasp]), 0.9, hand(OPEN, HOME), hand(CLOSED, HOME)):
            return False
        if not runner.hold(pregrasp, hand(CLOSED, HOME), 0.35):
            return False
        runner.phase("lift")
        if not runner.follow(right.pick_to_lift, 1.2, hand(CLOSED, HOME)):
            return False
        runner.verify_physical_grasp()
        runner.phase("carry")
        if not runner.follow(right.lift_to_place_safe, 2.0, hand(CLOSED, HOME)):
            return False
        runner.phase("lower")
        if not runner.follow(right.place_safe_to_place, 0.7, hand(CLOSED, HOME)):
            return False
        runner.phase("release")
        runner.print_release_position()
        if not runner.follow(np.vstack([place, place]), 0.55, hand(CLOSED, HOME), hand(OPEN, HOME)):
            return False
        if not runner.hold(place, hand(OPEN, HOME), 1.15):
            return False
        runner.phase("retreat")
        if not runner.follow(right.place_to_retreat, 0.7, hand(OPEN, HOME)):
            return False
        runner.phase("return_home")
        if not runner.follow(right.retreat_to_home, 1.5, hand(OPEN, HOME), hand(HOME, HOME)):
            return False
        return runner.hold(home, hand(HOME, HOME), 0.35)


__all__ = ["BimanualOpenArmPolicy"]
