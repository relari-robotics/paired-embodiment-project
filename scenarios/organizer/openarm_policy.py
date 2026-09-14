"""State-based pick, seat, release policy using physical jaw contact only.

The simulator supplies object poses (privileged state), not pre-recorded joint
trajectories. Every transfer is replanned from measured robot/object state.
Placement compensates the measured grasp offset after lifting. No object is
teleported, attached to a gripper, or marked seated by the policy.
"""

import numpy as np
from superdex import physics

from superdex_scenarios.embodiments.openarm_v2 import right_gripper_pose
from superdex_scenarios.planning import sample_natural_cubic_spline
from .episode import PlanningError, PolicyOptions
from .geometry import DIVIDER_GRASP, PART_GRASP
from .scenario import DESK_TOP_Z


class OpenArmPolicy:
    def __init__(self, scenario, options=None):
        self.scenario = scenario
        self.options = options or PolicyOptions()
        self.info = scenario.bot_info
        self.kin = scenario.kinematics
        self.left_divider_rotation = scenario.left_kinematics.level_rotation
        self.left_part_rotation = (
            physics.Quaternion.from_rotation_vector([0, np.radians(-5), 0])
            * self.left_divider_rotation
        )
        self.home = self.info.default_pose[self.info.arm_dofs].copy()
        self.points = []
        self.jobs = [
            ("divider", i, a, DIVIDER_GRASP) for i, a in enumerate(scenario.dividers)
        ]
        self.jobs += [("part", i, a, PART_GRASP) for i, a in enumerate(scenario.parts)]
        self.grip = right_gripper_pose(self.info, [-0.025, -0.025])

    def home_pose(self):
        return self.info.target_pose(self.home, "home")

    def preshape_pose(self):
        return self.home_pose()

    @staticmethod
    def position(actor, offset=(0, 0, 0)):
        return np.asarray(
            (
                actor.get_root_transform() * physics.TransformRT(translation=offset)
            ).translation,
            dtype=float,
        )

    def select_arm(self, kind, index):
        self.side = (
            "left"
            if (kind == "divider" and index == 1) or (kind == "part" and index == 2)
            else "right"
        )
        self.kin = (
            self.scenario.left_kinematics
            if self.side == "left"
            else self.scenario.kinematics
        )
        if self.side == "left":
            self.kin.level_rotation = (
                self.left_part_rotation
                if kind == "part"
                else self.left_divider_rotation
            )
        self.open_hand = self.info.hand_poses["home"].copy()
        self.grip = self.open_hand.copy()
        mask = np.isin(
            self.info.hand_dofs, self.info.dof_groups[f"{self.side}_gripper"]
        )
        self.open_hand[mask] = 0.78 if self.side == "left" else -0.78
        self.grip[mask] = 0.025 if self.side == "left" else -0.025

    def solve(self, position, seed):
        # Coarse IK is followed by measured Cartesian tracking correction at contact.
        seed = np.array(seed, dtype=float)
        inactive = ~np.isin(
            self.info.arm_dofs, self.info.dof_groups[f"{self.side}_arm"]
        )
        seed[inactive] = self.home[inactive]
        rotation = np.asarray(self.kin.level_rotation, dtype=float)
        return self.kin.solve_pose(position, rotation, seed, position_tolerance=0.015)

    def line(self, seed, destination):
        start = self.kin.grasp_point_world(seed)
        knots = max(3, int(np.ceil(np.linalg.norm(destination - start) / 0.018)) + 1)
        path = [seed]
        for p in np.linspace(start, destination, knots)[1:]:
            path.append(self.solve(p, path[-1]))
        return np.asarray(path)

    def transfer_waypoints(self, actor, offset, target):
        pick = self.position(actor, offset)
        hover_z = max(
            DESK_TOP_Z + 0.115, pick[2] + 0.063, target[2] + offset[2] + 0.055
        )
        # Descend from above; the pitched jaws clear the desk during approach.
        pre = pick + [-0.025, 0, 0.045]
        return (
            pick,
            pre,
            np.array([pick[0], pick[1], hover_z]),
            np.array([target[0] + offset[0], target[1], hover_z]),
        )

    def plan(self):
        self.points = []
        q = self.home.copy()
        try:
            for kind, i, actor, offset in self.jobs:
                self.select_arm(kind, i)
                q = self.home.copy()
                pick, pre, lift, hover = self.transfer_waypoints(
                    actor, offset, self.scenario.target(kind, i)
                )
                safe = self.kin.grasp_point_world(q).copy()
                safe[2] = lift[2]
                for p in (
                    safe,
                    np.array([pre[0], pre[1], lift[2]]),
                    pre,
                    pick,
                    lift,
                    hover,
                    self.scenario.target(kind, i)
                    + offset
                    + [0, 0, 0.045 if kind == "part" else 0],
                ):
                    path = self.line(q, p)
                    self.points.extend(self.kin.grasp_point_world(v) for v in path)
                    q = path[-1]
                q = self.line(q, hover)[-1]
            self.points.append(self.kin.grasp_point_world(self.home))
        except RuntimeError as exc:
            raise PlanningError(str(exc)) from exc

    def trajectory_points(self):
        return np.asarray(self.points)

    def move(
        self, runner, destination, hand, duration=1.0, *, free=False, precise=False
    ):
        seed = runner.measured_pose()[self.info.arm_dofs]
        path = self.line(seed, np.asarray(destination))
        if free and self.options.check_collisions:
            # Preserve Cartesian orientation and the parked arm. Unconstrained
            # joint-space shortening can sweep the held object through the tray.
            clearance = min(
                self.kin.collision_cost(q)[1] for q in sample_natural_cubic_spline(path)
            )
            if clearance < -0.005:
                raise PlanningError(
                    f"Free-space route intersects an obstacle proxy by {-clearance:.3f} m"
                )
        if not runner.follow(path, duration, hand):
            return False
        if precise:
            # Correct compliant tracking sag using measured end-effector position.
            # This changes only robot targets, never object state.
            command = np.asarray(destination).copy()
            for _ in range(5):
                error = destination - runner.grasp_position()
                if np.linalg.norm(error) < 0.0008:
                    break
                if np.linalg.norm(error) > 0.020:
                    raise RuntimeError(
                        f"Blocked motion: tracking error {np.linalg.norm(error):.3f} m"
                    )
                command += np.clip(error, -0.006, 0.006)
                q = self.solve(command, runner.measured_pose()[self.info.arm_dofs])
                if not runner.follow(
                    np.vstack([runner.measured_pose()[self.info.arm_dofs], q]),
                    0.25,
                    hand,
                ):
                    return False
        return True

    def run(self, runner):
        runner.phase("home")
        if not runner.hold(self.home, "open", 0.5):
            return False
        for kind, index, actor, offset in self.jobs:
            self.select_arm(kind, index)
            runner.select_arm(self.side)
            name = f"{kind}_{index}"
            self.kin.collision_model.active = actor
            pick, pre, lift, hover = self.transfer_waypoints(
                actor, offset, self.scenario.target(kind, index)
            )
            runner.phase(f"{name}/approach")
            safe = runner.grasp_position().copy()
            safe[2] = lift[2]
            for p, duration, free in (
                (safe, 0.8, False),
                ([pre[0], pre[1], lift[2]], 1.4, True),
                (pre, 0.8, False),
                (pick, 1.0, False),
            ):
                if not self.move(
                    runner,
                    np.asarray(p),
                    self.open_hand,
                    duration,
                    free=free,
                    precise=not free,
                ):
                    return False
            runner.phase(f"{name}/grasp")
            q = runner.measured_pose()[self.info.arm_dofs]
            if not runner.follow(np.vstack([q, q]), 0.8, self.open_hand, self.grip):
                return False
            if not runner.hold(q, self.grip, 0.3):
                return False
            before = self.position(actor)
            runner.phase(f"{name}/lift")
            if not self.move(runner, lift, self.grip, 1.3, precise=True):
                return False
            rise = self.position(actor)[2] - before[2]
            if rise < 0.060:
                raise RuntimeError(
                    f"{name}: physical grasp failed (object rose {rise:.4f} m)"
                )
            # Measure the held object's offset, including any slip during grasp.
            held_offset = runner.grasp_position() - self.position(actor)
            runner.phase(f"{name}/carry")
            target = self.scenario.target(kind, index)
            hover[:2] = (target + held_offset)[:2]
            if not self.move(runner, hover, self.grip, 1.8, free=True, precise=True):
                return False
            held_offset = runner.grasp_position() - self.position(actor)
            placement = target + held_offset
            if kind == "part":
                # Keep the jaws above the installed walls and release a short
                # distance above the bin. Account for held orientation via AABB.
                bounds = actor.get_aabb_world()
                centre = (np.asarray(bounds.min) + np.asarray(bounds.max)) / 2
                placement[:2] = target[:2] + runner.grasp_position()[:2] - centre[:2]
                placement[2] = (
                    target[2]
                    + 0.045
                    + runner.grasp_position()[2]
                    - np.asarray(bounds.min)[2]
                )
            else:
                placement[2] += 0.001
            runner.phase(f"{name}/seat")
            if not self.move(runner, placement, self.grip, 1.5, precise=True):
                return False
            runner.phase(f"{name}/release")
            q = runner.measured_pose()[self.info.arm_dofs]
            if self.side == "right" and kind == "part":
                hover[2] = max(hover[2], DESK_TOP_Z + 0.157)
            if self.side == "left" and kind == "part":
                hover[2] = max(hover[2], DESK_TOP_Z + 0.147)
                path = self.line(q, hover)
                if not runner.follow(path, 0.75, self.grip, self.open_hand):
                    return False
            elif kind == "part":
                release_hand = self.open_hand.copy()
                mask = np.isin(
                    self.info.hand_dofs, self.info.dof_groups[f"{self.side}_gripper"]
                )
                release_hand[mask] = 0.45 if self.side == "left" else -0.45
                if not runner.follow(np.vstack([q, q]), 0.35, self.grip, release_hand):
                    return False
                # Withdraw before opening fully so a falling/rebounding part
                # cannot be caught by the moving fingertips.
                if not self.move(runner, hover, release_hand, 0.6):
                    return False
                q = runner.measured_pose()[self.info.arm_dofs]
                if not runner.follow(
                    np.vstack([q, q]), 0.35, release_hand, self.open_hand
                ):
                    return False
            else:
                if not runner.follow(np.vstack([q, q]), 0.7, self.grip, self.open_hand):
                    return False
                if not runner.hold(q, self.open_hand, 0.4):
                    return False
            runner.phase(f"{name}/retreat")
            if not self.move(runner, hover, self.open_hand, 1.1):
                return False
            if not runner.hold(
                runner.measured_pose()[self.info.arm_dofs], self.open_hand, 0.35
            ):
                return False
            status = self.scenario.outcome()[kind + "s"][index]
            runner.event("placement_checked", **status)
            if not status["success"]:
                raise RuntimeError(f"{name}: placement failed: {status}")
            # Leave the tray laterally at clearance height before folding the arm.
            # A direct joint interpolation to home sweeps the fingers through dividers.
            exit_point = np.array(
                [0.010, -0.230 if self.side == "right" else 0.230, hover[2]]
            )
            if not self.move(runner, exit_point, self.open_hand, 1.3, free=True):
                return False
            q = runner.measured_pose()[self.info.arm_dofs]
            if not runner.follow(
                np.linspace(q, self.home, 12), 1.5, self.open_hand, "home"
            ):
                return False
        runner.phase("return_home")
        self.kin.collision_model.active = None
        q = runner.measured_pose()[self.info.arm_dofs]
        if not runner.follow(np.linspace(q, self.home, 12), 1.6, "open", "home"):
            return False
        return runner.hold(self.home, "home", 0.75)
