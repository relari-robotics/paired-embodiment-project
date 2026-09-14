"""Pose-relative motor policy for thin packets; reuses measured-state tracking."""

import numpy as np
from scenarios.organizer.openarm_policy import OpenArmPolicy
from .geometry import GRASP
from .scenario import DESK_TOP_Z


class TeaPolicy(OpenArmPolicy):
    def __init__(self, scenario, options=None):
        super().__init__(scenario, options)
        self.jobs = [
            ("part", i, actor, GRASP) for i, actor in enumerate(scenario.parts)
        ]

    def select_arm(self, kind, index):
        super().select_arm(kind, index)
        mask = np.isin(
            self.info.hand_dofs, self.info.dof_groups[f"{self.side}_gripper"]
        )
        self.grip[mask] = 0.0

    def run(self, runner):
        runner.phase("home")
        if not runner.hold(self.home, "open", 0.5):
            return False
        for kind, index, actor, offset in self.jobs:
            self.select_arm(kind, index)
            runner.select_arm(self.side)
            name = f"tea_{index}"
            self.kin.collision_model.active = actor
            target = self.scenario.target(kind, index)
            pick, pre, lift, hover = self.transfer_waypoints(actor, offset, target)
            hover[2] = max(hover[2], DESK_TOP_Z + 0.157)
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
            if rise < 0.055:
                raise RuntimeError(f"{name}: physical grasp failed (rise {rise:.4f} m)")
            held = runner.grasp_position() - self.position(actor)
            hover[:2] = (target + held)[:2]
            runner.phase(f"{name}/carry")
            if not self.move(runner, hover, self.grip, 1.8, free=True, precise=True):
                return False
            bounds = actor.get_aabb_world()
            centre = (np.asarray(bounds.min) + np.asarray(bounds.max)) / 2
            placement = runner.grasp_position().copy()
            placement[:2] += target[:2] - centre[:2]
            placement[2] += target[2] + 0.075 - np.asarray(bounds.min)[2]
            runner.phase(f"{name}/place")
            if not self.move(runner, placement, self.grip, 1.5, precise=True):
                return False
            runner.phase(f"{name}/release")
            q = runner.measured_pose()[self.info.arm_dofs]
            release = self.open_hand.copy()
            mask = np.isin(
                self.info.hand_dofs, self.info.dof_groups[f"{self.side}_gripper"]
            )
            release[mask] = 0.13 if self.side == "left" else -0.13
            if not runner.follow(np.vstack([q, q]), 0.6, self.grip, release):
                return False
            if not runner.hold(q, release, 0.8):
                return False
            runner.phase(f"{name}/retreat")
            if not self.move(runner, hover, release, 0.9):
                return False
            q = runner.measured_pose()[self.info.arm_dofs]
            if not runner.follow(np.vstack([q, q]), 0.6, release, self.open_hand):
                return False
            if not runner.hold(q, self.open_hand, 0.8):
                return False
            status = self.scenario.outcome()["parts"][index]
            # Lightweight sachets can lean briefly on the rim before sliding
            # fully into their compartment. Check actual geometry and motion.
            for _ in range(30):
                if status["success"]:
                    break
                if not runner.hold(q, self.open_hand, 0.5):
                    return False
                status = self.scenario.outcome()["parts"][index]
            runner.event("placement_checked", **status)
            if not status["success"]:
                raise RuntimeError(f"{name}: placement failed: {status}")
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
        return runner.hold(self.home, "home", 1.0)
