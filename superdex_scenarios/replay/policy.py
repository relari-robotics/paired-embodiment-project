"""Episode policy that replays a :class:`JointTrajectory` open loop.

The policy satisfies the same contract as the scripted reference policies
(``plan``, ``trajectory_points``, ``home_pose``, ``preshape_pose``, ``run``),
so every scenario runner can drive it through its unchanged pipeline:
compliant controller, physics, recording, and the task's success check.
Nothing in the file is closed-loop -- the trajectory is executed as given.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from superdex_scenarios.embodiments import EmbodimentModel

from .trajectory import JointTrajectory, phase_sequence_for

DEFAULT_SETTLE_S = 0.5
DEFAULT_TAIL_S = 0.5


class ReplayPolicy:
    """Drive an embodiment along a trajectory file.

    ``scenario`` is any task scenario exposing ``bot_info`` (an
    :class:`EmbodimentModel` with ``dof_groups``) and ``kinematics`` (for the
    trajectory display).  ``options`` are the runner's :class:`PolicyOptions`.
    """

    def __init__(
        self,
        scenario: Any,
        options: Any,
        trajectory: JointTrajectory,
        *,
        settle_s: float = DEFAULT_SETTLE_S,
        tail_s: float = DEFAULT_TAIL_S,
        speed: float = 1.0,
    ) -> None:
        self.scenario = scenario
        self.options = options
        self.info: EmbodimentModel = scenario.bot_info
        self.trajectory = trajectory.rebased()
        self.settle_s = float(settle_s)
        self.tail_s = float(tail_s)
        self.speed = float(speed)
        self.phase_sequence: tuple[str, ...] = phase_sequence_for(self.trajectory)
        self.poses: npt.NDArray[np.float64] | None = None
        self.limit_violations: dict[str, float] = {}
        self.tracking: dict[str, Any] = {}
        self._measured: list[npt.NDArray[np.float64]] = []
        self._targets: list[npt.NDArray[np.float64]] = []

    # -- planning -----------------------------------------------------------

    def plan(self) -> None:
        """Expand the file into full articulation poses and check joint limits."""
        info = self.info
        groups = info.dof_groups
        n = self.trajectory.sample_count
        poses = np.tile(info.default_pose, (n, 1))
        # Every controlled gripper starts from its "home" jaw pose unless the
        # file commands it.
        poses[:, info.hand_dofs] = info.hand_poses["home"]
        missing = [
            f"{side}_arm" for side in self.trajectory.sides if f"{side}_arm" not in groups
        ]
        if missing:
            raise ValueError(
                f"Trajectory drives {missing} but embodiment {info.embodiment_id!r} "
                f"controls only {sorted(groups)}. Use the bimanual scenario for two arms."
            )
        for side, arm in self.trajectory.arms.items():
            poses[:, groups[f"{side}_arm"]] = arm
        for side, fingers in self.trajectory.grippers.items():
            poses[:, groups[f"{side}_gripper"]] = fingers

        # Joint limits: clip, but remember the largest excursion per group so
        # the report can flag an infeasible trajectory.
        dof_count = int(info.actor.get_num_dofs())
        lower = np.full(dof_count, -np.inf)
        upper = np.full(dof_count, np.inf)
        lower[info.arm_dofs] = info.arm_limits[:, 0]
        upper[info.arm_dofs] = info.arm_limits[:, 1]
        lower[info.hand_dofs] = info.hand_limits[:, 0]
        upper[info.hand_dofs] = info.hand_limits[:, 1]
        self.limit_violations = {}
        for name, dofs in groups.items():
            lo = lower[dofs]
            hi = upper[dofs]
            values = poses[:, dofs]
            finite = np.isfinite(lo) & np.isfinite(hi)
            if not np.any(finite):
                continue
            excess = np.maximum(
                np.maximum(lo[finite] - values[:, finite], values[:, finite] - hi[finite]), 0.0
            )
            worst = float(np.max(excess)) if excess.size else 0.0
            if worst > 1e-6:
                self.limit_violations[name] = worst
                print(
                    f"  WARNING: '{name}' exceeds a joint limit by up to {worst:.4f} rad; "
                    "clipping."
                )
            poses[:, dofs] = np.clip(values, lo, hi)
        self.poses = poses
        print(
            f"Replay: {self.trajectory.sample_count} samples over "
            f"{self.trajectory.duration_s:.2f} s, sides {list(self.trajectory.sides)}, "
            f"phases {list(self.phase_sequence)}, speed x{self.speed:g}."
        )

    def _poses(self) -> npt.NDArray[np.float64]:
        if self.poses is None:
            raise RuntimeError("ReplayPolicy.plan() must run before execution.")
        return self.poses

    def trajectory_points(self) -> npt.NDArray[np.float64]:
        """Grasp-point routes of every commanded arm, for validation and display.

        Routes are concatenated with an all-NaN separator row so the viewer can
        draw each arm as its own curve (see ``replay.cli.split_routes``).
        """
        poses = self._poses()
        stride = max(1, len(poses) // 400)
        kinematics = self.scenario.kinematics
        per_side = getattr(kinematics, "grasp_point_world_side", None)
        routes = []
        for side in self.trajectory.sides:
            if per_side is not None:
                route = [per_side(side, pose[self.info.arm_dofs]) for pose in poses[::stride]]
            elif side == "right":
                route = [kinematics.grasp_point_world(pose[self.info.arm_dofs]) for pose in poses[::stride]]
            else:
                continue
            if routes:
                routes.append(np.full((1, 3), np.nan))
            routes.append(np.asarray(route, dtype=float))
        return np.vstack(routes) if routes else np.zeros((0, 3))

    def home_pose(self) -> npt.NDArray[np.float64]:
        return self._poses()[0].copy()

    def preshape_pose(self) -> npt.NDArray[np.float64]:
        return self._poses()[0].copy()

    # -- execution ----------------------------------------------------------

    def run(self, runner: Any) -> bool:
        poses = self._poses()
        times = self.trajectory.time_s
        phases = list(self.trajectory.phases)
        next_phase = 0
        self._measured = []
        self._targets = []

        def on_time(t: float) -> None:
            nonlocal next_phase
            while next_phase < len(phases) and phases[next_phase].time_s <= t + 1e-9:
                runner.phase(phases[next_phase].name)
                next_phase += 1

        measure = getattr(runner, "measured_pose", None)
        controlled = self.info.controlled_dofs

        def on_step(step: int, target: npt.NDArray[np.float64]) -> None:
            if measure is not None and step % 8 == 0:
                self._measured.append(np.asarray(measure(), dtype=float)[controlled])
                self._targets.append(np.asarray(target, dtype=float)[controlled])

        executor = runner.executor
        previous_callback = executor.step_callback

        def chained(step: int, target: npt.NDArray[np.float64]) -> None:
            if previous_callback is not None:
                previous_callback(step, target)
            on_step(step, target)

        executor.step_callback = chained
        try:
            print(f"Executing {self.info.display_name} open-loop replay...")
            if not phases:
                runner.phase(self.phase_sequence[0])
            else:
                on_time(times[0])
            if self.settle_s > 0.0 and not executor.hold(poses[0], self.settle_s):
                return False
            if not executor.track(times, poses, speed=self.speed, on_time=on_time):
                return False
            # Late phase marks (exactly at the final time) are flushed here.
            on_time(times[-1] + 1e-6)
            if self.tail_s > 0.0 and not executor.hold(poses[-1], self.tail_s):
                return False
        finally:
            executor.step_callback = previous_callback
        self._finish_tracking()
        return True

    def _finish_tracking(self) -> None:
        if not self._measured:
            self.tracking = {}
            return
        measured = np.asarray(self._measured)
        targets = np.asarray(self._targets)
        error = measured - targets
        groups = self.info.dof_groups
        controlled = list(self.info.controlled_dofs)
        report: dict[str, Any] = {}
        for name, dofs in groups.items():
            columns = [controlled.index(int(d)) for d in dofs if int(d) in controlled]
            if not columns:
                continue
            e = error[:, columns]
            report[name] = {
                "rms_rad": float(np.sqrt(np.mean(e**2))),
                "max_abs_rad": float(np.max(np.abs(e))),
            }
        self.tracking = report
        print(
            "  tracking error (measured - commanded): "
            + ", ".join(
                f"{name} rms {v['rms_rad']:.3f} / max {v['max_abs_rad']:.3f} rad"
                for name, v in report.items()
            )
        )

    def summary(self) -> dict[str, Any]:
        return {
            "trajectory": self.trajectory.summary(),
            "speed": self.speed,
            "settle_s": self.settle_s,
            "tail_s": self.tail_s,
            "joint_limit_violations_rad": self.limit_violations,
            "tracking_error": self.tracking,
        }


__all__ = ["DEFAULT_SETTLE_S", "DEFAULT_TAIL_S", "ReplayPolicy"]
