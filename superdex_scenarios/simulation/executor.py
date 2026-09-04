"""Smooth articulated-pose execution independent of a task or embodiment."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from scipy.interpolate import CubicSpline
from superdex import physics, robotics


class PoseExecutor:
    """Advance physics while tracking smooth full-articulation pose targets."""

    def __init__(
        self,
        scene: physics.Scene,
        controller: robotics.ControllerBase,
        target: robotics.ControllerMochiArticulatedPoseTarget,
        time_step: float,
        render_every_steps: int,
        viewer: object | None = None,
        real_time: bool = False,
        frame_callback: Callable[[], None] | None = None,
        step_callback: Callable[[int, npt.NDArray[np.float64]], None] | None = None,
    ) -> None:
        self.scene = scene
        self.controller = controller
        self.target = target
        self.time_step = time_step
        self.render_every_steps = render_every_steps
        self.viewer = viewer
        self.real_time = real_time
        self.frame_callback = frame_callback
        self.step_callback = step_callback
        self.step_count = 0
        self.wall_start: float | None = None

    @staticmethod
    def minimum_jerk(fraction: float) -> float:
        fraction = float(np.clip(fraction, 0.0, 1.0))
        return fraction**3 * (10.0 - 15.0 * fraction + 6.0 * fraction**2)

    def step(self, pose: npt.ArrayLike) -> bool:
        pose = np.asarray(pose, dtype=float)
        self.target.pose_dofs = pose
        self.controller.compute_output(
            robotics.ControllerMochiArticulatedPoseObsv(), self.target
        )
        self.scene.step(self.time_step)
        self.step_count += 1
        if self.step_callback is not None:
            self.step_callback(self.step_count, pose)
        if self.real_time:
            if self.wall_start is None:
                self.wall_start = time.perf_counter() - self.time_step
            remaining = (
                self.wall_start + self.step_count * self.time_step - time.perf_counter()
            )
            if remaining > 0.0:
                time.sleep(remaining)
        if self.step_count % self.render_every_steps == 0:
            if self.frame_callback is not None:
                self.frame_callback()
            if self.viewer is not None:
                self.viewer.render()
                return not self.viewer.user_requested_close()
        return True

    def follow(self, path: npt.ArrayLike, duration: float) -> bool:
        path = np.asarray(path, dtype=float)
        spline = CubicSpline(
            np.linspace(0.0, 1.0, len(path)), path, axis=0, bc_type="natural"
        )
        steps = max(2, round(duration / self.time_step))
        for step in range(steps):
            progress = self.minimum_jerk(step / (steps - 1))
            if not self.step(spline(progress)):
                return False
        return True

    def interpolate(
        self, start: npt.ArrayLike, end: npt.ArrayLike, duration: float
    ) -> bool:
        return self.follow(np.vstack((start, end)), duration)

    def hold(self, pose: npt.ArrayLike, duration: float) -> bool:
        return self.follow(np.vstack((pose, pose)), duration)
