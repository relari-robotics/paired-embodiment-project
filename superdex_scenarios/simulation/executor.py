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

    def track(
        self,
        times: npt.ArrayLike,
        poses: npt.ArrayLike,
        *,
        speed: float = 1.0,
        on_time: Callable[[float], None] | None = None,
    ) -> bool:
        """Follow a timed pose trajectory open loop, one physics step at a time.

        ``times`` (seconds, strictly increasing) and ``poses`` (one full
        articulation pose per row) are linearly interpolated at every physics
        step; ``speed`` scales playback (2.0 plays twice as fast).  ``on_time``
        receives the trajectory time reached before each step so callers can
        mark phase boundaries.  Unlike :meth:`follow`, no time warping or
        smoothing is applied: what the file says is what the controller gets.
        """
        times = np.asarray(times, dtype=float).reshape(-1)
        poses = np.asarray(poses, dtype=float)
        if len(times) != len(poses) or len(times) < 1:
            raise ValueError("track() needs one pose per time sample.")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("track() times must be strictly increasing.")
        if speed <= 0.0:
            raise ValueError("track() speed must be positive.")
        duration = (times[-1] - times[0]) / speed
        steps = max(1, int(round(duration / self.time_step)))
        for step in range(steps + 1):
            t = times[0] + min(step * self.time_step * speed, times[-1] - times[0])
            if on_time is not None:
                on_time(t)
            pose = np.array([np.interp(t, times, poses[:, d]) for d in range(poses.shape[1])])
            if not self.step(pose):
                return False
        return True


class KinematicExecutor(PoseExecutor):
    """A :class:`PoseExecutor` that sets poses directly instead of stepping physics.

    Used for kinematic playback (``--kinematic``): the articulation is placed
    exactly at every target, no controller or contact acts, objects stay where
    they are, and the recorder/viewer callbacks still fire so a replay can be
    watched or exported without simulating.
    """

    def __init__(
        self,
        scene: physics.Scene,
        actor: physics.Actor,
        time_step: float,
        render_every_steps: int,
        viewer: object | None = None,
        real_time: bool = False,
        frame_callback: Callable[[], None] | None = None,
        step_callback: Callable[[int, npt.NDArray[np.float64]], None] | None = None,
    ) -> None:
        super().__init__(
            scene,
            None,  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
            time_step,
            render_every_steps,
            viewer=viewer,
            real_time=real_time,
            frame_callback=frame_callback,
            step_callback=step_callback,
        )
        self.actor = actor
        self._real = np.float64 if physics.uses_double_precision() else np.float32

    def step(self, pose: npt.ArrayLike) -> bool:
        pose = np.asarray(pose, dtype=float)
        self.actor.set_articulated_pose_from_joints(np.asarray(pose, dtype=self._real))
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
