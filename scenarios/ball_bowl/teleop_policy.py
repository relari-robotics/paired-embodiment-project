"""Live bimanual OpenArm policy driven by Kyber desk-frame hand poses."""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from superdex_scenarios.camera_profiles import resolve_profile
from superdex_scenarios.embodiments.openarm_v2 import finger_joint_from_aperture
from superdex_scenarios.teleop import TeleopMapping, TeleopReceiver

from . import scenario as task
from .episode import PolicyOptions

_MAX_CATCH_UP_STEPS = 200


def top_down_quaternion_xyzw(
    mapped_quaternion_xyzw: npt.ArrayLike, side: str
) -> npt.NDArray[np.float64]:
    """Point the gripper straight down while keeping the demonstrated jaw yaw.

    The end-effector's local ``-Z`` is finger-forward and local ``Y`` is the
    jaw axis. A horizontal gripper collides with the workcell close to the
    desk; this keeps the mapped jaw direction projected onto the desk plane
    and turns the fingers down, which stays collision-free at every height.
    """

    rotation = Rotation.from_quat(np.asarray(mapped_quaternion_xyzw, dtype=float)).as_matrix()
    z = np.array([0.0, 0.0, 1.0])
    jaw = rotation[:, 1]
    y = jaw - float(np.dot(jaw, z)) * z
    norm = float(np.linalg.norm(y))
    if norm < 1e-6:
        y = np.array([0.0, 1.0 if side == "left" else -1.0, 0.0])
    else:
        y /= norm
    x = np.cross(y, z)
    quaternion = Rotation.from_matrix(np.column_stack((x, y, z))).as_quat()
    return -quaternion if quaternion[3] < 0 else quaternion


def table_parallel_quaternion_xyzw(
    mapped_quaternion_xyzw: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Keep the gripper plane parallel to the desk and preserve hand heading.

    OpenArm local ``-Z`` is finger-forward and local ``Y`` is the jaw axis, so
    both must lie in the horizontal plane. The demonstrated finger-forward
    direction supplies yaw while world-up supplies a fixed surface normal,
    preventing 180-degree wrist flips when the observed palm normal is noisy.
    """

    mapped = Rotation.from_quat(
        np.asarray(mapped_quaternion_xyzw, dtype=float)
    ).as_matrix()
    up = np.array([0.0, 0.0, 1.0])
    forward = -mapped[:, 2]
    forward -= float(np.dot(forward, up)) * up
    norm = float(np.linalg.norm(forward))
    if norm < 1e-6:
        # A nearly vertical finger direction has no tabletop heading. The jaw
        # axis remains orthogonal to it, so it supplies a stable horizontal
        # heading for this singular case.
        jaw = mapped[:, 1] - float(np.dot(mapped[:, 1], up)) * up
        jaw_norm = float(np.linalg.norm(jaw))
        if jaw_norm < 1e-6:
            raise ValueError("mapped gripper has no horizontal heading")
        forward = np.cross(jaw / jaw_norm, up)
    else:
        forward /= norm
    local_x = up
    local_z = -forward
    local_y = np.cross(local_z, local_x)
    rotation = np.column_stack((local_x, local_y, local_z))
    quaternion = Rotation.from_matrix(rotation).as_quat()
    return -quaternion if quaternion[3] < 0 else quaternion


class _LatestPacketIK:
    """Solve only the newest pending packet without blocking physics stepping."""

    def __init__(
        self,
        solve: Callable[[dict, npt.NDArray[np.float64]], tuple[int, dict[str, int]]],
    ) -> None:
        self._solve = solve
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="teleop-ik")
        self._future: Future | None = None
        self._pending: dict | None = None
        self._generation = 0

    def offer(self, packet: dict) -> None:
        self._pending = packet

    def reset(self) -> None:
        """Discard pending and in-flight results after stale-input recovery."""

        self._generation += 1
        self._pending = None

    def _run(
        self,
        generation: int,
        packet: dict,
        candidate: npt.NDArray[np.float64],
    ) -> tuple[int, npt.NDArray[np.float64], int, dict[str, int]]:
        solved, kinds = self._solve(packet, candidate)
        return generation, candidate, solved, kinds

    def _collect(
        self, desired: npt.NDArray[np.float64]
    ) -> tuple[int, dict[str, int]]:
        if self._future is None or not self._future.done():
            return 0, {}
        generation, candidate, solved, kinds = self._future.result()
        self._future = None
        if generation != self._generation:
            return 0, {}
        desired[:] = candidate
        return solved, kinds

    def update(
        self, desired: npt.NDArray[np.float64]
    ) -> tuple[int, dict[str, int], int]:
        """Publish a completed result and start the newest pending packet."""

        solved, kinds = self._collect(desired)
        started = 0
        if self._future is None and self._pending is not None:
            packet = self._pending
            self._pending = None
            self._future = self._executor.submit(
                self._run,
                self._generation,
                packet,
                desired.copy(),
            )
            started = 1
        return solved, kinds, started

    def close(
        self, desired: npt.NDArray[np.float64]
    ) -> tuple[int, dict[str, int]]:
        self._pending = None
        self._executor.shutdown(wait=True, cancel_futures=True)
        return self._collect(desired)


class TeleopPolicy:
    """Apply fresh absolute bimanual hand targets while holding on invalid input.

    Orientation handling is selected by the mapping. Fixed table-parallel mode
    keeps the demonstrated finger heading while holding the gripper plane
    level; adaptive mode retains the mapped/top-down/level fallback chain for
    comparison.
    """

    phase_sequence = ("teleop",)

    def __init__(
        self,
        scenario: task.BallBowlScenario,
        options: PolicyOptions,
        *,
        endpoint: str,
        mapping_path: Path,
        duration_s: float,
    ) -> None:
        self.scenario = scenario
        self.options = options
        self.endpoint = endpoint
        self.mapping = TeleopMapping.load(mapping_path)
        self.duration_s = float(duration_s)
        if self.duration_s < 0:
            raise ValueError("teleop duration must be non-negative")
        _, self.profile_sha256 = resolve_profile(
            Path(__file__).resolve().parent / "camera_calibration.json"
        )
        self.info = scenario.bot_info
        self.kinematics = {
            "right": scenario.kinematics,
            "left": scenario.left_kinematics,
        }

    def plan(self) -> None:
        required = {"right_arm", "left_arm", "right_gripper", "left_gripper"}
        if not required.issubset(self.info.dof_groups):
            raise ValueError("teleoperation requires the bimanual OpenArm embodiment")
        if self.kinematics["left"] is None:
            raise ValueError("teleoperation requires a left-arm kinematic twin")

    def trajectory_points(self) -> npt.NDArray[np.float64]:
        return np.empty((0, 3), dtype=float)

    def home_pose(self) -> npt.NDArray[np.float64]:
        return self.info.default_pose.copy()

    def preshape_pose(self) -> npt.NDArray[np.float64]:
        return self.home_pose()

    ATTEMPT_KINDS = (
        "mapped",
        "table-parallel",
        "top-down",
        "level",
        "table-parallel/home",
        "top-down/home",
    )

    def _home_seed(self, side: str, arm_pose: npt.NDArray[np.float64]):
        home = getattr(self.info, "default_pose", None)
        if home is None:
            return None
        side_mask = np.isin(self.info.arm_dofs, self.info.dof_groups[f"{side}_arm"])
        seed = arm_pose.copy()
        seed[side_mask] = np.asarray(home, dtype=float)[self.info.arm_dofs][side_mask]
        return seed

    def _solve_hand(
        self, kinematics, side: str, mapped, arm_pose: npt.NDArray[np.float64]
    ) -> tuple[str, npt.NDArray[np.float64]]:
        iterations = getattr(self.mapping, "ik_max_iterations", None)
        top_down = top_down_quaternion_xyzw(mapped.quaternion_world_xyzw, side)
        table_parallel = table_parallel_quaternion_xyzw(
            mapped.quaternion_world_xyzw
        )
        orientation_mode = getattr(self.mapping, "orientation_mode", "adaptive")
        errors = []
        if orientation_mode == "top-down":
            attempts = [("top-down", top_down, arm_pose)]
            home_kind = "top-down/home"
            home_orientation = top_down
        elif orientation_mode == "table-parallel":
            attempts = [("table-parallel", table_parallel, arm_pose)]
            home_kind = "table-parallel/home"
            home_orientation = table_parallel
        else:
            attempts = [
                ("mapped", mapped.quaternion_world_xyzw, arm_pose),
                ("top-down", top_down, arm_pose),
            ]
            home_kind = "top-down/home"
            home_orientation = top_down
        for kind, quaternion, seed in attempts:
            try:
                return kind, kinematics.solve_pose_optimized(
                    mapped.position_world_m,
                    quaternion,
                    seed,
                    max_iterations=iterations,
                )
            except (RuntimeError, ValueError) as error:
                errors.append(f"{kind}: {error}")
        if orientation_mode == "adaptive":
            try:
                return "level", kinematics.solve_optimized(
                    mapped.position_world_m, arm_pose, max_iterations=iterations
                )
            except (RuntimeError, ValueError) as error:
                errors.append(f"level: {error}")
        home_seed = self._home_seed(side, arm_pose)
        if home_seed is not None:
            try:
                # The home seed is far from the target, so the last resort gets
                # twice the warm-start budget without changing orientation.
                return home_kind, kinematics.solve_pose_optimized(
                    mapped.position_world_m,
                    home_orientation,
                    home_seed,
                    max_iterations=None if iterations is None else 2 * iterations,
                )
            except (RuntimeError, ValueError) as error:
                errors.append(f"{home_kind}: {error}")
        raise RuntimeError("; ".join(errors))

    def _solve_packet(
        self,
        packet: dict,
        desired: npt.NDArray[np.float64],
        last_error_s: dict[str, float],
        accepted_kinds: dict[str, int] | None = None,
    ) -> int:
        arm_pose = desired[self.info.arm_dofs].copy()
        solved = 0
        for hand in packet["hands"]:
            side = str(hand["side"])
            previous = arm_pose.copy()
            try:
                mapped = self.mapping.map_hand(hand)
                kinematics = self.kinematics[side]
                assert kinematics is not None
                kind, candidate = self._solve_hand(kinematics, side, mapped, arm_pose)
                _, clearance = kinematics.collision_cost(candidate)
                if clearance < 0.0:
                    raise RuntimeError(
                        f"candidate intersects the workcell by {-clearance:.4f} m"
                    )
                arm_pose = candidate
                gripper = self.info.dof_groups[f"{side}_gripper"]
                desired[gripper] = finger_joint_from_aperture(mapped.aperture, side)
                solved += 1
                if accepted_kinds is not None:
                    accepted_kinds[kind] = accepted_kinds.get(kind, 0) + 1
            except (RuntimeError, ValueError) as error:
                arm_pose = previous
                now = time.monotonic()
                if now - last_error_s[side] >= 1.0:
                    print(f"  teleop {side} target held: {error}", flush=True)
                    last_error_s[side] = now
        desired[self.info.arm_dofs] = arm_pose
        return solved

    @staticmethod
    def _advance_physics(
        runner,
        current,
        desired,
        filtered,
        velocity,
        controlled,
        max_velocity,
        max_acceleration,
        target_filter_time_constant_s,
        joint_tracking_time_constant_s,
    ) -> bool:
        """Catch simulated time up to wall time after a scheduler delay."""

        executor = runner.executor
        dt = task.TIME_STEP
        filter_alpha = -np.expm1(-dt / target_filter_time_constant_s)
        natural_frequency = 2.0 / joint_tracking_time_constant_s
        for _ in range(_MAX_CATCH_UP_STEPS):
            filtered[controlled] += filter_alpha * (
                desired[controlled] - filtered[controlled]
            )
            acceleration = (
                natural_frequency**2
                * (filtered[controlled] - current[controlled])
                - 2.0 * natural_frequency * velocity[controlled]
            )
            acceleration = np.clip(
                acceleration,
                -max_acceleration[controlled],
                max_acceleration[controlled],
            )
            velocity[controlled] += acceleration * dt
            velocity[controlled] = np.clip(
                velocity[controlled],
                -max_velocity[controlled],
                max_velocity[controlled],
            )
            current[controlled] += velocity[controlled] * dt
            if not executor.step(current):
                return False
            if not getattr(executor, "real_time", False):
                return True
            wall_start = getattr(executor, "wall_start", None)
            if wall_start is None:
                return True
            simulated_wall_s = wall_start + executor.step_count * task.TIME_STEP
            if simulated_wall_s >= time.perf_counter():
                return True
        return True

    def run(self, runner) -> bool:
        print(
            f"Listening for bimanual Kyber teleoperation on {self.endpoint}; "
            "tracking loss holds each arm independently."
        )
        runner.phase("teleop")
        current = runner.measured_pose()
        desired = current.copy()
        filtered = current.copy()
        velocity = np.zeros_like(current)
        controlled = self.info.controlled_dofs
        max_velocity = np.zeros(len(current), dtype=float)
        max_velocity[self.info.arm_dofs] = self.mapping.max_arm_joint_speed_rad_s
        max_velocity[self.info.hand_dofs] = self.mapping.max_gripper_joint_speed_rad_s
        max_acceleration = np.zeros(len(current), dtype=float)
        max_acceleration[self.info.arm_dofs] = (
            self.mapping.max_arm_joint_acceleration_rad_s2
        )
        max_acceleration[self.info.hand_dofs] = (
            self.mapping.max_gripper_joint_acceleration_rad_s2
        )
        # PoseExecutor normally initializes this clock on its first step. Live
        # teleop anchors it at policy start so _advance_physics can execute
        # every step that becomes due during a scheduler delay.
        if getattr(runner.executor, "real_time", False):
            runner.executor.wall_start = (
                time.perf_counter() - runner.executor.step_count * task.TIME_STEP
            )
        last_error_s = {"left": -np.inf, "right": -np.inf}
        reset_for_stale_packet = False
        packet_count = 0
        solved_hand_count = 0
        accepted_kinds: dict[str, int] = {}

        def solve_packet(packet, candidate):
            kinds: dict[str, int] = {}
            solved = self._solve_packet(packet, candidate, last_error_s, kinds)
            return solved, kinds

        async_ik = _LatestPacketIK(solve_packet)

        def collect_ik() -> None:
            nonlocal packet_count, solved_hand_count
            solved, kinds, started = async_ik.update(desired)
            packet_count += started
            solved_hand_count += solved
            for kind, count in kinds.items():
                accepted_kinds[kind] = accepted_kinds.get(kind, 0) + count

        start = time.monotonic()
        with_receiver = TeleopReceiver(
            self.endpoint,
            self.profile_sha256,
            max_age_s=self.mapping.max_age_s,
        )
        try:
            while self.duration_s == 0 or time.monotonic() - start < self.duration_s:
                now = time.monotonic()
                packet, updated = with_receiver.poll(now)
                if packet is not None and updated:
                    async_ik.offer(packet)
                    reset_for_stale_packet = False
                collect_ik()
                latest = with_receiver.latest
                if (
                    latest is not None
                    and now - float(latest["capture_timestamp_s"])
                    > self.mapping.reset_age_s
                    and not reset_for_stale_packet
                ):
                    current = runner.measured_pose()
                    filtered[controlled] = current[controlled]
                    desired[controlled] = current[controlled]
                    velocity[controlled] = 0.0
                    async_ik.reset()
                    reset_for_stale_packet = True
                if not self._advance_physics(
                    runner,
                    current,
                    desired,
                    filtered,
                    velocity,
                    controlled,
                    max_velocity,
                    max_acceleration,
                    self.mapping.target_filter_time_constant_s,
                    self.mapping.joint_tracking_time_constant_s,
                ):
                    return False
        except KeyboardInterrupt:
            return False
        finally:
            solved, kinds = async_ik.close(desired)
            solved_hand_count += solved
            for kind, count in kinds.items():
                accepted_kinds[kind] = accepted_kinds.get(kind, 0) + count
            with_receiver.close()
            kinds = ", ".join(
                f"{accepted_kinds.get(kind, 0)} {kind}" for kind in self.ATTEMPT_KINDS
            )
            print(
                f"  teleop summary: {packet_count} fresh packets, "
                f"{solved_hand_count} accepted hand targets ({kinds}), "
                f"{with_receiver.invalid_packets} invalid packets ignored",
                flush=True,
            )
        return True


__all__ = [
    "TeleopPolicy",
    "table_parallel_quaternion_xyzw",
    "top_down_quaternion_xyzw",
]
