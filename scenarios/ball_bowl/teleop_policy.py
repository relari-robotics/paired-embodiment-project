"""Live bimanual OpenArm policy driven by Kyber desk-frame hand poses."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import numpy.typing as npt

from superdex_scenarios.camera_profiles import resolve_profile
from superdex_scenarios.embodiments.openarm_v2 import finger_joint_from_aperture
from superdex_scenarios.teleop import TeleopMapping, TeleopReceiver

from . import scenario as task
from .episode import PolicyOptions


class TeleopPolicy:
    """Apply fresh absolute bimanual hand targets while holding on invalid input."""

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

    def _solve_packet(
        self,
        packet: dict,
        desired: npt.NDArray[np.float64],
        last_error_s: dict[str, float],
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
                try:
                    candidate = kinematics.solve_pose_optimized(
                        mapped.position_world_m,
                        mapped.quaternion_world_xyzw,
                        arm_pose,
                    )
                except (RuntimeError, ValueError):
                    # Preserve the MANO-retargeted position when the operator's
                    # palm orientation would push OpenArm beyond a joint limit.
                    # This is a direct per-frame IK fallback, not trajectory
                    # optimization; the outer loop still rate-limits joints.
                    candidate = kinematics.solve_optimized(
                        mapped.position_world_m, arm_pose
                    )
                _, clearance = kinematics.collision_cost(candidate)
                if clearance < 0.0:
                    raise RuntimeError(
                        f"candidate intersects the workcell by {-clearance:.4f} m"
                    )
                arm_pose = candidate
                gripper = self.info.dof_groups[f"{side}_gripper"]
                desired[gripper] = finger_joint_from_aperture(mapped.aperture, side)
                solved += 1
            except (RuntimeError, ValueError) as error:
                arm_pose = previous
                now = time.monotonic()
                if now - last_error_s[side] >= 1.0:
                    print(f"  teleop {side} target held: {error}")
                    last_error_s[side] = now
        desired[self.info.arm_dofs] = arm_pose
        return solved

    def run(self, runner) -> bool:
        print(
            f"Listening for bimanual Kyber teleoperation on {self.endpoint}; "
            "tracking loss holds each arm independently."
        )
        runner.phase("teleop")
        current = runner.measured_pose()
        desired = current.copy()
        controlled = self.info.controlled_dofs
        max_step = np.full(len(current), np.inf, dtype=float)
        max_step[self.info.arm_dofs] = (
            self.mapping.max_arm_joint_speed_rad_s * task.TIME_STEP
        )
        max_step[self.info.hand_dofs] = (
            self.mapping.max_gripper_joint_speed_rad_s * task.TIME_STEP
        )
        last_error_s = {"left": -np.inf, "right": -np.inf}
        reset_for_stale_packet = False
        packet_count = 0
        solved_hand_count = 0
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
                    packet_count += 1
                    solved_hand_count += self._solve_packet(
                        packet, desired, last_error_s
                    )
                    reset_for_stale_packet = False
                latest = with_receiver.latest
                if (
                    latest is not None
                    and now - float(latest["capture_timestamp_s"])
                    > self.mapping.reset_age_s
                    and not reset_for_stale_packet
                ):
                    current = runner.measured_pose()
                    desired[controlled] = current[controlled]
                    reset_for_stale_packet = True
                delta = np.clip(desired - current, -max_step, max_step)
                current[controlled] += delta[controlled]
                if not runner.executor.step(current):
                    return False
        except KeyboardInterrupt:
            return False
        finally:
            with_receiver.close()
            print(
                f"  teleop summary: {packet_count} fresh packets, "
                f"{solved_hand_count} accepted hand targets, "
                f"{with_receiver.invalid_packets} invalid packets ignored"
            )
        return True


__all__ = ["TeleopPolicy"]
