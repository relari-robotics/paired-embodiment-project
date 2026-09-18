"""Dependency-light live teleoperation transport and workspace mapping."""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

FORMAT = "kyber-teleop-v1"
MAPPING_FORMAT = "superdex-teleop-mapping-v1"


def parse_endpoint(value: str) -> tuple[str, int]:
    host, separator, port = value.rpartition(":")
    if not separator or not host:
        raise ValueError("endpoint must have the form HOST:PORT")
    port_number = int(port)
    if not 1 <= port_number <= 65535:
        raise ValueError("endpoint port must be in [1, 65535]")
    return host, port_number


def _vector(value, size: int, name: str) -> npt.NDArray[np.float64]:
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite {size}-vector")
    return array


def _quaternion_multiply(left, right) -> npt.NDArray[np.float64]:
    x1, y1, z1, w1 = _vector(left, 4, "left quaternion")
    x2, y2, z2, w2 = _vector(right, 4, "right quaternion")
    result = np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )
    norm = float(np.linalg.norm(result))
    if norm < 1e-8:
        raise ValueError("quaternion product is degenerate")
    result /= norm
    return -result if result[3] < 0 else result


@dataclass(frozen=True)
class MappedHand:
    side: str
    position_world_m: npt.NDArray[np.float64]
    quaternion_world_xyzw: npt.NDArray[np.float64]
    aperture: float


@dataclass(frozen=True)
class ArmMapping:
    source_min_desk_m: npt.NDArray[np.float64]
    source_max_desk_m: npt.NDArray[np.float64]
    target_min_world_m: npt.NDArray[np.float64]
    target_max_world_m: npt.NDArray[np.float64]
    palm_to_gripper_quaternion_xyzw: npt.NDArray[np.float64]

    @classmethod
    def from_dict(cls, payload: dict) -> ArmMapping:
        result = cls(
            _vector(payload["source_min_desk_m"], 3, "source_min_desk_m"),
            _vector(payload["source_max_desk_m"], 3, "source_max_desk_m"),
            _vector(payload["target_min_world_m"], 3, "target_min_world_m"),
            _vector(payload["target_max_world_m"], 3, "target_max_world_m"),
            _vector(
                payload["palm_to_gripper_quaternion_xyzw"],
                4,
                "palm_to_gripper_quaternion_xyzw",
            ),
        )
        if np.any(result.source_max_desk_m <= result.source_min_desk_m):
            raise ValueError("source workspace bounds must be strictly increasing")
        if np.any(result.target_max_world_m <= result.target_min_world_m):
            raise ValueError("target workspace bounds must be strictly increasing")
        if np.linalg.norm(result.palm_to_gripper_quaternion_xyzw) < 1e-8:
            raise ValueError("palm-to-gripper quaternion must be nonzero")
        return result

    def map(self, hand: dict) -> MappedHand:
        source = _vector(hand["palm_position_desk_m"], 3, "palm position")
        fraction = np.clip(
            (source - self.source_min_desk_m)
            / (self.source_max_desk_m - self.source_min_desk_m),
            0.0,
            1.0,
        )
        position = self.target_min_world_m + fraction * (
            self.target_max_world_m - self.target_min_world_m
        )
        palm = _vector(hand["palm_quaternion_xyzw"], 4, "palm quaternion")
        palm /= np.linalg.norm(palm)
        orientation = _quaternion_multiply(palm, self.palm_to_gripper_quaternion_xyzw)
        aperture = float(hand["aperture"])
        if not np.isfinite(aperture):
            raise ValueError("hand aperture must be finite")
        return MappedHand(
            str(hand["side"]), position, orientation, float(np.clip(aperture, 0.0, 1.0))
        )


@dataclass(frozen=True)
class TeleopMapping:
    arms: dict[str, ArmMapping]
    max_age_s: float
    reset_age_s: float
    max_arm_joint_speed_rad_s: float
    max_gripper_joint_speed_rad_s: float
    ik_max_iterations: int = 14
    orientation_mode: str = "adaptive"
    target_filter_time_constant_s: float = 0.08
    joint_tracking_time_constant_s: float = 0.12
    max_arm_joint_acceleration_rad_s2: float = 8.0
    max_gripper_joint_acceleration_rad_s2: float = 16.0

    @classmethod
    def load(cls, path: Path) -> TeleopMapping:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != MAPPING_FORMAT:
            raise ValueError("unsupported teleoperation mapping format")
        arms = {
            side: ArmMapping.from_dict(payload["arms"][side])
            for side in ("left", "right")
        }
        result = cls(
            arms=arms,
            max_age_s=float(payload.get("max_age_s", 0.1)),
            reset_age_s=float(payload.get("reset_age_s", 0.5)),
            max_arm_joint_speed_rad_s=float(
                payload.get("max_arm_joint_speed_rad_s", 1.5)
            ),
            max_gripper_joint_speed_rad_s=float(
                payload.get("max_gripper_joint_speed_rad_s", 2.0)
            ),
            ik_max_iterations=int(payload.get("ik_max_iterations", 14)),
            orientation_mode=str(payload.get("orientation_mode", "adaptive")),
            target_filter_time_constant_s=float(
                payload.get("target_filter_time_constant_s", 0.08)
            ),
            joint_tracking_time_constant_s=float(
                payload.get("joint_tracking_time_constant_s", 0.12)
            ),
            max_arm_joint_acceleration_rad_s2=float(
                payload.get("max_arm_joint_acceleration_rad_s2", 8.0)
            ),
            max_gripper_joint_acceleration_rad_s2=float(
                payload.get("max_gripper_joint_acceleration_rad_s2", 16.0)
            ),
        )
        if result.ik_max_iterations < 1:
            raise ValueError("teleop ik_max_iterations must be at least 1")
        if result.orientation_mode not in {"adaptive", "table-parallel", "top-down"}:
            raise ValueError(
                "teleop orientation_mode must be 'adaptive', 'table-parallel', or 'top-down'"
            )
        if not 0 < result.max_age_s < result.reset_age_s:
            raise ValueError("teleop reset age must exceed positive maximum packet age")
        if (
            result.max_arm_joint_speed_rad_s <= 0
            or result.max_gripper_joint_speed_rad_s <= 0
        ):
            raise ValueError("teleop joint speed limits must be positive")
        if (
            result.target_filter_time_constant_s <= 0
            or result.joint_tracking_time_constant_s <= 0
        ):
            raise ValueError("teleop smoothing time constants must be positive")
        if (
            result.max_arm_joint_acceleration_rad_s2 <= 0
            or result.max_gripper_joint_acceleration_rad_s2 <= 0
        ):
            raise ValueError(
                "teleop joint acceleration limits must be positive"
            )
        return result

    def map_hand(self, hand: dict) -> MappedHand:
        side = str(hand.get("side"))
        if side not in self.arms:
            raise ValueError("teleop packet contains an unknown hand side")
        return self.arms[side].map(hand)


def decode_packet(payload: bytes, expected_profile_sha256: str) -> dict:
    packet = json.loads(payload.decode("utf-8"))
    if not isinstance(packet, dict):
        raise TypeError("teleop packet must be a JSON object")
    if packet.get("format") != FORMAT:
        raise ValueError("unsupported Kyber teleop packet")
    if packet.get("profile_sha256") != expected_profile_sha256:
        raise ValueError("Kyber and SuperDex desk calibration profiles do not match")
    if not isinstance(packet.get("sequence"), int) or packet["sequence"] < 0:
        raise ValueError("teleop sequence must be a non-negative integer")
    timestamp = float(packet["capture_timestamp_s"])
    if not np.isfinite(timestamp) or timestamp < 0:
        raise ValueError("teleop capture timestamp is invalid")
    hands = packet.get("hands")
    if not isinstance(hands, list) or len(hands) > 2:
        raise ValueError("teleop packet must contain at most two hands")
    if any(not isinstance(hand, dict) for hand in hands):
        raise TypeError("each teleop hand must be a JSON object")
    sides = [hand.get("side") for hand in hands]
    if any(side not in {"left", "right"} for side in sides) or len(set(sides)) != len(
        sides
    ):
        raise ValueError("teleop packet contains duplicate or invalid hand sides")
    for hand in hands:
        _vector(hand["palm_position_desk_m"], 3, "palm position")
        quaternion = _vector(hand["palm_quaternion_xyzw"], 4, "palm quaternion")
        if np.linalg.norm(quaternion) < 1e-8:
            raise ValueError("teleop palm quaternion must be nonzero")
        aperture = float(hand["aperture"])
        confidence = float(hand["confidence"])
        if not 0 <= aperture <= 1 or not 0 <= confidence <= 1:
            raise ValueError("teleop aperture and confidence must be in [0, 1]")
    return packet


class TeleopReceiver:
    """Non-blocking latest-packet receiver with sequence and age rejection."""

    def __init__(
        self,
        endpoint: str,
        expected_profile_sha256: str,
        *,
        max_age_s: float = 0.1,
    ) -> None:
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setblocking(False)
        self.socket.bind(parse_endpoint(endpoint))
        self.expected_profile_sha256 = expected_profile_sha256
        self.max_age_s = float(max_age_s)
        self.last_sequence = -1
        self.latest: dict | None = None
        self.invalid_packets = 0
        self.last_packet_error: str | None = None

    def poll(self, now_s: float) -> tuple[dict | None, bool]:
        updated = False
        while True:
            try:
                payload, _ = self.socket.recvfrom(65535)
            except BlockingIOError:
                break
            try:
                packet = decode_packet(payload, self.expected_profile_sha256)
            except (
                KeyError,
                OverflowError,
                TypeError,
                UnicodeDecodeError,
                ValueError,
            ) as error:
                self.invalid_packets += 1
                self.last_packet_error = str(error)
                continue
            if packet["sequence"] <= self.last_sequence:
                continue
            if float(packet["capture_timestamp_s"]) > now_s + 0.25:
                continue
            self.last_sequence = packet["sequence"]
            self.latest = packet
            updated = True
        if self.latest is None:
            return None, updated
        if now_s - float(self.latest["capture_timestamp_s"]) > self.max_age_s:
            return None, updated
        return self.latest, updated

    def close(self) -> None:
        self.socket.close()
