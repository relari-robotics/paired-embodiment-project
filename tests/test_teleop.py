import json
import select
import socket
from types import SimpleNamespace

import numpy as np
import pytest

from scenarios.ball_bowl.teleop_policy import TeleopPolicy
from superdex_scenarios.embodiments.openarm_v2 import finger_joint_from_aperture
from superdex_scenarios.teleop import (
    ArmMapping,
    TeleopReceiver,
    decode_packet,
    parse_endpoint,
)


def _packet(sequence=1):
    return {
        "format": "kyber-teleop-v1",
        "sequence": sequence,
        "capture_timestamp_s": 10.0,
        "profile_sha256": "a" * 64,
        "hands": [
            {
                "side": "right",
                "confidence": 0.9,
                "palm_position_desk_m": [0.5, 0.25, 0.0],
                "palm_quaternion_xyzw": [0, 0, 0, 1],
                "aperture": 0.4,
                "depth_support": 15,
                "joints_desk_m": [[0, 0, 0]] * 21,
            }
        ],
    }


def test_decode_packet_checks_profile():
    payload = json.dumps(_packet()).encode()
    assert decode_packet(payload, "a" * 64)["sequence"] == 1
    with pytest.raises(ValueError, match="profiles do not match"):
        decode_packet(payload, "b" * 64)
    with pytest.raises(TypeError, match="JSON object"):
        decode_packet(b"[]", "a" * 64)


def test_affine_mapping_clamps_and_maps_orientation():
    mapping = ArmMapping.from_dict(
        {
            "source_min_desk_m": [0, 0, 0],
            "source_max_desk_m": [1, 1, 1],
            "target_min_world_m": [-1, -2, 3],
            "target_max_world_m": [1, 2, 5],
            "palm_to_gripper_quaternion_xyzw": [0, 0, 0, 1],
        }
    )
    hand = _packet()["hands"][0]
    hand["palm_position_desk_m"] = [0.5, -1.0, 2.0]
    result = mapping.map(hand)
    np.testing.assert_allclose(result.position_world_m, [0, -2, 5])
    np.testing.assert_allclose(result.quaternion_world_xyzw, [0, 0, 0, 1])
    assert result.aperture == pytest.approx(0.4)


def test_policy_falls_back_to_level_gripper_when_palm_orientation_is_infeasible():
    calls = []

    class Kinematics:
        def solve_pose_optimized(self, position, quaternion, seed):
            calls.append("pose")
            raise RuntimeError("orientation exceeds a joint limit")

        def solve_optimized(self, position, seed):
            calls.append("level")
            return np.array([0.1, 0.2])

        def collision_cost(self, candidate):
            return 0.0, 0.1

    mapped = SimpleNamespace(
        position_world_m=np.array([0.1, -0.1, 0.5]),
        quaternion_world_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        aperture=0.4,
    )
    policy = TeleopPolicy.__new__(TeleopPolicy)
    policy.info = SimpleNamespace(
        arm_dofs=np.array([0, 1]),
        dof_groups={"right_gripper": np.array([2, 3])},
    )
    policy.mapping = SimpleNamespace(map_hand=lambda hand: mapped)
    policy.kinematics = {"right": Kinematics()}
    desired = np.zeros(4)

    solved = policy._solve_packet(
        {"hands": [{"side": "right"}]},
        desired,
        {"right": -np.inf},
    )

    assert solved == 1
    assert calls == ["pose", "level"]
    np.testing.assert_allclose(desired[:2], [0.1, 0.2])
    np.testing.assert_allclose(
        desired[2:], finger_joint_from_aperture(mapped.aperture, "right")
    )


def test_endpoint_validation():
    assert parse_endpoint("127.0.0.1:7447") == ("127.0.0.1", 7447)
    with pytest.raises(ValueError, match="HOST:PORT"):
        parse_endpoint("7447")


def test_receiver_keeps_latest_sequence_and_expires_stale_packet():
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    receiver = TeleopReceiver(f"127.0.0.1:{port}", "a" * 64, max_age_s=0.1)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        newest = _packet(sequence=2)
        newest["capture_timestamp_s"] = 20.0
        older = _packet(sequence=1)
        older["capture_timestamp_s"] = 20.0
        sender.sendto(json.dumps(newest).encode(), ("127.0.0.1", port))
        sender.sendto(json.dumps(older).encode(), ("127.0.0.1", port))
        assert select.select([receiver.socket], [], [], 0.2)[0]
        packet, updated = receiver.poll(20.01)
        assert updated
        assert packet["sequence"] == 2
        assert receiver.poll(20.2)[0] is None
    finally:
        sender.close()
        receiver.close()


def test_receiver_ignores_malformed_packet_and_keeps_running():
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    receiver = TeleopReceiver(f"127.0.0.1:{port}", "a" * 64)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sender.sendto(b"not-json", ("127.0.0.1", port))
        sender.sendto(json.dumps(_packet()).encode(), ("127.0.0.1", port))
        assert select.select([receiver.socket], [], [], 0.2)[0]
        packet, updated = receiver.poll(10.01)
        assert updated
        assert packet is not None
        assert packet["sequence"] == 1
        assert receiver.invalid_packets == 1
        assert receiver.last_packet_error
    finally:
        sender.close()
        receiver.close()
