import json
import select
import socket
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from scenarios.ball_bowl.teleop_policy import (
    TeleopPolicy,
    _LatestPacketIK,
    table_parallel_quaternion_xyzw,
    top_down_quaternion_xyzw,
)
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


def test_policy_falls_back_through_top_down_to_level_when_the_pose_is_infeasible():
    calls = []

    class Kinematics:
        def solve_pose_optimized(self, position, quaternion, seed, *, max_iterations=None):
            calls.append(("pose", tuple(np.round(quaternion, 6)), max_iterations))
            raise RuntimeError("orientation exceeds a joint limit")

        def solve_optimized(self, position, seed, *, max_iterations=None):
            calls.append(("level", None, max_iterations))
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
    policy.mapping = SimpleNamespace(map_hand=lambda hand: mapped, ik_max_iterations=6)
    policy.kinematics = {"right": Kinematics()}
    desired = np.zeros(4)
    accepted: dict[str, int] = {}

    solved = policy._solve_packet(
        {"hands": [{"side": "right"}]},
        desired,
        {"right": -np.inf},
        accepted,
    )

    assert solved == 1
    assert accepted == {"level": 1}
    top_down = tuple(np.round(top_down_quaternion_xyzw([0, 0, 0, 1], "right"), 6))
    assert calls == [
        ("pose", (0.0, 0.0, 0.0, 1.0), 6),
        ("pose", top_down, 6),
        ("level", None, 6),
    ]
    np.testing.assert_allclose(desired[:2], [0.1, 0.2])
    np.testing.assert_allclose(
        desired[2:], finger_joint_from_aperture(mapped.aperture, "right")
    )


def test_policy_top_down_mode_does_not_switch_orientation_fallbacks():
    calls = []

    class Kinematics:
        def solve_pose_optimized(self, position, quaternion, seed, *, max_iterations=None):
            calls.append((tuple(np.round(quaternion, 6)), max_iterations))
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
    policy.mapping = SimpleNamespace(
        map_hand=lambda hand: mapped,
        ik_max_iterations=5,
        orientation_mode="top-down",
    )
    policy.kinematics = {"right": Kinematics()}
    desired = np.zeros(4)
    accepted = {}

    assert policy._solve_packet(
        {"hands": [{"side": "right"}]}, desired, {"right": -np.inf}, accepted
    ) == 1
    expected = tuple(np.round(top_down_quaternion_xyzw([0, 0, 0, 1], "right"), 6))
    assert calls == [(expected, 5)]
    assert accepted == {"top-down": 1}


def test_policy_table_parallel_mode_does_not_switch_orientation_fallbacks():
    from scipy.spatial.transform import Rotation

    calls = []

    class Kinematics:
        def solve_pose_optimized(self, position, quaternion, seed, *, max_iterations=None):
            calls.append((tuple(np.round(quaternion, 6)), max_iterations))
            return np.array([0.1, 0.2])

        def collision_cost(self, candidate):
            return 0.0, 0.1

    quaternion = Rotation.from_euler("zyx", [35, 20, -10], degrees=True).as_quat()
    mapped = SimpleNamespace(
        position_world_m=np.array([0.1, -0.1, 0.5]),
        quaternion_world_xyzw=quaternion,
        aperture=0.4,
    )
    policy = TeleopPolicy.__new__(TeleopPolicy)
    policy.info = SimpleNamespace(
        arm_dofs=np.array([0, 1]),
        dof_groups={"right_gripper": np.array([2, 3])},
    )
    policy.mapping = SimpleNamespace(
        map_hand=lambda hand: mapped,
        ik_max_iterations=5,
        orientation_mode="table-parallel",
    )
    policy.kinematics = {"right": Kinematics()}
    desired = np.zeros(4)
    accepted = {}

    assert policy._solve_packet(
        {"hands": [{"side": "right"}]}, desired, {"right": -np.inf}, accepted
    ) == 1
    expected = tuple(np.round(table_parallel_quaternion_xyzw(quaternion), 6))
    assert calls == [(expected, 5)]
    assert accepted == {"table-parallel": 1}


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


def test_top_down_quaternion_points_fingers_down_and_keeps_jaw_yaw():
    from scipy.spatial.transform import Rotation

    # Mapped pose: gripper horizontal, jaw axis rotated 30 degrees about Z.
    mapped = Rotation.from_euler("z", 30, degrees=True) * Rotation.from_euler("y", 90, degrees=True)
    result = Rotation.from_quat(top_down_quaternion_xyzw(mapped.as_quat(), "right")).as_matrix()
    np.testing.assert_allclose(result[:, 2], [0.0, 0.0, 1.0], atol=1e-12)  # local +Z up: fingers down
    jaw = mapped.as_matrix()[:, 1]
    jaw[2] = 0.0
    np.testing.assert_allclose(result[:, 1], jaw / np.linalg.norm(jaw), atol=1e-12)
    assert np.linalg.det(result) == pytest.approx(1.0)

    vertical_jaw = Rotation.from_euler("x", 90, degrees=True)  # jaw axis along Z: degenerate
    for side, sign in (("left", 1.0), ("right", -1.0)):
        result = Rotation.from_quat(top_down_quaternion_xyzw(vertical_jaw.as_quat(), side)).as_matrix()
        np.testing.assert_allclose(result[:, 1], [0.0, sign, 0.0], atol=1e-12)


def test_table_parallel_quaternion_keeps_gripper_level_and_human_heading():
    from scipy.spatial.transform import Rotation

    mapped = Rotation.from_euler("zyx", [35, 25, -15], degrees=True).as_matrix()
    result = Rotation.from_quat(
        table_parallel_quaternion_xyzw(Rotation.from_matrix(mapped).as_quat())
    ).as_matrix()

    # Local Y is the jaw axis and local -Z is finger-forward. Both are
    # horizontal, so the whole finger/jaw plane is parallel to the desk.
    assert result[2, 0] == pytest.approx(1.0)
    assert result[2, 1] == pytest.approx(0.0, abs=1e-12)
    assert result[2, 2] == pytest.approx(0.0, abs=1e-12)
    mapped_forward = -mapped[:, 2]
    mapped_forward[2] = 0.0
    mapped_forward /= np.linalg.norm(mapped_forward)
    np.testing.assert_allclose(-result[:, 2], mapped_forward, atol=1e-12)
    assert np.linalg.det(result) == pytest.approx(1.0)

    # If finger-forward is vertical, the horizontal jaw axis provides the
    # otherwise undefined tabletop heading.
    singular = Rotation.from_quat(
        table_parallel_quaternion_xyzw([0.0, 0.0, 0.0, 1.0])
    ).as_matrix()
    np.testing.assert_allclose(-singular[:, 2], [1.0, 0.0, 0.0], atol=1e-12)
    assert singular[2, 0] == pytest.approx(1.0)

    # Palm-normal noise must never select the equivalent upside-down frame;
    # that choice would introduce an abrupt 180-degree wrist flip.
    palm_down = mapped.copy()
    palm_down[:, 0] *= -1.0
    palm_down[:, 1] *= -1.0
    stable = Rotation.from_quat(
        table_parallel_quaternion_xyzw(Rotation.from_matrix(palm_down).as_quat())
    ).as_matrix()
    assert stable[2, 0] == pytest.approx(1.0)


def test_mapping_reads_ik_iteration_budget(tmp_path):
    from superdex_scenarios.teleop import TeleopMapping

    payload = {
        "format": "superdex-teleop-mapping-v1",
        "ik_max_iterations": 6,
        "orientation_mode": "table-parallel",
        "target_filter_time_constant_s": 0.06,
        "joint_tracking_time_constant_s": 0.10,
        "max_arm_joint_acceleration_rad_s2": 7.0,
        "max_gripper_joint_acceleration_rad_s2": 14.0,
        "arms": {
            side: {
                "source_min_desk_m": [0, 0, 0],
                "source_max_desk_m": [1, 1, 1],
                "target_min_world_m": [0, 0, 0],
                "target_max_world_m": [1, 1, 1],
                "palm_to_gripper_quaternion_xyzw": [0, 0, 0, 1],
            }
            for side in ("left", "right")
        },
    }
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(payload))
    mapping = TeleopMapping.load(path)
    assert mapping.ik_max_iterations == 6
    assert mapping.orientation_mode == "table-parallel"
    assert mapping.target_filter_time_constant_s == pytest.approx(0.06)
    assert mapping.joint_tracking_time_constant_s == pytest.approx(0.10)
    assert mapping.max_arm_joint_acceleration_rad_s2 == pytest.approx(7.0)
    assert mapping.max_gripper_joint_acceleration_rad_s2 == pytest.approx(14.0)
    payload["ik_max_iterations"] = 0
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="ik_max_iterations"):
        TeleopMapping.load(path)
    payload["ik_max_iterations"] = 6
    payload["orientation_mode"] = "sideways"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="orientation_mode"):
        TeleopMapping.load(path)
    payload["orientation_mode"] = "table-parallel"
    payload["target_filter_time_constant_s"] = 0
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="smoothing time constants"):
        TeleopMapping.load(path)


def test_policy_catches_physics_up_with_smooth_acceleration(monkeypatch):
    class Executor:
        real_time = True
        wall_start = 10.0
        step_count = 0

        def __init__(self):
            self.poses = []

        def step(self, pose):
            self.poses.append(np.asarray(pose).copy())
            self.step_count += 1
            return True

    executor = Executor()
    runner = SimpleNamespace(executor=executor)
    current = np.zeros(2)
    desired = np.ones(2)
    filtered = current.copy()
    velocity = np.zeros(2)
    controlled = np.array([0, 1])
    max_velocity = np.full(2, 1.0)
    max_acceleration = np.full(2, 2.0)
    monkeypatch.setattr("scenarios.ball_bowl.teleop_policy.time.perf_counter", lambda: 10.011)

    assert TeleopPolicy._advance_physics(
        runner,
        current,
        desired,
        filtered,
        velocity,
        controlled,
        max_velocity,
        max_acceleration,
        0.08,
        0.12,
    )

    assert executor.step_count == 5
    assert np.all(current > 0)
    poses = np.vstack((np.zeros(2), executor.poses))
    velocities = np.diff(poses, axis=0) / 0.0025
    accelerations = np.diff(np.vstack((np.zeros(2), velocities)), axis=0) / 0.0025
    assert np.max(np.abs(velocities)) <= 1.0 + 1e-12
    assert np.max(np.abs(accelerations)) <= 2.0 + 1e-12
    np.testing.assert_allclose(executor.poses[-1], current)


def test_policy_retarget_preserves_velocity_continuity():
    class Executor:
        real_time = False
        step_count = 0

        def __init__(self):
            self.poses = []

        def step(self, pose):
            self.poses.append(float(pose[0]))
            self.step_count += 1
            return True

    executor = Executor()
    runner = SimpleNamespace(executor=executor)
    current = np.zeros(1)
    desired = np.ones(1)
    filtered = current.copy()
    velocity = np.zeros(1)
    args = (
        runner,
        current,
        desired,
        filtered,
        velocity,
        np.array([0]),
        np.array([0.6]),
        np.array([2.0]),
        0.08,
        0.12,
    )
    for _ in range(120):
        assert TeleopPolicy._advance_physics(*args)

    velocity_before_retarget = float(velocity[0])
    desired[:] = -1.0
    assert TeleopPolicy._advance_physics(*args)
    assert velocity[0] > 0.0
    assert velocity_before_retarget - velocity[0] <= 2.0 * 0.0025 + 1e-12
    for _ in range(399):
        assert TeleopPolicy._advance_physics(*args)

    positions = np.asarray(executor.poses)
    velocities = np.diff(np.concatenate(([0.0], positions))) / 0.0025
    accelerations = np.diff(np.concatenate(([0.0], velocities))) / 0.0025
    assert np.max(np.abs(velocities)) <= 0.6 + 1e-12
    assert np.max(np.abs(accelerations)) <= 2.0 + 1e-10
    assert positions[-1] < positions[120]


def test_async_ik_keeps_only_the_latest_packet_without_blocking():
    first_started = threading.Event()
    release_first = threading.Event()
    seen = []

    def solve(packet, candidate):
        sequence = packet["sequence"]
        seen.append(sequence)
        if sequence == 1:
            first_started.set()
            assert release_first.wait(timeout=1.0)
        candidate[:] = sequence
        return 1, {"top-down": 1}

    desired = np.zeros(2)
    worker = _LatestPacketIK(solve)
    solved_total = 0
    started_total = 0
    try:
        worker.offer({"sequence": 1})
        solved, _, started = worker.update(desired)
        solved_total += solved
        started_total += started
        assert started == 1
        assert first_started.wait(timeout=1.0)

        worker.offer({"sequence": 2})
        worker.offer({"sequence": 3})
        assert worker.update(desired) == (0, {}, 0)
        release_first.set()

        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and solved_total < 2:
            solved, _, started = worker.update(desired)
            solved_total += solved
            started_total += started
            if solved_total < 2:
                time.sleep(0.001)
    finally:
        solved, _ = worker.close(desired)
        solved_total += solved

    assert seen == [1, 3]
    assert started_total == 2
    assert solved_total == 2
    np.testing.assert_allclose(desired, [3.0, 3.0])
