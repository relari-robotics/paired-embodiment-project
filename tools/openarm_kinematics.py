#!/usr/bin/env python
"""OpenArm v2 kinematics helpers for retargeting work.

``export`` writes the robot's joint tree (link parents, joint frames, axes,
limits, render models) to JSON so forward kinematics can be implemented
without the simulator.  ``fk`` evaluates the simulator's own kinematics for a
trajectory file and prints the grasp-point pose of each commanded arm at a few
sample times, which is a quick sanity check of a retargeted file before any
physics runs.  ``ik`` solves one grasp-point pose and prints the joint vector.

    python tools/openarm_kinematics.py export --output openarm_v2_kinematics.json
    python tools/openarm_kinematics.py fk trajectory.json --every 1.0
    python tools/openarm_kinematics.py ik --side right --position 0.0 -0.23 0.42 \\
        --quaternion -0.012 -0.685 -0.012 0.728

All poses are metres and (x, y, z, w) quaternions in the simulation world
frame (X forward from the robot, Y left, Z up).  The grasp point is 0.155 m
along the gripper's finger axis from the wrist flange (``GRASP_POINT_EE``) and
shares the flange orientation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def _transform_dict(transform) -> dict[str, list[float]]:
    rotation = transform.rotation
    return {
        "translation_m": np.round(np.asarray(transform.translation, dtype=float), 7).tolist(),
        "quaternion_xyzw": [round(float(rotation[i]), 7) for i in range(4)],
    }


def _build(context, sides):
    from scenarios.ball_bowl.collision import BallBowlCollisionModel
    from scenarios.ball_bowl.scenario import (
        DESK_MIN,
        DESK_SIZE,
        ScenarioSpecification,
        contact_params,
    )
    from superdex import physics
    from superdex_scenarios.embodiments.openarm_v2 import OpenArmKinematics, build_openarm_v2

    scene = physics.create_scene("openarm kinematics tool")
    info = build_openarm_v2(scene, context, contact_params, sides=sides)
    collision = BallBowlCollisionModel(
        ScenarioSpecification.fixed(), DESK_MIN, DESK_SIZE, avoid_bowl=False
    )
    twins = {
        side: OpenArmKinematics(context, info, contact_params, collision, side=side) for side in sides
    }
    return scene, info, twins


def command_export(args: argparse.Namespace) -> None:
    from superdex import physics, robotics
    from superdex.physics.paths import resolve_asset
    from superdex_scenarios.embodiments import openarm_v2 as oa

    physics.initialize(num_worker_threads=0)
    context = robotics.create_context()
    scene, info, twins = _build(context, ("right", "left"))
    assets_root = Path(resolve_asset(oa.ROBOT_ASSET)).resolve()
    while assets_root.name != "assets" and assets_root.parent != assets_root:
        assets_root = assets_root.parent
    dof = 0
    links = []
    for index, (link, joint) in enumerate(zip(info.prefab.links, info.prefab.joints)):
        entry = {
            "index": index,
            "name": link.name,
            "parent_index": int(link.parent_link),
            "parent_name": info.prefab.links[int(link.parent_link)].name if int(link.parent_link) >= 0 else None,
            "parent_link_from_joint": _transform_dict(joint.parent_link_from_joint),
            "parent_joint_from_link": _transform_dict(link.parent_joint_from_link),
            "joint": {
                "name": joint.name,
                "type": str(joint.type).split(".")[-1].lower(),
            },
            "mass_kg": float(link.mass) if link.mass is not None else None,
            "render_model": (
                str(Path(link.render_model_file).resolve().relative_to(assets_root))
                if link.render_model_file
                else None
            ),
        }
        if joint.type == physics.ArticulatedJointType.REVOLUTE:
            axis = np.asarray(joint.axis, dtype=float)
            lo = float(np.dot(np.asarray(joint.min_limit, dtype=float), axis))
            hi = float(np.dot(np.asarray(joint.max_limit, dtype=float), axis))
            entry["joint"].update(
                {
                    "dof_index": dof,
                    "axis_local": axis.tolist(),
                    "limits_rad": sorted((lo, hi)),
                    "effort_limit_nm": float(joint.effort_limit),
                }
            )
            dof += 1
        links.append(entry)
    right_home = twins["right"].grasp_point_pose(info.default_pose[info.arm_dofs])
    left_home = twins["left"].grasp_point_pose(info.default_pose[info.arm_dofs])
    payload = {
        "format": "openarm-v2-kinematics-v1",
        "notes": [
            "Joint i connects links[i].parent_name to links[i].name. The joint frame is",
            "parent_link_from_joint in the parent link frame; the child link frame is",
            "parent_joint_from_link in the joint frame after rotating by the joint angle",
            "about axis_local (right-hand rule). dof_index is the position in the actor",
            "pose vector; trajectory files address joints by name through dof_groups.",
            "World frame: X forward from the robot, Y left, Z up; metres and radians.",
        ],
        "asset": oa.ROBOT_ASSET,
        "world_from_root": {"translation_m": list(map(float, oa.ROBOT_ROOT_POSITION)), "quaternion_xyzw": [0, 0, 0, 1]},
        "dof_names": list(info.dof_names),
        "dof_groups": {key: value.tolist() for key, value in info.dof_groups.items()},
        "default_pose_rad": np.round(info.default_pose, 7).tolist(),
        "grasp_point_in_ee_frame_m": oa.GRASP_POINT_EE.tolist(),
        "end_effector_links": {"right": oa.ee_link("right"), "left": oa.ee_link("left")},
        "finger_links": {"right": list(oa.finger_links("right")), "left": list(oa.finger_links("left"))},
        "gripper": {
            "open_joint_magnitude_rad": oa.FINGER_JOINT_OPEN_MAGNITUDE,
            "open_sign": {"right": oa.finger_sign("right"), "left": oa.finger_sign("left")},
            "aperture_to_joint": "joint = open_sign[side] * open_joint_magnitude_rad * aperture, aperture in [0, 1]",
            "reference_apertures": {
                "fully_open": 1.0,
                "parked_home": abs(oa.RIGHT_FINGERS_HOME) / oa.FINGER_JOINT_OPEN_MAGNITUDE,
                "holding_67mm_ball": abs(oa.RIGHT_FINGERS_CLOSED) / oa.FINGER_JOINT_OPEN_MAGNITUDE,
                "pinching_thin_rim": 0.04 / oa.FINGER_JOINT_OPEN_MAGNITUDE,
            },
        },
        "level_gripper_orientation_xyzw": {
            side: [round(float(twin.level_rotation[i]), 7) for i in range(4)] for side, twin in twins.items()
        },
        "home_grasp_point_pose": {
            "right": {"position_m": np.round(right_home[0], 5).tolist(), "quaternion_xyzw": np.round(right_home[1], 6).tolist()},
            "left": {"position_m": np.round(left_home[0], 5).tolist(), "quaternion_xyzw": np.round(left_home[1], 6).tolist()},
        },
        "links": links,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output} ({len(links)} links, {dof} DOFs)")
    for twin in twins.values():
        twin.close()
    physics.destroy_scene(scene)
    physics.shutdown()


def command_fk(args: argparse.Namespace) -> None:
    from superdex import physics, robotics
    from superdex_scenarios.replay.trajectory import JointTrajectory

    trajectory = JointTrajectory.load(args.trajectory).rebased()
    physics.initialize(num_worker_threads=0)
    context = robotics.create_context()
    scene, info, twins = _build(context, trajectory.sides if "right" in trajectory.sides else ("right", *trajectory.sides))
    sample_times = np.arange(0.0, trajectory.duration_s + 1e-9, args.every)
    poses = np.tile(info.default_pose, (len(sample_times), 1))
    for side, arm in trajectory.arms.items():
        poses[:, info.dof_groups[f"{side}_arm"]] = np.column_stack(
            [np.interp(sample_times, trajectory.time_s, arm[:, d]) for d in range(7)]
        )
    print(f"{'t [s]':>7} side {'grasp point x y z [m]':>28}  {'quaternion x y z w':>34}")
    for t, pose in zip(sample_times, poses):
        for side in trajectory.sides:
            position, quaternion = twins[side].grasp_point_pose(pose[info.arm_dofs])
            print(
                f"{t:7.2f} {side:5s} {np.round(position, 4).tolist()!s:>28}  {np.round(quaternion, 4).tolist()!s:>34}"
            )
    for twin in twins.values():
        twin.close()
    physics.destroy_scene(scene)
    physics.shutdown()


def command_ik(args: argparse.Namespace) -> None:
    from superdex import physics, robotics

    physics.initialize(num_worker_threads=0)
    context = robotics.create_context()
    scene, info, twins = _build(context, ("right", "left"))
    twin = twins[args.side]
    seed = info.default_pose[info.arm_dofs]
    if args.seed is not None:
        seed = seed.copy()
        seed[np.isin(info.arm_dofs, info.dof_groups[f"{args.side}_arm"])] = np.asarray(args.seed, dtype=float)
    try:
        if args.quaternion is None:
            arm = twin.solve(args.position, seed)
        else:
            arm = twin.solve_pose(args.position, args.quaternion, seed)
        joints = arm[np.isin(info.arm_dofs, info.dof_groups[f"{args.side}_arm"])]
        position, quaternion = twin.grasp_point_pose(arm)
        print(json.dumps({
            "side": args.side,
            f"{args.side}_arm_rad": np.round(joints, 6).tolist(),
            "achieved_position_m": np.round(position, 5).tolist(),
            "achieved_quaternion_xyzw": np.round(quaternion, 6).tolist(),
        }, indent=2))
    except RuntimeError as error:
        raise SystemExit(f"IK failed: {error}")
    finally:
        for t in twins.values():
            t.close()
        physics.destroy_scene(scene)
        physics.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export", help="write the joint tree to JSON")
    export.add_argument("--output", default="openarm_v2_kinematics.json")
    export.set_defaults(func=command_export)
    fk = sub.add_parser("fk", help="print grasp-point poses along a trajectory file")
    fk.add_argument("trajectory")
    fk.add_argument("--every", type=float, default=0.5, help="sample spacing in seconds")
    fk.set_defaults(func=command_fk)
    ik = sub.add_parser("ik", help="solve one grasp-point pose")
    ik.add_argument("--side", choices=("right", "left"), default="right")
    ik.add_argument("--position", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    ik.add_argument("--quaternion", type=float, nargs=4, default=None, metavar=("QX", "QY", "QZ", "QW"),
                    help="world orientation of the grasp-point frame (default: the level gripper)")
    ik.add_argument("--seed", type=float, nargs=7, default=None, help="seed joint vector for this arm")
    ik.set_defaults(func=command_ik)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
