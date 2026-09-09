"""Full OpenArm v2 model with right-arm-only control and kinematics."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
from superdex import physics, robotics
from superdex.physics.paths import resolve_asset
from superdex.physics.utils import render_model_registry

from .base import CameraSpec, ContactGroup, EmbodimentModel, JointTrackingSpec

ROBOT_ASSET = "bots/arm_hand_combos/openarm_v20/openarm_v20.superdex_bot"
ROBOT_ROOT_POSITION = [-0.45, 0.0, 0.0]
RIGHT_ARM_JOINT_PREFIX = "arm_openarm_right_joint"
RIGHT_FINGER_JOINT_PREFIX = "openarm_right_finger_joint"
RIGHT_EE_LINK = "openarm_right_ee_base_link"
GRASP_POINT_EE = np.array([0.0, 0.0, -0.155], dtype=float)
RIGHT_FINGERS_OPEN = -0.78
RIGHT_FINGERS_HOME = -0.34906587
RIGHT_FINGERS_CLOSED = -0.20
GRIPPER_LEVEL_PITCH = np.radians(-6.0)
GRIPPER_LEVEL_ROLL = np.radians(-30.0)

np_real = np.float64 if physics.uses_double_precision() else np.float32


def wrist_camera_specs() -> tuple[CameraSpec, ...]:
    path = Path(__file__).resolve().parent / "config" / "openarm_v2_wrist_cameras.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    intrinsics = config["lens_model"]["intrinsics"]
    return tuple(
        CameraSpec(
            name=name,
            label=f"{entry['side'].title()} wrist",
            kind="actor",
            intrinsics=intrinsics,
            actor_suffix=entry["actor"],
            parent_from_camera_cv=entry["parent_from_camera_cv"],
            calibration_status=config["calibration_status"],
            provenance=config["provenance"],
        )
        for name, entry in config["cameras"].items()
    )


def _joint_dof_metadata(
    prefab: robotics.BotPrefab,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    right_arm: list[int] = []
    right_fingers: list[int] = []
    right_limits: list[list[float]] = []
    finger_limits: list[list[float]] = []
    dof = 0
    for joint in prefab.joints:
        if joint.type != physics.ArticulatedJointType.REVOLUTE:
            continue
        axis = np.asarray(joint.axis, dtype=float)
        lo = float(np.dot(np.asarray(joint.min_limit, dtype=float), axis))
        hi = float(np.dot(np.asarray(joint.max_limit, dtype=float), axis))
        if joint.name.startswith(RIGHT_ARM_JOINT_PREFIX):
            right_arm.append(dof)
            right_limits.append(sorted((lo, hi)))
        elif joint.name.startswith(RIGHT_FINGER_JOINT_PREFIX):
            right_fingers.append(dof)
            finger_limits.append(sorted((lo, hi)))
        dof += 1
    if len(right_arm) != 7 or len(right_fingers) != 2:
        raise RuntimeError(
            "Expected seven right-arm and two right-gripper DOFs, got "
            f"{len(right_arm)} and {len(right_fingers)}."
        )
    return (
        np.asarray(right_arm, dtype=np.int32),
        np.asarray(right_fingers, dtype=np.int32),
        np.asarray(right_limits, dtype=float),
        np.asarray(finger_limits, dtype=float),
    )


def _register_visuals(scene: physics.Scene, info: EmbodimentModel) -> None:
    for handle, link in zip(info.actor.get_nested_link_actors(), info.prefab.links):
        if not link.render_model_file:
            continue
        render_model_registry.register(
            scene,
            handle,
            link.render_model_file,
            physics.TransformRT(
                link.render_model_rotation, link.render_model_translation
            ),
            link.render_model_scale,
        )


def build_openarm_v2(
    scene: physics.Scene,
    context: robotics.RoboticsContext,
    contact_factory: Callable[[float], physics.ContactParams],
    *,
    fingertip_friction: float = 0.95,
) -> EmbodimentModel:
    """Import the whole robot while enabling gravity/control on its right side."""
    prefab = robotics.load_bot_prefab_from_file(str(resolve_asset(ROBOT_ASSET)))
    prefab.world_from_root = physics.TransformRT(translation=ROBOT_ROOT_POSITION)
    controlled_names: set[str] = set()
    for link in prefab.links:
        is_right_moving = link.name.startswith(
            "arm_openarm_right_link"
        ) or link.name in {
            RIGHT_EE_LINK,
            "openarm_right_ee_link1",
            "openarm_right_ee_link2",
        }
        link.has_gravity = is_right_moving
        if is_right_moving:
            controlled_names.add(link.name)
            friction = (
                fingertip_friction
                if link.name in {"openarm_right_ee_link1", "openarm_right_ee_link2"}
                else 0.55
            )
            link.contact = contact_factory(friction)

    for joint in prefab.joints:
        is_parked_left = joint.type == physics.ArticulatedJointType.REVOLUTE and (
            joint.name.startswith("arm_openarm_left_joint")
            or joint.name.startswith("openarm_left_finger_joint")
        )
        if is_parked_left:
            effort_limit = float(joint.effort_limit)
            joint.friction = physics.ArticulatedJointFrictionParams(
                viscous=0.04 * effort_limit,
                coulomb=0.12 * effort_limit,
                falloff_vel=0.002,
            )

    arm_dofs, finger_dofs, arm_limits, finger_limits = _joint_dof_metadata(prefab)
    tracking: dict[str, JointTrackingSpec] = {}
    for link, joint in zip(prefab.links, prefab.joints):
        if (
            link.name not in controlled_names
            or joint.type == physics.ArticulatedJointType.HARD
        ):
            continue
        effort_limit = float(joint.effort_limit)
        if link.name.startswith("arm_openarm_right_link") or link.name == RIGHT_EE_LINK:
            tracking[link.name] = JointTrackingSpec(
                stiffness=0.70 * effort_limit / 0.05,
                damping=0.45 * effort_limit,
                saturation=0.05,
            )
        else:
            tracking[link.name] = JointTrackingSpec(
                stiffness=0.35 * effort_limit / 0.15,
                damping=0.28,
                saturation=0.15,
            )
    bot = robotics.create_bot(scene, prefab, context)
    actor = bot.get_articulated_actor()
    pose = physics.DynamicArrayReal(actor.get_num_dofs())
    actor.get_articulated_pose(pose)
    info = EmbodimentModel(
        embodiment_id="openarm_v2",
        display_name="OpenArm v2 right arm",
        bot=bot,
        prefab=prefab,
        actor=actor,
        arm_dofs=arm_dofs,
        hand_dofs=finger_dofs,
        arm_limits=arm_limits,
        hand_limits=finger_limits,
        default_pose=np.asarray(pose, dtype=float).copy(),
        end_effector_link=RIGHT_EE_LINK,
        grasp_point_local=GRASP_POINT_EE.copy(),
        hand_poses={
            "home": np.full(2, RIGHT_FINGERS_HOME),
            "open": np.full(2, RIGHT_FINGERS_OPEN),
            "preshape": np.full(2, RIGHT_FINGERS_OPEN),
            "closed": np.full(2, RIGHT_FINGERS_CLOSED),
        },
        contact_groups=(
            ContactGroup(
                "finger1", ("openarm_right_ee_link1",), (int(finger_dofs[0]),)
            ),
            ContactGroup(
                "finger2", ("openarm_right_ee_link2",), (int(finger_dofs[1]),)
            ),
        ),
        approach_direction_world=np.array([1.0, 0.0, 0.0], dtype=float),
        pregrasp_distance=0.13,
        grasp_tolerance_m=0.025,
        cameras=wrist_camera_specs(),
        controlled_link_names=frozenset(controlled_names),
        tracking=tracking,
    )
    _register_visuals(scene, info)
    return info


def destroy_openarm_v2(scene: physics.Scene, info: EmbodimentModel) -> None:
    render_model_registry.unregister_actors(scene, info.actor.get_nested_link_actors())
    robotics.destroy_bot(scene, info.bot)


class OpenArmKinematics:
    """OpenArm kinematic twin with task-supplied collision evaluation."""

    def __init__(
        self,
        context: robotics.RoboticsContext,
        reference: EmbodimentModel,
        contact_factory: Callable[[float], physics.ContactParams],
        collision_model: object,
        *,
        approach_pitch: float = 0.0,
    ) -> None:
        """``approach_pitch`` [rad] tilts the level gripper nose-down about world Y."""
        self.scene = physics.create_scene("OpenArm v2 trajectory optimization")
        self.info = build_openarm_v2(self.scene, context, contact_factory)
        self.actor = self.info.actor
        self.reference_pose = reference.default_pose.copy()
        self.collision_model = collision_model
        self.link_actors: dict[str, physics.Actor] = {}
        for handle in self.actor.get_nested_link_actors():
            actor = self.scene.get_actor(handle)
            self.link_actors[actor.get_name().split("/", 1)[-1]] = actor
        self.ee = self.link_actors[RIGHT_EE_LINK]
        self.ee_handle = self.ee.get_handle()
        level_rotation = (
            self.ee.get_root_transform().rotation
            * physics.Quaternion.rotation_y(GRIPPER_LEVEL_PITCH)
            * physics.Quaternion.rotation_z(GRIPPER_LEVEL_ROLL)
        )
        if approach_pitch:
            level_rotation = (
                physics.Quaternion.from_rotation_vector(
                    [0.0, float(approach_pitch), 0.0]
                )
                * level_rotation
            )
        self.approach_pitch = float(approach_pitch)
        self.solver = physics.experimental.create_ik_solver(self.scene)
        self.position_target = self.solver.create_position_target(
            self.ee_handle, GRASP_POINT_EE, [0.0, 0.0, 0.0], 1.0e5
        )
        self.solver.create_rotation_target(
            self.ee_handle,
            [0.0, 0.0, 0.0],
            level_rotation.to_rotation_vector(),
            1.0e3,
        )
        self._moving_proxies = self._make_proxies("right")
        self._static_proxies = self._make_proxies("left")
        self.moving_proxy_actors = [proxy[0] for proxy in self._moving_proxies]

    def _make_proxies(
        self, side: str
    ) -> list[tuple[physics.Actor, npt.NDArray[np.float64], float]]:
        proxies = []
        for name, actor in self.link_actors.items():
            is_arm = name.startswith(f"arm_openarm_{side}_link")
            is_hand = name in {
                f"openarm_{side}_ee_base_link",
                f"openarm_{side}_ee_link1",
                f"openarm_{side}_ee_link2",
            }
            if not (is_arm or is_hand):
                continue
            bounds = actor.get_aabb_local()
            lo = np.asarray(bounds.min, dtype=float)
            hi = np.asarray(bounds.max, dtype=float)
            proxies.append(
                (
                    actor,
                    0.5 * (lo + hi),
                    0.55 * float(np.linalg.norm(0.5 * (hi - lo))) + 0.003,
                )
            )
        return proxies

    def set_arm_pose(self, arm_pose: npt.ArrayLike) -> None:
        pose = self.reference_pose.copy()
        pose[self.info.arm_dofs] = arm_pose
        pose[self.info.hand_dofs] = self.info.hand_poses["home"]
        self.actor.set_articulated_pose_from_joints(np.asarray(pose, dtype=np_real))

    def grasp_point_world(self, arm_pose: npt.ArrayLike) -> npt.NDArray[np.float64]:
        self.set_arm_pose(arm_pose)
        transform = self.ee.get_root_transform() * physics.TransformRT(
            translation=GRASP_POINT_EE
        )
        return np.asarray(transform.translation, dtype=float)

    def solve(
        self, target: npt.ArrayLike, seed: npt.ArrayLike
    ) -> npt.NDArray[np.float64]:
        seed_pose = self.reference_pose.copy()
        seed_pose[self.info.arm_dofs] = seed
        seed_pose[self.info.hand_dofs] = self.info.hand_poses["open"]
        self.actor.set_articulated_pose_from_joints(
            np.asarray(seed_pose, dtype=np_real)
        )
        target = np.asarray(target, dtype=float)
        self.position_target.set_target_position(target)
        self.solver.solve_ik()
        solved = physics.DynamicArrayReal(self.actor.get_num_dofs())
        self.actor.get_articulated_pose(solved)
        arm = np.asarray(solved, dtype=float)[self.info.arm_dofs].copy()
        limits = self.info.arm_limits
        violation = float(np.max(np.maximum(limits[:, 0] - arm, arm - limits[:, 1])))
        if violation > 0.015:
            raise RuntimeError(
                f"IK target {target.tolist()} exceeds a right-arm joint limit by "
                f"{violation:.4f} rad."
            )
        arm = np.clip(arm, limits[:, 0], limits[:, 1])
        error = float(np.linalg.norm(self.grasp_point_world(arm) - target))
        if error > 0.015:
            raise RuntimeError(
                f"IK target {target.tolist()} was missed by {error:.4f} m."
            )
        return arm

    @staticmethod
    def _world_proxy(
        proxy: tuple[physics.Actor, npt.NDArray[np.float64], float],
    ) -> tuple[npt.NDArray[np.float64], float]:
        actor, center, radius = proxy
        transform = actor.get_root_transform() * physics.TransformRT(translation=center)
        return np.asarray(transform.translation, dtype=float), radius

    def world_proxy_spheres(
        self, role: str
    ) -> list[tuple[npt.NDArray[np.float64], float]]:
        proxies = self._static_proxies if role == "static" else self._moving_proxies
        return [self._world_proxy(proxy) for proxy in proxies]

    def collision_cost(self, arm_pose: npt.ArrayLike) -> tuple[float, float]:
        return self.collision_model.evaluate(self, arm_pose)

    def close(self) -> None:
        self.solver.clear_position_target(self.ee_handle)
        self.solver.clear_rotation_target(self.ee_handle)
        destroy_openarm_v2(self.scene, self.info)
        physics.experimental.destroy_ik_solver(self.solver)
        physics.destroy_scene(self.scene)
