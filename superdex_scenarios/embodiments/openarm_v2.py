"""Full OpenArm v2 model with per-side control and kinematics.

The robot is imported whole.  ``sides`` selects which arm/gripper chains are
gravity-enabled and motor-controlled; the other side stays parked under joint
friction.  The ball-and-bowl and sponge-and-plate tasks use the right arm
only; the bowl-moving task drives both arms.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
from superdex import physics, robotics
from superdex.physics.paths import resolve_asset
from superdex.physics.utils import render_model_registry

from ..planning.pose_ik import PoseIKOptimizer
from .base import CameraSpec, ContactGroup, EmbodimentModel, JointTrackingSpec

ROBOT_ASSET = "bots/arm_hand_combos/openarm_v20/openarm_v20.superdex_bot"
ROBOT_ROOT_POSITION = [-0.45, 0.0, 0.0]
SIDES = ("right", "left")
RIGHT_ARM_JOINT_PREFIX = "arm_openarm_right_joint"
RIGHT_FINGER_JOINT_PREFIX = "openarm_right_finger_joint"
RIGHT_EE_LINK = "openarm_right_ee_base_link"
GRASP_POINT_EE = np.array([0.0, 0.0, -0.155], dtype=float)
# Right-gripper finger joints are negative when open; the left gripper is the
# mirror image and opens with positive values.
RIGHT_FINGERS_OPEN = -0.78
RIGHT_FINGERS_HOME = -0.34906587
RIGHT_FINGERS_CLOSED = -0.20
FINGER_JOINT_OPEN_MAGNITUDE = 0.78
GRIPPER_LEVEL_PITCH = np.radians(-6.0)
GRIPPER_LEVEL_ROLL = np.radians(-30.0)

np_real = np.float64 if physics.uses_double_precision() else np.float32


def arm_joint_prefix(side: str) -> str:
    return f"arm_openarm_{side}_joint"


def finger_joint_prefix(side: str) -> str:
    return f"openarm_{side}_finger_joint"


def ee_link(side: str) -> str:
    return f"openarm_{side}_ee_base_link"


def finger_links(side: str) -> tuple[str, str]:
    return (f"openarm_{side}_ee_link1", f"openarm_{side}_ee_link2")


def finger_sign(side: str) -> float:
    """Sign of an *open* finger joint value for ``side``."""
    return -1.0 if side == "right" else 1.0


def finger_joint_from_aperture(aperture: npt.ArrayLike, side: str) -> npt.NDArray[np.float64]:
    """Map a normalized gripper aperture (0 closed .. 1 open) to a finger joint value."""
    aperture = np.clip(np.asarray(aperture, dtype=float), 0.0, 1.0)
    return finger_sign(side) * FINGER_JOINT_OPEN_MAGNITUDE * aperture


def aperture_from_finger_joint(value: npt.ArrayLike, side: str) -> npt.NDArray[np.float64]:
    value = np.asarray(value, dtype=float)
    return np.clip(value / (finger_sign(side) * FINGER_JOINT_OPEN_MAGNITUDE), 0.0, 1.0)


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
    prefab: robotics.BotPrefab, sides: Sequence[str]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], tuple[str, ...]]:
    """Return per-side arm/finger DOF indices, their limits, and every DOF's name."""
    dofs: dict[str, list[int]] = {}
    limits: dict[str, list[list[float]]] = {}
    names: list[str] = []
    for side in SIDES:
        dofs[f"{side}_arm"] = []
        dofs[f"{side}_gripper"] = []
        limits[f"{side}_arm"] = []
        limits[f"{side}_gripper"] = []
    dof = 0
    for joint in prefab.joints:
        if joint.type == physics.ArticulatedJointType.HARD:
            continue
        if joint.type != physics.ArticulatedJointType.REVOLUTE:
            raise RuntimeError(f"Unexpected joint type {joint.type} for {joint.name}.")
        names.append(joint.name)
        axis = np.asarray(joint.axis, dtype=float)
        lo = float(np.dot(np.asarray(joint.min_limit, dtype=float), axis))
        hi = float(np.dot(np.asarray(joint.max_limit, dtype=float), axis))
        for side in SIDES:
            if joint.name.startswith(arm_joint_prefix(side)):
                dofs[f"{side}_arm"].append(dof)
                limits[f"{side}_arm"].append(sorted((lo, hi)))
            elif joint.name.startswith(finger_joint_prefix(side)):
                dofs[f"{side}_gripper"].append(dof)
                limits[f"{side}_gripper"].append(sorted((lo, hi)))
        dof += 1
    for side in sides:
        if len(dofs[f"{side}_arm"]) != 7 or len(dofs[f"{side}_gripper"]) != 2:
            raise RuntimeError(
                f"Expected seven {side}-arm and two {side}-gripper DOFs, got "
                f"{len(dofs[f'{side}_arm'])} and {len(dofs[f'{side}_gripper'])}."
            )
    return (
        {key: np.asarray(value, dtype=np.int32) for key, value in dofs.items()},
        {key: np.asarray(value, dtype=float).reshape(-1, 2) for key, value in limits.items()},
        tuple(names),
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


def _is_moving_link(name: str, side: str) -> bool:
    return name.startswith(f"arm_openarm_{side}_link") or name in {
        ee_link(side),
        *finger_links(side),
    }


def build_openarm_v2(
    scene: physics.Scene,
    context: robotics.RoboticsContext,
    contact_factory: Callable[[float], physics.ContactParams],
    *,
    fingertip_friction: float = 0.95,
    sides: Sequence[str] = ("right",),
) -> EmbodimentModel:
    """Import the whole robot while enabling gravity/control on ``sides``.

    ``sides`` is ``("right",)`` for the single-arm tasks or ``("right", "left")``
    for bimanual control.  The right side always comes first in ``arm_dofs`` and
    ``hand_dofs``; the embodiment's grasp point and end-effector link stay on the
    right gripper, and ``dof_groups`` names each side's arm and gripper DOFs.
    """
    sides = tuple(sides)
    for side in sides:
        if side not in SIDES:
            raise ValueError(f"Unknown OpenArm side {side!r}; expected one of {SIDES}.")
    if "right" in sides:
        sides = ("right", *(s for s in sides if s != "right"))
    prefab = robotics.load_bot_prefab_from_file(str(resolve_asset(ROBOT_ASSET)))
    prefab.world_from_root = physics.TransformRT(translation=ROBOT_ROOT_POSITION)
    controlled_names: set[str] = set()
    for link in prefab.links:
        moving_side = next((s for s in sides if _is_moving_link(link.name, s)), None)
        link.has_gravity = moving_side is not None
        if moving_side is not None:
            controlled_names.add(link.name)
            friction = (
                fingertip_friction if link.name in finger_links(moving_side) else 0.55
            )
            link.contact = contact_factory(friction)

    parked_sides = tuple(s for s in SIDES if s not in sides)
    for joint in prefab.joints:
        is_parked = joint.type == physics.ArticulatedJointType.REVOLUTE and any(
            joint.name.startswith(arm_joint_prefix(s))
            or joint.name.startswith(finger_joint_prefix(s))
            for s in parked_sides
        )
        if is_parked:
            effort_limit = float(joint.effort_limit)
            joint.friction = physics.ArticulatedJointFrictionParams(
                viscous=0.04 * effort_limit,
                coulomb=0.12 * effort_limit,
                falloff_vel=0.002,
            )

    dofs, limits, dof_names = _joint_dof_metadata(prefab, sides)
    tracking: dict[str, JointTrackingSpec] = {}
    for link, joint in zip(prefab.links, prefab.joints):
        if (
            link.name not in controlled_names
            or joint.type == physics.ArticulatedJointType.HARD
        ):
            continue
        effort_limit = float(joint.effort_limit)
        is_arm = any(
            link.name.startswith(f"arm_openarm_{s}_link") or link.name == ee_link(s)
            for s in sides
        )
        if is_arm:
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

    arm_dofs = np.concatenate([dofs[f"{s}_arm"] for s in sides]).astype(np.int32)
    hand_dofs = np.concatenate([dofs[f"{s}_gripper"] for s in sides]).astype(np.int32)
    arm_limits = np.vstack([limits[f"{s}_arm"] for s in sides])
    hand_limits = np.vstack([limits[f"{s}_gripper"] for s in sides])

    def fingers(magnitude: float, *, left_magnitude: float | None = None) -> npt.NDArray[np.float64]:
        """Right gripper at ``magnitude``; other sides at ``left_magnitude`` (default: same)."""
        parts = []
        for s in sides:
            value = magnitude if (s == "right" or left_magnitude is None) else left_magnitude
            parts.append(np.full(2, finger_sign(s) * value))
        return np.concatenate(parts)

    contact_groups: list[ContactGroup] = []
    for s in sides:
        prefix = "" if s == "right" else f"{s}_"
        for index, link_name in enumerate(finger_links(s), start=1):
            contact_groups.append(
                ContactGroup(
                    f"{prefix}finger{index}",
                    (link_name,),
                    (int(dofs[f"{s}_gripper"][index - 1]),),
                )
            )

    dof_groups = {
        key: dofs[key].copy() for s in sides for key in (f"{s}_arm", f"{s}_gripper")
    }
    info = EmbodimentModel(
        embodiment_id="openarm_v2" if sides == ("right",) else "openarm_v2_bimanual",
        display_name=(
            "OpenArm v2 right arm" if sides == ("right",) else "OpenArm v2 both arms"
        ),
        bot=bot,
        prefab=prefab,
        actor=actor,
        arm_dofs=arm_dofs,
        hand_dofs=hand_dofs,
        arm_limits=arm_limits,
        hand_limits=hand_limits,
        default_pose=np.asarray(pose, dtype=float).copy(),
        end_effector_link=RIGHT_EE_LINK,
        grasp_point_local=GRASP_POINT_EE.copy(),
        # Named poses move the right gripper; on a bimanual build the left
        # gripper stays parked so single-arm policies leave it alone.  Policies
        # that drive both grippers pass explicit vectors (see move_bowl_ball).
        hand_poses={
            "home": fingers(-RIGHT_FINGERS_HOME),
            "open": fingers(-RIGHT_FINGERS_OPEN, left_magnitude=-RIGHT_FINGERS_HOME),
            "preshape": fingers(-RIGHT_FINGERS_OPEN, left_magnitude=-RIGHT_FINGERS_HOME),
            "closed": fingers(-RIGHT_FINGERS_CLOSED, left_magnitude=-RIGHT_FINGERS_HOME),
        },
        contact_groups=tuple(contact_groups),
        approach_direction_world=np.array([1.0, 0.0, 0.0], dtype=float),
        pregrasp_distance=0.13,
        grasp_tolerance_m=0.025,
        cameras=wrist_camera_specs(),
        controlled_link_names=frozenset(controlled_names),
        tracking=tracking,
        dof_groups=dof_groups,
        dof_names=dof_names,
    )
    _register_visuals(scene, info)
    return info


def right_gripper_pose(info: EmbodimentModel, values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Full gripper vector with the right jaws at ``values`` and any other gripper parked."""
    hand = info.hand_poses["home"].copy()
    right = np.isin(info.hand_dofs, info.dof_groups["right_gripper"])
    hand[right] = np.asarray(values, dtype=float)
    return hand


def freeze_other_arms(
    path: npt.ArrayLike, info: EmbodimentModel, side: str = "right"
) -> npt.NDArray[np.float64]:
    """Pin every arm except ``side`` to its parked pose along an arm path.

    Single-arm planners run unchanged on a bimanual build; this keeps the
    optimizer from moving the other arm.
    """
    path = np.array(path, dtype=float)
    keep = np.isin(info.arm_dofs, info.dof_groups[f"{side}_arm"])
    path[:, ~keep] = info.default_pose[info.arm_dofs][~keep]
    return path


def destroy_openarm_v2(scene: physics.Scene, info: EmbodimentModel) -> None:
    render_model_registry.unregister_actors(scene, info.actor.get_nested_link_actors())
    robotics.destroy_bot(scene, info.bot)


def _quaternion_xyzw(rotation: physics.Quaternion) -> npt.NDArray[np.float64]:
    return np.array(
        [rotation[0], rotation[1], rotation[2], rotation[3]], dtype=float
    )


def quaternion_from_xyzw(quaternion_xyzw: npt.ArrayLike) -> physics.Quaternion:
    """Build a SuperDex quaternion from an (x, y, z, w) array via its rotation vector."""
    q = np.asarray(quaternion_xyzw, dtype=float).reshape(4)
    q = q / np.linalg.norm(q)
    if q[3] < 0.0:
        q = -q
    angle = 2.0 * np.arctan2(np.linalg.norm(q[:3]), q[3])
    axis_norm = np.linalg.norm(q[:3])
    if axis_norm < 1e-12:
        return physics.Quaternion.identity()
    return physics.Quaternion.from_rotation_vector((q[:3] / axis_norm) * angle)


class OpenArmKinematics:
    """OpenArm kinematic twin for one side with task-supplied collision evaluation.

    The twin lives in a private physics scene, so solving poses never disturbs
    the simulated robot.  ``solve`` keeps the authored level gripper
    orientation (optionally pitched nose-down); ``solve_pose`` accepts an
    arbitrary world orientation of the grasp-point frame, which is what a
    retargeting pipeline needs.
    """

    def __init__(
        self,
        context: robotics.RoboticsContext,
        reference: EmbodimentModel,
        contact_factory: Callable[[float], physics.ContactParams],
        collision_model: object,
        *,
        approach_pitch: float = 0.0,
        side: str = "right",
    ) -> None:
        """``approach_pitch`` [rad] tilts the level gripper nose-down about world Y."""
        if side not in SIDES:
            raise ValueError(f"Unknown OpenArm side {side!r}.")
        self.side = side
        sides = tuple(
            s for s in SIDES if f"{s}_arm" in reference.dof_groups
        ) or ("right",)
        self.scene = physics.create_scene(f"OpenArm v2 {side} kinematics")
        self.info = build_openarm_v2(self.scene, context, contact_factory, sides=sides)
        self.actor = self.info.actor
        self.reference_pose = reference.default_pose.copy()
        self.collision_model = collision_model
        self.link_actors: dict[str, physics.Actor] = {}
        for handle in self.actor.get_nested_link_actors():
            actor = self.scene.get_actor(handle)
            self.link_actors[actor.get_name().split("/", 1)[-1]] = actor
        self.ee = self.link_actors[ee_link(side)]
        self.ee_handle = self.ee.get_handle()
        self.side_arm_dofs = self.info.dof_groups[f"{side}_arm"]
        self.side_hand_dofs = self.info.dof_groups[f"{side}_gripper"]
        mirror = 1.0 if side == "right" else -1.0
        level_rotation = (
            self.ee.get_root_transform().rotation
            * physics.Quaternion.rotation_y(mirror * GRIPPER_LEVEL_PITCH)
            * physics.Quaternion.rotation_z(mirror * GRIPPER_LEVEL_ROLL)
        )
        if approach_pitch:
            level_rotation = (
                physics.Quaternion.from_rotation_vector(
                    [0.0, float(approach_pitch), 0.0]
                )
                * level_rotation
            )
        self.level_rotation = level_rotation
        self.approach_pitch = float(approach_pitch)
        self.solver = physics.experimental.create_ik_solver(self.scene)
        self.position_target = self.solver.create_position_target(
            self.ee_handle, GRASP_POINT_EE, [0.0, 0.0, 0.0], 1.0e5
        )
        self.rotation_target = self.solver.create_rotation_target(
            self.ee_handle,
            [0.0, 0.0, 0.0],
            level_rotation.to_rotation_vector(),
            1.0e3,
        )
        self._moving_proxies = self._make_proxies(side)
        self._static_proxies = self._make_proxies("left" if side == "right" else "right")
        self.moving_proxy_actors = [proxy[0] for proxy in self._moving_proxies]

    def _make_proxies(
        self, side: str
    ) -> list[tuple[physics.Actor, npt.NDArray[np.float64], float]]:
        proxies = []
        for name, actor in self.link_actors.items():
            is_arm = name.startswith(f"arm_openarm_{side}_link")
            is_hand = name in {ee_link(side), *finger_links(side)}
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

    # -- pose bookkeeping --------------------------------------------------

    def set_arm_pose(self, arm_pose: npt.ArrayLike) -> None:
        pose = self.reference_pose.copy()
        pose[self.info.arm_dofs] = arm_pose
        pose[self.info.hand_dofs] = self.info.hand_poses["home"]
        self.actor.set_articulated_pose_from_joints(np.asarray(pose, dtype=np_real))

    def set_full_pose(self, pose: npt.ArrayLike) -> None:
        """Set every articulation DOF of the twin (forward kinematics)."""
        self.actor.set_articulated_pose_from_joints(np.asarray(pose, dtype=np_real))

    def grasp_point_world(self, arm_pose: npt.ArrayLike) -> npt.NDArray[np.float64]:
        self.set_arm_pose(arm_pose)
        transform = self.ee.get_root_transform() * physics.TransformRT(
            translation=GRASP_POINT_EE
        )
        return np.asarray(transform.translation, dtype=float)

    def grasp_point_pose(
        self, arm_pose: npt.ArrayLike
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """World position and (x, y, z, w) quaternion of the grasp-point frame."""
        self.set_arm_pose(arm_pose)
        transform = self.ee.get_root_transform() * physics.TransformRT(
            translation=GRASP_POINT_EE
        )
        return (
            np.asarray(transform.translation, dtype=float),
            _quaternion_xyzw(transform.rotation),
        )

    def grasp_point_world_side(
        self, side: str, arm_pose: npt.ArrayLike
    ) -> npt.NDArray[np.float64]:
        """Grasp point of ``side``'s gripper for an arm pose (forward kinematics only)."""
        self.set_arm_pose(arm_pose)
        transform = self.link_actors[ee_link(side)].get_root_transform() * physics.TransformRT(
            translation=GRASP_POINT_EE
        )
        return np.asarray(transform.translation, dtype=float)

    def link_transforms(self, pose: npt.ArrayLike) -> dict[str, npt.NDArray[np.float64]]:
        """World (x, y, z, qx, qy, qz, qw) of every link for a full pose."""
        self.set_full_pose(pose)
        result = {}
        for name, actor in self.link_actors.items():
            transform = actor.get_root_transform()
            result[name] = np.concatenate(
                [
                    np.asarray(transform.translation, dtype=float),
                    _quaternion_xyzw(transform.rotation),
                ]
            )
        return result

    # -- inverse kinematics -------------------------------------------------

    def _solve(
        self,
        target: npt.NDArray[np.float64],
        rotation: physics.Quaternion,
        seed: npt.ArrayLike,
        *,
        position_tolerance: float,
        rotation_weight: float,
    ) -> npt.NDArray[np.float64]:
        seed_pose = self.reference_pose.copy()
        seed_pose[self.info.arm_dofs] = seed
        seed_pose[self.info.hand_dofs] = self.info.hand_poses["open"]
        self.actor.set_articulated_pose_from_joints(
            np.asarray(seed_pose, dtype=np_real)
        )
        self.position_target.set_target_position(target)
        self.rotation_target.set_target_rotation(rotation)
        self.rotation_target.set_stiffness(rotation_weight)
        self.solver.solve_ik()
        solved = physics.DynamicArrayReal(self.actor.get_num_dofs())
        self.actor.get_articulated_pose(solved)
        arm = np.asarray(solved, dtype=float)[self.info.arm_dofs].copy()
        # Only this side's chain is solved; keep the other arm exactly at the seed.
        seed_arm = np.asarray(seed, dtype=float)
        side_mask = np.isin(self.info.arm_dofs, self.side_arm_dofs)
        arm[~side_mask] = seed_arm[~side_mask]
        limits = self.info.arm_limits
        violation = float(np.max(np.maximum(limits[:, 0] - arm, arm - limits[:, 1])))
        if violation > 0.015:
            raise RuntimeError(
                f"IK target {target.tolist()} exceeds a {self.side}-arm joint limit by "
                f"{violation:.4f} rad."
            )
        arm = np.clip(arm, limits[:, 0], limits[:, 1])
        error = float(np.linalg.norm(self.grasp_point_world(arm) - target))
        if error > position_tolerance:
            raise RuntimeError(
                f"IK target {target.tolist()} was missed by {error:.4f} m."
            )
        return arm

    def solve(
        self, target: npt.ArrayLike, seed: npt.ArrayLike
    ) -> npt.NDArray[np.float64]:
        """Place the grasp point at ``target`` with the level gripper orientation."""
        return self._solve(
            np.asarray(target, dtype=float),
            self.level_rotation,
            seed,
            position_tolerance=0.015,
            rotation_weight=1.0e3,
        )

    def solve_pose(
        self,
        position: npt.ArrayLike,
        quaternion_xyzw: npt.ArrayLike,
        seed: npt.ArrayLike,
        *,
        position_tolerance: float = 0.015,
        rotation_weight: float = 1.0e3,
    ) -> npt.NDArray[np.float64]:
        """Place the grasp-point frame at a world position and orientation.

        ``quaternion_xyzw`` is the desired world rotation of the end-effector
        frame (the grasp point shares the end-effector orientation).  The
        rotation objective is soft; the result reports the achieved pose through
        :meth:`grasp_point_pose`.
        """
        return self._solve(
            np.asarray(position, dtype=float),
            quaternion_from_xyzw(quaternion_xyzw),
            seed,
            position_tolerance=position_tolerance,
            rotation_weight=rotation_weight,
        )

    def solve_pose_optimized(
        self,
        position: npt.ArrayLike,
        quaternion_xyzw: npt.ArrayLike,
        seed: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        """Solve one pose with bounded nonlinear SQP, warm-started from ``seed``."""

        seed_arm = np.asarray(seed, dtype=float).copy()
        side_mask = np.isin(self.info.arm_dofs, self.side_arm_dofs)
        lower = self.info.arm_limits[side_mask, 0] + 1.0e-4
        upper = self.info.arm_limits[side_mask, 1] - 1.0e-4

        def full_pose(side_pose: npt.ArrayLike) -> npt.NDArray[np.float64]:
            pose = seed_arm.copy()
            pose[side_mask] = np.asarray(side_pose, dtype=float)
            return pose

        optimizer = PoseIKOptimizer(
            lambda side_pose: self.grasp_point_pose(full_pose(side_pose)),
            lambda side_pose: self.collision_cost(full_pose(side_pose))[1],
        )
        result = optimizer.solve(
            position,
            quaternion_xyzw,
            seed_arm[side_mask],
            lower,
            upper,
        )
        arm = full_pose(result.configuration)
        achieved_position, _ = self.grasp_point_pose(arm)
        error = float(
            np.linalg.norm(achieved_position - np.asarray(position, dtype=float))
        )
        _, clearance = self.collision_cost(arm)
        if error > optimizer.settings.position_tolerance_m or clearance < 0.0:
            raise RuntimeError(
                f"optimized IK failed: position error={error:.4f} m, "
                f"clearance={clearance:+.4f} m, backend={result.backend}, "
                f"status={result.message}"
            )
        return arm

    def solve_optimized(
        self, target: npt.ArrayLike, seed: npt.ArrayLike
    ) -> npt.NDArray[np.float64]:
        """Bounded SQP position IK using the authored level orientation."""

        return self.solve_pose_optimized(
            target, _quaternion_xyzw(self.level_rotation), seed
        )

    # -- collision proxies --------------------------------------------------

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
