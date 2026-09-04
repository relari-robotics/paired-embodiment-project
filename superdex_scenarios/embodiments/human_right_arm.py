"""Physical Meta XR right-hand embodiment and neutral kinematic scaffold."""

from __future__ import annotations

import math
from collections.abc import Callable
from itertools import combinations

import numpy as np
import numpy.typing as npt
from superdex import physics, robotics
from superdex.physics.paths import resolve_asset
from superdex.physics.utils import render_model_registry

from .base import ContactGroup, EmbodimentModel, JointTrackingSpec

HAND_ASSET = "bots/hands/oculus_xr/right/oculus_xr_hand_lowpoly_right.superdex_bot"
ARM_DOF_COUNT = 6
MOCHI_REAL = np.float64 if physics.uses_double_precision() else np.float32

ARM_LINK_NAMES = (
    "hand_carrier_anchor",
    "hand_carrier_x",
    "hand_carrier_y",
    "hand_carrier_z",
    "hand_carrier_yaw",
    "hand_carrier_pitch",
    "hand_carrier_roll",
)

# Initial placement only. Projects are expected to choose every commanded pose.
ARM_HOME = np.array(
    [0.263, -0.418, 0.651, -0.145, 1.431, 0.050],
    dtype=float,
)


def _joint(
    name: str,
    axis: npt.ArrayLike,
    limits: tuple[float, float],
    effort_limit: float,
    joint_type: physics.ArticulatedJointType = physics.ArticulatedJointType.REVOLUTE,
) -> robotics.BotJointPrefab:
    axis_array = np.asarray(axis, dtype=float)
    lo, hi = limits
    return robotics.BotJointPrefab(
        name=name,
        type=joint_type,
        axis=axis_array,
        min_limit=axis_array * lo,
        max_limit=axis_array * hi,
        limit_stiffness=1200.0,
        limit_damping=4.0,
        effort_limit=effort_limit,
        friction=physics.ArticulatedJointFrictionParams(
            viscous=0.06,
            coulomb=0.025,
            falloff_vel=0.01,
        ),
    )


def _invisible_link(
    name: str,
    parent: int,
    contact: physics.ContactParams,
) -> robotics.BotLinkPrefab:
    return robotics.BotLinkPrefab(
        name=name,
        parent_link=parent,
        shape_file=str(resolve_asset("prefabs/sphere/collision/sphere.mochi.h5")),
        shape_scale=np.full(3, 0.001 / 0.015),
        collider_type=physics.ColliderType.NONE,
        layer="human_arm",
        contact=contact,
        has_gravity=False,
        mass=0.001,
    )


def _make_carrier(
    contact: physics.ContactParams,
) -> tuple[list[robotics.BotJointPrefab], list[robotics.BotLinkPrefab]]:
    """Create a neutral six-DoF Cartesian carrier for the physical hand."""
    joints = [
        robotics.BotJointPrefab(
            name="human_hand_anchor",
            type=physics.ArticulatedJointType.HARD,
        ),
        _joint(
            "human_hand_x",
            [1, 0, 0],
            (-0.45, 0.65),
            180.0,
            physics.ArticulatedJointType.PRISMATIC,
        ),
        _joint(
            "human_hand_y",
            [0, 1, 0],
            (-0.45, 0.45),
            180.0,
            physics.ArticulatedJointType.PRISMATIC,
        ),
        _joint(
            "human_hand_z",
            [0, 0, 1],
            (0.39, 0.85),
            220.0,
            physics.ArticulatedJointType.PRISMATIC,
        ),
        _joint("human_hand_yaw", [0, 0, 1], (-math.pi, math.pi), 45.0),
        _joint("human_hand_pitch", [0, 1, 0], (-math.pi / 2, math.pi / 2), 45.0),
        _joint("human_hand_roll", [1, 0, 0], (-math.pi / 2, math.pi / 2), 45.0),
    ]
    links = [
        _invisible_link(name, index - 1, contact)
        for index, name in enumerate(ARM_LINK_NAMES)
    ]
    return joints, links


def _finger_dof_layout(
    prefab: robotics.BotPrefab,
) -> tuple[npt.NDArray[np.int32], dict[str, list[int]]]:
    dof = 0
    hand_dofs: list[int] = []
    groups = {name: [] for name in ("thumb", "index", "middle", "ring", "pinky")}
    for joint in prefab.joints:
        if joint.type == physics.ArticulatedJointType.HARD:
            count = 0
        elif joint.type == physics.ArticulatedJointType.SPHERICAL:
            count = 3
        elif joint.type == physics.ArticulatedJointType.FREE:
            count = 6
        else:
            count = 1
        if joint.name.startswith("joint_") and count:
            indices = list(range(dof, dof + count))
            hand_dofs.extend(indices)
            for digit, digit_indices in groups.items():
                if digit in joint.name:
                    digit_indices.extend(indices)
                    break
        dof += count
    return np.asarray(hand_dofs, dtype=np.int32), groups


def _dof_limits(prefab: robotics.BotPrefab) -> npt.NDArray[np.float64]:
    limits: list[tuple[float, float]] = []
    for joint in prefab.joints:
        if joint.type == physics.ArticulatedJointType.HARD:
            continue
        if joint.type == physics.ArticulatedJointType.SPHERICAL:
            minimum = np.asarray(joint.min_limit, dtype=float)
            maximum = np.asarray(joint.max_limit, dtype=float)
            limits.extend(
                zip(np.minimum(minimum, maximum), np.maximum(minimum, maximum))
            )
        elif joint.type == physics.ArticulatedJointType.FREE:
            limits.extend([(-np.inf, np.inf)] * 6)
        else:
            axis = np.asarray(joint.axis, dtype=float)
            lo = float(np.dot(np.asarray(joint.min_limit, dtype=float), axis))
            hi = float(np.dot(np.asarray(joint.max_limit, dtype=float), axis))
            limits.append(tuple(sorted((lo, hi))))
    return np.asarray(limits, dtype=float)


def _register_visuals(scene: physics.Scene, info: EmbodimentModel) -> None:
    for handle, link in zip(info.actor.get_nested_link_actors(), info.prefab.links):
        if not link.render_model_file:
            continue
        render_model_registry.register(
            scene,
            handle,
            link.render_model_file,
            physics.TransformRT(
                link.render_model_rotation,
                link.render_model_translation,
            ),
            link.render_model_scale,
        )


def build_human_right_arm(
    scene: physics.Scene,
    context: robotics.RoboticsContext,
    contact_factory: Callable[[float], physics.ContactParams],
) -> EmbodimentModel:
    """Create the articulated hand and expose its joints without planning motion."""
    prefab = robotics.load_bot_prefab_from_file(str(resolve_asset(HAND_ASSET)))
    original_links = list(prefab.links)
    original_link_names = [link.name for link in original_links]
    original_joints = list(prefab.joints)
    carrier_joints, carrier_links = _make_carrier(contact_factory(0.62))
    offset = len(carrier_links)

    original_joints[0].type = physics.ArticulatedJointType.HARD
    original_joints[0].name = "human_hand_wrist_mount"
    original_joints[0].parent_link_from_joint = physics.TransformRT()
    original_links[0].parent_link = offset - 1
    original_links[0].parent_joint_from_link = physics.TransformRT()
    for link in original_links[1:]:
        link.parent_link += offset
    for link in original_links:
        link.layer = "human_hand"
        link.contact = contact_factory(1.15)
        link.has_gravity = True
    for joint in original_joints[1:]:
        joint.effort_limit = 3.60 if "thumb" in joint.name else 2.00

    prefab.name = "human_right_hand"
    prefab.joints = [*carrier_joints, *original_joints]
    prefab.links = [*carrier_links, *original_links]
    all_link_names = [*ARM_LINK_NAMES, *original_link_names]
    prefab.contact_overrides = [
        robotics.BotContactOverride(link_a=a, link_b=b, enable=False)
        for a, b in combinations(all_link_names, 2)
    ]
    prefab.world_from_root = physics.TransformRT()
    prefab.default_pose = np.concatenate(
        (ARM_HOME, np.asarray(prefab.default_pose, dtype=float))
    )

    hand_dofs, digit_dofs = _finger_dof_layout(prefab)
    if len(hand_dofs) != 27:
        raise RuntimeError(f"Expected 27 Meta XR hand DOFs, found {len(hand_dofs)}.")
    arm_dofs = np.arange(ARM_DOF_COUNT, dtype=np.int32)
    arm_limits = np.asarray(
        [
            (-0.45, 0.65),
            (-0.45, 0.45),
            (0.39, 0.85),
            (-math.pi, math.pi),
            (-math.pi / 2, math.pi / 2),
            (-math.pi / 2, math.pi / 2),
        ],
        dtype=float,
    )
    hand_limits = _dof_limits(prefab)[hand_dofs]
    neutral_hand = np.clip(
        np.asarray(prefab.default_pose, dtype=float)[hand_dofs],
        hand_limits[:, 0],
        hand_limits[:, 1],
    )
    prefab.default_pose = np.concatenate((ARM_HOME, neutral_hand))

    bot = robotics.create_bot(scene, prefab, context)
    actor = bot.get_articulated_actor()
    digit_links = {
        "thumb": (
            "bone_02_thumb_metacarpal",
            "bone_03_thumb_proximal",
            "bone_04_thumb_medial",
            "bone_05_thumb_distal",
        ),
        "index": (
            "bone_06_index_proximal",
            "bone_07_index_medial",
            "bone_08_index_distal",
        ),
        "middle": (
            "bone_09_middle_proximal",
            "bone_10_middle_medial",
            "bone_11_middle_distal",
        ),
        "ring": (
            "bone_12_ring_proximal",
            "bone_13_ring_medial",
            "bone_14_ring_distal",
        ),
        "pinky": (
            "bone_15_pinky_metacarpal",
            "bone_16_pinky_proximal",
            "bone_17_pinky_medial",
            "bone_18_pinky_distal",
        ),
    }
    contacts = tuple(
        ContactGroup(digit, digit_links[digit], tuple(digit_dofs[digit]))
        for digit in ("thumb", "index", "middle", "ring", "pinky")
    )

    controlled_names = frozenset(link.name for link in prefab.links)
    tracking: dict[str, JointTrackingSpec] = {}
    for link, joint in zip(prefab.links, prefab.joints):
        if joint.type == physics.ArticulatedJointType.HARD:
            continue
        effort_limit = max(0.25, float(joint.effort_limit))
        if link.name.startswith("bone_"):
            tracking[link.name] = JointTrackingSpec(
                stiffness=0.80 * effort_limit / 0.30,
                damping=0.080,
                saturation=0.30,
            )
        else:
            tracking[link.name] = JointTrackingSpec(
                stiffness=0.95 * effort_limit / 0.05,
                damping=0.60 * effort_limit,
                saturation=0.05,
            )

    info = EmbodimentModel(
        embodiment_id="human_right_hand",
        display_name="Human right hand",
        bot=bot,
        prefab=prefab,
        actor=actor,
        arm_dofs=arm_dofs,
        hand_dofs=hand_dofs,
        arm_limits=arm_limits,
        hand_limits=hand_limits,
        default_pose=np.asarray(prefab.default_pose, dtype=float).copy(),
        end_effector_link="bone_00_wrist_root",
        grasp_point_local=np.zeros(3, dtype=float),
        hand_poses={"home": neutral_hand.copy()},
        contact_groups=contacts,
        hidden_render_link_names=frozenset(ARM_LINK_NAMES),
        controlled_link_names=controlled_names,
        tracking=tracking,
    )
    _register_visuals(scene, info)
    return info


def destroy_human_right_arm(scene: physics.Scene, info: EmbodimentModel) -> None:
    render_model_registry.unregister_actors(scene, info.actor.get_nested_link_actors())
    robotics.destroy_bot(scene, info.bot)


class HumanArmKinematics:
    """Neutral kinematic twin matching the OpenArm scaffold's small interface."""

    def __init__(
        self,
        context: robotics.RoboticsContext,
        reference: EmbodimentModel,
        contact_factory: Callable[[float], physics.ContactParams],
        collision_model: object,
    ) -> None:
        self.scene = physics.create_scene("Human right-hand kinematic scaffold")
        self.info = build_human_right_arm(self.scene, context, contact_factory)
        self.actor = self.info.actor
        self.reference_pose = reference.default_pose.copy()
        self.collision_model = collision_model
        self.link_actors: dict[str, physics.Actor] = {}
        for handle in self.actor.get_nested_link_actors():
            actor = self.scene.get_actor(handle)
            self.link_actors[actor.get_name().split("/", 1)[-1]] = actor
        self.ee = self.link_actors[self.info.end_effector_link]
        self.moving_proxy_actors = [
            actor
            for name, actor in self.link_actors.items()
            if name.startswith("bone_")
        ]
        self._proxies = self._make_proxies(self.moving_proxy_actors)

    @staticmethod
    def _make_proxies(
        actors: list[physics.Actor],
    ) -> list[tuple[physics.Actor, npt.NDArray[np.float64], float]]:
        proxies = []
        for actor in actors:
            bounds = actor.get_aabb_local()
            lo = np.asarray(bounds.min, dtype=float)
            hi = np.asarray(bounds.max, dtype=float)
            center = 0.5 * (lo + hi)
            radius = 0.42 * float(np.linalg.norm(0.5 * (hi - lo))) + 0.002
            proxies.append((actor, center, radius))
        return proxies

    def set_arm_pose(self, arm_pose: npt.ArrayLike) -> None:
        pose = self.reference_pose.copy()
        pose[self.info.arm_dofs] = np.asarray(arm_pose, dtype=float)
        self.actor.set_articulated_pose_from_joints(np.asarray(pose, dtype=MOCHI_REAL))

    def grasp_point_world(self, arm_pose: npt.ArrayLike) -> npt.NDArray[np.float64]:
        self.set_arm_pose(arm_pose)
        transform = self.ee.get_root_transform() * physics.TransformRT(
            translation=self.info.grasp_point_local
        )
        return np.asarray(transform.translation, dtype=float)

    def solve(
        self,
        target: npt.ArrayLike,
        seed: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        """Position the Cartesian carrier while preserving the seed orientation."""
        target_array = np.asarray(target, dtype=float)
        if target_array.shape != (3,):
            raise ValueError("target must be an XYZ vector.")
        arm = np.asarray(seed, dtype=float).copy()
        if arm.shape != (ARM_DOF_COUNT,):
            raise ValueError(f"seed must contain {ARM_DOF_COUNT} carrier coordinates.")
        for _ in range(2):
            arm[:3] += target_array - self.grasp_point_world(arm)
            arm = np.clip(arm, self.info.arm_limits[:, 0], self.info.arm_limits[:, 1])
        error = float(np.linalg.norm(self.grasp_point_world(arm) - target_array))
        if error > 1.0e-5:
            raise RuntimeError(
                f"Human hand Cartesian target was missed by {error:.4f} m "
                "after applying carrier limits."
            )
        return arm

    def world_proxy_spheres(
        self,
        role: str,
    ) -> list[tuple[npt.NDArray[np.float64], float]]:
        if role == "static":
            return []
        result = []
        for actor, local_center, radius in self._proxies:
            transform = actor.get_root_transform() * physics.TransformRT(
                translation=local_center
            )
            result.append((np.asarray(transform.translation, dtype=float), radius))
        return result

    def collision_cost(self, arm_pose: npt.ArrayLike) -> tuple[float, float]:
        return self.collision_model.evaluate(self, arm_pose)

    def close(self) -> None:
        destroy_human_right_arm(self.scene, self.info)
        physics.destroy_scene(self.scene)
