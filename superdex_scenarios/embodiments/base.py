"""Narrow contracts shared by task scenarios and physical embodiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
from superdex import physics, robotics


@dataclass(frozen=True)
class CameraSpec:
    """A render camera exposed by an embodiment or workcell."""

    name: str
    label: str
    kind: str
    intrinsics: dict[str, float | int]
    world_from_camera_cv: dict[str, object] | None = None
    actor_suffix: str | None = None
    parent_from_camera_cv: dict[str, object] | None = None
    calibration_status: str = "modeled"
    provenance: dict[str, object] | str = ""

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "label": self.label,
            "kind": self.kind,
            "intrinsics": self.intrinsics,
            "calibration_status": self.calibration_status,
            "provenance": self.provenance,
        }
        if self.world_from_camera_cv is not None:
            payload["world_from_camera_cv"] = self.world_from_camera_cv
        if self.actor_suffix is not None:
            payload["actor_suffix"] = self.actor_suffix
        if self.parent_from_camera_cv is not None:
            payload["parent_from_camera_cv"] = self.parent_from_camera_cv
        return payload


@dataclass(frozen=True)
class ContactGroup:
    """One logical gripper jaw or human digit assembled from physical links."""

    name: str
    actor_suffixes: tuple[str, ...]
    dofs: tuple[int, ...] = ()


@dataclass(frozen=True)
class JointTrackingSpec:
    """Finite compliant pose-tracking gains for one articulated link."""

    stiffness: float
    damping: float
    saturation: float


@dataclass
class EmbodimentModel:
    """Runtime model metadata needed by otherwise embodiment-neutral code."""

    embodiment_id: str
    display_name: str
    bot: robotics.Bot
    prefab: robotics.BotPrefab
    actor: physics.Actor
    arm_dofs: npt.NDArray[np.int32]
    hand_dofs: npt.NDArray[np.int32]
    arm_limits: npt.NDArray[np.float64]
    hand_limits: npt.NDArray[np.float64]
    default_pose: npt.NDArray[np.float64]
    end_effector_link: str
    grasp_point_local: npt.NDArray[np.float64]
    hand_poses: dict[str, npt.NDArray[np.float64]]
    contact_groups: tuple[ContactGroup, ...]
    approach_direction_world: npt.NDArray[np.float64] | None = None
    pregrasp_distance: float | None = None
    grasp_tolerance_m: float | None = None
    cameras: tuple[CameraSpec, ...] = ()
    hidden_render_link_names: frozenset[str] = field(default_factory=frozenset)
    controlled_link_names: frozenset[str] = field(default_factory=frozenset)
    tracking: dict[str, JointTrackingSpec] = field(default_factory=dict)
    dof_groups: dict[str, npt.NDArray[np.int32]] = field(default_factory=dict)
    """Named DOF groups (``right_arm``, ``right_gripper``, ``left_arm`` ...).

    Trajectory files address joints through these names; see
    :mod:`superdex_scenarios.replay`.
    """
    dof_names: tuple[str, ...] = ()
    """Joint name of every articulation DOF, in actor DOF order."""

    @property
    def controlled_dofs(self) -> npt.NDArray[np.int32]:
        return np.concatenate((self.arm_dofs, self.hand_dofs)).astype(
            np.int32, copy=False
        )

    # Compatibility names used by the original OpenArm scenario API.
    @property
    def right_arm_dofs(self) -> npt.NDArray[np.int32]:
        return self.arm_dofs

    @property
    def right_finger_dofs(self) -> npt.NDArray[np.int32]:
        return self.hand_dofs

    @property
    def right_arm_limits(self) -> npt.NDArray[np.float64]:
        return self.arm_limits

    def target_pose(
        self, arm_pose: npt.ArrayLike, hand_pose: str | npt.ArrayLike
    ) -> npt.NDArray[np.float64]:
        target = self.default_pose.copy()
        target[self.arm_dofs] = np.asarray(arm_pose, dtype=float)
        hand = self.hand_poses[hand_pose] if isinstance(hand_pose, str) else hand_pose
        target[self.hand_dofs] = np.asarray(hand, dtype=float)
        return target

    def link_actor(self, scene: physics.Scene, suffix: str) -> physics.Actor:
        matches = [
            scene.get_actor(handle)
            for handle in self.actor.get_nested_link_actors()
            if scene.get_actor(handle).get_name().endswith(suffix)
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one {self.embodiment_id} link ending in {suffix!r}, "
                f"found {len(matches)}."
            )
        return matches[0]


@runtime_checkable
class ArmKinematics(Protocol):
    """The minimal kinematics contract required by generic TrajOpt."""

    info: EmbodimentModel

    def solve(
        self, target: npt.ArrayLike, seed: npt.ArrayLike
    ) -> npt.NDArray[np.float64]: ...

    def grasp_point_world(self, arm_pose: npt.ArrayLike) -> npt.NDArray[np.float64]: ...

    def collision_cost(self, arm_pose: npt.ArrayLike) -> tuple[float, float]: ...

    def close(self) -> None: ...
