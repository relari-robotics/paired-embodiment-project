# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Randomized ball-and-bowl task shared by every embodiment.

One :class:`BallBowlScenario` holds the task state -- scene, sampled
specification, workcell, the embodiment's physical model and its kinematic
twins.  Nothing in this module knows a particular embodiment: the robot and
the human hand live in :mod:`scenarios.ball_bowl.embodiments`, one subpackage
each, and register themselves in ``embodiments.EMBODIMENTS``.

The ball is a dynamic rigid body, and so is the bowl unless the specification
says ``bowl_static`` (the runner's ``--fixed`` regression scene).  A
specification may name a ``bowl_target_xy``: the bowl then has to be moved
there before the ball goes in (the bimanual variant of the task), and success
also requires the bowl to rest upright at that target.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from superdex import physics, robotics
from superdex.physics.paths import resolve_asset

from superdex_scenarios.assets import bowl as bowl_asset
from superdex_scenarios.embodiments.base import ArmKinematics, EmbodimentModel

from .embodiments import DEFAULT_EMBODIMENT

# Workcell dimensions. SuperDex uses metres and a Z-up world. The robot is at
# the -X edge of the desk and faces +X, so the 47-inch edge runs left-to-right
# along Y and faces the robot; the 24-inch dimension is the desk depth along X.
DESK_SIZE = np.array([24.0 * 0.0254, 47.0 * 0.0254, 0.05], dtype=float)
DESK_MIN = np.array([-0.36, -0.5 * DESK_SIZE[1], 0.338], dtype=float)
DESK_TOP_Z = float(DESK_MIN[2] + DESK_SIZE[2])
DESK_LEG_SIZE = np.array([0.055, 0.055, DESK_MIN[2]], dtype=float)

# The ball keeps a tennis-ball diameter while mass is randomized from 50-500 g.
# The collision model is rigid, so shell inertia and calibrated contact
# restitution approximate the dominant rigid-body response without pretending to
# reproduce visible shell deformation.
BALL_RADIUS = 0.0335
BALL_MASS = 0.0577
BALL_SHELL_INERTIA = (2.0 / 3.0) * BALL_MASS * BALL_RADIUS**2
BALL_TARGET_COR = 0.745
CONTACT_CHARACTERISTIC_SPEED = 1.0  # m/s for this short pick-and-drop task
# Produced by calibrate_normal_viscous_damping_coefficient(0.745, 1.0).
CONTACT_NORMAL_DAMPING = 0.49881
BALL_FRICTION = 0.78
FINGERTIP_FRICTION = 0.95
WOOD_FRICTION = 0.45
CERAMIC_FRICTION = 0.42
CONTACT_SMOOTHING = 0.0005
CONTACT_THRESHOLD = 0.0002
FRICTION_FALLOFF_SPEED = 0.001
BALL_XY = np.array([0.0, -0.23])
BALL_START = np.array(
    [
        BALL_XY[0],
        BALL_XY[1],
        DESK_TOP_Z + BALL_RADIUS + CONTACT_THRESHOLD,
    ]
)

# The bowl is the procedural bowl of superdex_scenarios.assets.bowl, measured
# from the recorded demonstrations (175 mm across, 70 mm tall, flat floor, no
# lip); shape variants scale it.
BOWL_CENTER_XY = np.array([-0.04, 0.02])
BOWL_MASS = 0.25
BOWL_MASS_RANGE = (0.15, 0.40)
BOWL_SCALE = np.array([1.0, 1.0, 1.0])
BOWL_OUTER_RADIUS = bowl_asset.OUTER_RADIUS * BOWL_SCALE[0]
BOWL_RIM_Z = DESK_TOP_Z + bowl_asset.HEIGHT * BOWL_SCALE[2]
BALL_RELEASE_CLEARANCE = 0.02

# The points below refer to GRASP_POINT_EE, which coincides with the ball centre
# while held.  Keeping the same local point for every IK target preserves the
# authored gripper orientation.  The gripper's local -Z approach axis points
# mostly along world +X, so pre-grasp retracts along that axis instead of dropping
# vertically onto the ball.
PICK = BALL_START.copy()
GRIPPER_APPROACH_WORLD = np.array([1.0, 0.0, 0.0])
PRE_PICK = PICK - 0.13 * GRIPPER_APPROACH_WORLD
PICK_LIFT = PICK + np.array([0.0, 0.0, 0.115])
PLACE = np.array(
    [
        BOWL_CENTER_XY[0],
        BOWL_CENTER_XY[1],
        BOWL_RIM_Z + BALL_RADIUS + BALL_RELEASE_CLEARANCE,
    ]
)
PLACE_SAFE = PLACE + np.array([0.0, 0.0, 0.04])

TIME_STEP = 1.0 / 400.0
RENDER_EVERY_STEPS = 6

# Colors used by the built-in viewer.  Visual identity is explicit even when the
# physics collision mesh is selected instead of a GLB render model.
WOOD_COLOR = np.array([0.43, 0.20, 0.075])
BLUE_COLOR = np.array([0.035, 0.20, 0.95])
GRAY_COLOR = np.array([0.43, 0.46, 0.50])
TRAJECTORY_COLOR = np.array([0.0, 0.85, 0.95])


# Conservative, empirically validated part of the right arm's tabletop
# workspace. Sampling in separate Y bands keeps the objects distinct while still
# allowing substantial continuous location variation.
BALL_X_RANGE = (-0.015, 0.030)
BALL_Y_RANGE = (-0.31, -0.19)
BOWL_X_RANGE = (-0.14, 0.015)
BOWL_Y_RANGE = (-0.035, 0.065)
MAX_RANDOMIZATION_ATTEMPTS = 24

# Move-the-bowl variant.  The bowl starts on the robot's left, forward of the
# left gripper's parked home footprint (whose jaws reach world x = -0.065 m at
# rim height), and has to end near the middle of the desk, where both arms
# reach; from there on it is the plain task.
MOVE_BOWL_START_XY = np.array([0.04, 0.20])
MOVE_BOWL_TARGET_XY = np.array([0.0, 0.02])
MOVE_BOWL_START_X_RANGE = (0.03, 0.08)
MOVE_BOWL_START_Y_RANGE = (0.17, 0.23)
MOVE_BOWL_TARGET_X_RANGE = (-0.08, 0.01)
MOVE_BOWL_TARGET_Y_RANGE = (-0.02, 0.05)
BOWL_TARGET_TOLERANCE = 0.06
"""Bowl centre must end within this XY distance of its target."""
BOWL_MAX_TILT_DEG = 15.0


@dataclass(frozen=True)
class ColorChoice:
    """Named RGB color available to the randomized scene."""

    name: str
    rgb: tuple[float, float, float]


@dataclass(frozen=True)
class BowlShapeChoice:
    """An (x, y, z) scale variant of the measured bowl; (1, 1, 1) is the recorded bowl."""

    name: str
    scale: tuple[float, float, float]


BALL_COLORS = (
    ColorChoice("blue", (0.035, 0.20, 0.95)),
    ColorChoice("red", (0.88, 0.055, 0.045)),
    ColorChoice("yellow", (0.96, 0.72, 0.035)),
    ColorChoice("green", (0.055, 0.62, 0.22)),
    ColorChoice("orange", (0.96, 0.30, 0.035)),
    ColorChoice("purple", (0.47, 0.12, 0.78)),
)
BOWL_COLORS = (
    ColorChoice("gray", (0.43, 0.46, 0.50)),
    ColorChoice("ivory", (0.88, 0.84, 0.72)),
    ColorChoice("teal", (0.035, 0.42, 0.43)),
    ColorChoice("terracotta", (0.62, 0.20, 0.095)),
    ColorChoice("navy", (0.045, 0.11, 0.32)),
    ColorChoice("mustard", (0.72, 0.49, 0.055)),
)
BOWL_SHAPES = (
    BowlShapeChoice("measured", (1.00, 1.00, 1.00)),
    BowlShapeChoice("wide", (1.15, 1.15, 0.90)),
    BowlShapeChoice("deep", (0.95, 0.95, 1.25)),
    BowlShapeChoice("oval", (1.15, 0.95, 1.00)),
)


@dataclass(frozen=True)
class ScenarioSpecification:
    """All randomized values needed to reproduce one task instance exactly."""

    seed: int | None
    sample_attempt: int
    ball_mass_kg: float
    ball_color: ColorChoice
    bowl_color: ColorChoice
    bowl_shape: BowlShapeChoice
    ball_xy: tuple[float, float]
    bowl_xy: tuple[float, float]
    """Bowl centre at the start of the episode (world XY)."""
    bowl_mass_kg: float = BOWL_MASS
    bowl_target_xy: tuple[float, float] | None = None
    """Where the bowl has to be moved before the ball goes in; ``None`` leaves it in place."""
    bowl_static: bool = False
    """Bolt the bowl to the desk, as in the original regression scene."""

    def __post_init__(self) -> None:
        if self.bowl_static and self.bowl_target_xy is not None:
            raise ValueError("A static bowl cannot be moved to a bowl_target_xy.")

    @classmethod
    def fixed(
        cls, *, move_bowl: bool = False, bowl_static: bool = False
    ) -> ScenarioSpecification:
        """Return the deterministic layout.

        ``move_bowl`` selects the bimanual variant: the bowl starts on the
        robot's left and has a target near the middle of the desk.
        ``bowl_static`` bolts the bowl down; the bowl is movable otherwise.
        """
        bowl_xy, bowl_target_xy = BOWL_CENTER_XY, None
        if move_bowl:
            bowl_xy = MOVE_BOWL_START_XY
            bowl_target_xy = (float(MOVE_BOWL_TARGET_XY[0]), float(MOVE_BOWL_TARGET_XY[1]))
        return cls(
            seed=None,
            sample_attempt=1,
            ball_mass_kg=BALL_MASS,
            ball_color=BALL_COLORS[0],
            bowl_color=BOWL_COLORS[0],
            bowl_shape=BOWL_SHAPES[0],
            ball_xy=(float(BALL_XY[0]), float(BALL_XY[1])),
            bowl_xy=(float(bowl_xy[0]), float(bowl_xy[1])),
            bowl_target_xy=bowl_target_xy,
            bowl_static=bowl_static,
        )

    @classmethod
    def from_layout(cls, layout: dict[str, Any]) -> ScenarioSpecification:
        """Build the fixed scenario with placements/values overridden by a layout file.

        Recognised keys: ``ball_xy`` and ``bowl_xy`` (metres, world XY on the
        desk; ``bowl_xy`` is where the bowl starts), ``bowl_target_xy`` (where
        the bowl has to be moved; its presence selects the move-the-bowl
        variant and its defaults), ``ball_mass_kg``, ``bowl_mass_kg``,
        ``ball_color`` and ``bowl_color`` (names from
        ``BALL_COLORS``/``BOWL_COLORS``), ``bowl_shape`` (a ``BOWL_SHAPES``
        name), and ``bowl_static`` (``true`` bolts the bowl down; it is movable
        otherwise).  Unknown keys are rejected so typos do not pass silently.
        """
        known = {
            "ball_xy",
            "bowl_xy",
            "bowl_target_xy",
            "ball_mass_kg",
            "bowl_mass_kg",
            "bowl_static",
            "ball_color",
            "bowl_color",
            "bowl_shape",
        }
        unknown = sorted(set(layout) - known - {"comment", "source"})
        if unknown:
            raise ValueError(
                f"Unknown ball_bowl layout keys: {unknown}; expected {sorted(known)}."
            )
        base = cls.fixed(move_bowl="bowl_target_xy" in layout)

        def color(table: tuple[ColorChoice, ...], name: str) -> ColorChoice:
            for choice in table:
                if choice.name == name:
                    return choice
            raise ValueError(
                f"Unknown color {name!r}; expected one of {[c.name for c in table]}."
            )

        bowl_shape = base.bowl_shape
        if "bowl_shape" in layout:
            matches = [s for s in BOWL_SHAPES if s.name == layout["bowl_shape"]]
            if not matches:
                raise ValueError(
                    f"Unknown bowl_shape {layout['bowl_shape']!r}; expected one of "
                    f"{[s.name for s in BOWL_SHAPES]}."
                )
            bowl_shape = matches[0]

        def xy(key: str, default: tuple[float, float]) -> tuple[float, float]:
            value = layout.get(key, default)
            value = np.asarray(value, dtype=float).reshape(-1)
            if value.shape != (2,):
                raise ValueError(f"{key} must be [x, y] in metres.")
            return (float(value[0]), float(value[1]))

        return cls(
            seed=None,
            sample_attempt=1,
            ball_mass_kg=float(layout.get("ball_mass_kg", base.ball_mass_kg)),
            ball_color=color(BALL_COLORS, layout["ball_color"])
            if "ball_color" in layout
            else base.ball_color,
            bowl_color=color(BOWL_COLORS, layout["bowl_color"])
            if "bowl_color" in layout
            else base.bowl_color,
            bowl_shape=bowl_shape,
            ball_xy=xy("ball_xy", base.ball_xy),
            bowl_xy=xy("bowl_xy", base.bowl_xy),
            bowl_mass_kg=float(layout.get("bowl_mass_kg", base.bowl_mass_kg)),
            bowl_target_xy=xy("bowl_target_xy", base.bowl_target_xy)
            if base.bowl_target_xy is not None
            else None,
            bowl_static=bool(layout.get("bowl_static", False)),
        )

    @property
    def ball_start(self) -> npt.NDArray[np.float64]:
        return np.array(
            [*self.ball_xy, DESK_TOP_Z + BALL_RADIUS + CONTACT_THRESHOLD],
            dtype=float,
        )

    @property
    def bowl_start(self) -> npt.NDArray[np.float64]:
        # A movable bowl rests one contact threshold above the desk it sits on.
        clearance = 0.0 if self.bowl_static else CONTACT_THRESHOLD
        return np.array([*self.bowl_xy, DESK_TOP_Z + clearance], dtype=float)

    @property
    def delivery_xy(self) -> tuple[float, float]:
        """Where the bowl is when the ball is delivered: its target, else its start."""
        return self.bowl_xy if self.bowl_target_xy is None else self.bowl_target_xy

    @property
    def bowl_scale(self) -> npt.NDArray[np.float64]:
        return np.asarray(self.bowl_shape.scale, dtype=float)

    @property
    def bowl_outer_radii(self) -> npt.NDArray[np.float64]:
        return bowl_asset.OUTER_RADIUS * self.bowl_scale[:2]

    @property
    def bowl_rim_height(self) -> float:
        """Rim height above the bowl's base (its local origin sits on the desk)."""
        return bowl_asset.HEIGHT * float(self.bowl_scale[2])

    @property
    def bowl_rim_z(self) -> float:
        return DESK_TOP_Z + self.bowl_rim_height

    @property
    def pick(self) -> npt.NDArray[np.float64]:
        return self.ball_start

    @property
    def pre_pick(self) -> npt.NDArray[np.float64]:
        return self.pick - 0.13 * GRIPPER_APPROACH_WORLD

    @property
    def pick_lift(self) -> npt.NDArray[np.float64]:
        return self.pick + np.array([0.0, 0.0, 0.115])

    @property
    def place(self) -> npt.NDArray[np.float64]:
        release_z = self.bowl_rim_z + BALL_RADIUS + BALL_RELEASE_CLEARANCE
        return np.array([*self.delivery_xy, release_z], dtype=float)

    @property
    def place_safe(self) -> npt.NDArray[np.float64]:
        return np.array([*self.delivery_xy, self.place[2] + 0.04], dtype=float)

    @property
    def ball_shell_inertia(self) -> float:
        return (2.0 / 3.0) * self.ball_mass_kg * BALL_RADIUS**2

    def contains_ball(
        self, ball_position: npt.ArrayLike, bowl_position: npt.ArrayLike | None = None
    ) -> bool:
        """Ball inside the elliptical rim of the bowl *as it actually sits*.

        The bowl is dynamic, so pass its measured ``bowl_position``; without
        one the bowl is assumed to rest where the ball is delivered.
        """
        ball = np.asarray(ball_position, dtype=float)
        bowl = (
            np.array([*self.delivery_xy, DESK_TOP_Z], dtype=float)
            if bowl_position is None
            else np.asarray(bowl_position, dtype=float)
        )
        normalized_xy = (ball[:2] - bowl[:2]) / self.bowl_outer_radii
        return bool(
            np.dot(normalized_xy, normalized_xy) < 1.0
            and ball[2] < bowl[2] + self.bowl_rim_height + 2.0 * BALL_RADIUS
            and ball[2] > bowl[2] - 0.01
        )

    def bowl_at_target(
        self,
        bowl_position: npt.ArrayLike,
        bowl_quaternion_xyzw: npt.ArrayLike | None = None,
    ) -> bool:
        """Bowl centre within tolerance of its target, resting upright on the desk."""
        if self.bowl_target_xy is None:
            raise ValueError("This specification has no bowl target.")
        position = np.asarray(bowl_position, dtype=float)
        offset = position[:2] - np.asarray(self.bowl_target_xy)
        if np.linalg.norm(offset) > BOWL_TARGET_TOLERANCE:
            return False
        if abs(position[2] - DESK_TOP_Z) > 0.02:
            return False
        return (
            bowl_quaternion_xyzw is None
            or bowl_tilt_deg(bowl_quaternion_xyzw) <= BOWL_MAX_TILT_DEG
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable reproducibility record."""
        return {
            "seed": self.seed,
            "sample_attempt": self.sample_attempt,
            "ball": {
                "mass_kg": self.ball_mass_kg,
                "radius_m": BALL_RADIUS,
                "color": self.ball_color.name,
                "color_rgb": list(self.ball_color.rgb),
                "position_m": self.ball_start.tolist(),
            },
            "bowl": {
                "shape": self.bowl_shape.name,
                "scale_xyz": list(self.bowl_shape.scale),
                "color": self.bowl_color.name,
                "color_rgb": list(self.bowl_color.rgb),
                "mass_kg": None if self.bowl_static else self.bowl_mass_kg,
                "dynamic": not self.bowl_static,
                "position_m": self.bowl_start.tolist(),
                "outer_radii_m": self.bowl_outer_radii.tolist(),
                "rim_height_m": self.bowl_rim_height,
                "rim_z_m": self.bowl_rim_z,
            },
            "targets": {
                "pick_m": self.pick.tolist(),
                "place_m": self.place.tolist(),
                **(
                    {}
                    if self.bowl_target_xy is None
                    else {
                        "bowl_target_m": [*self.bowl_target_xy, DESK_TOP_Z],
                        "bowl_target_tolerance_m": BOWL_TARGET_TOLERANCE,
                    }
                ),
            },
        }


def bowl_tilt_deg(quaternion_xyzw: npt.ArrayLike) -> float:
    """Angle between the bowl's local Z axis and world up, in degrees."""
    x, y, _, _ = np.asarray(quaternion_xyzw, dtype=float)
    up_z = 1.0 - 2.0 * (x * x + y * y)
    return float(np.degrees(np.arccos(np.clip(up_z, -1.0, 1.0))))


def sample_specification(
    rng: np.random.Generator, seed: int, attempt: int, *, move_bowl: bool = False
) -> ScenarioSpecification:
    """Sample one candidate while enforcing tabletop and object separation.

    ``move_bowl`` samples the bimanual variant: the measured bowl starts in the
    left arm's band and gets a target in the band both arms reach.
    """
    shapes = BOWL_SHAPES[:1] if move_bowl else BOWL_SHAPES
    bowl_shape = shapes[int(rng.integers(len(shapes)))]
    outer_radii = bowl_asset.OUTER_RADIUS * np.asarray(
        bowl_shape.scale[:2], dtype=float
    )
    minimum_separation = float(np.max(outer_radii)) + BALL_RADIUS + 0.045
    bowl_x_range, bowl_y_range = (
        (MOVE_BOWL_START_X_RANGE, MOVE_BOWL_START_Y_RANGE)
        if move_bowl
        else (BOWL_X_RANGE, BOWL_Y_RANGE)
    )
    for _ in range(128):
        ball_xy = np.array(
            [rng.uniform(*BALL_X_RANGE), rng.uniform(*BALL_Y_RANGE)], dtype=float
        )
        bowl_xy = np.array(
            [rng.uniform(*bowl_x_range), rng.uniform(*bowl_y_range)], dtype=float
        )
        target_xy = None
        if move_bowl:
            target_xy = np.array(
                [
                    rng.uniform(*MOVE_BOWL_TARGET_X_RANGE),
                    rng.uniform(*MOVE_BOWL_TARGET_Y_RANGE),
                ],
                dtype=float,
            )
            if np.linalg.norm(bowl_xy - target_xy) < 0.12:
                continue
        # The ball must stay clear of the bowl wherever the ball is delivered.
        delivery_xy = bowl_xy if target_xy is None else target_xy
        if np.linalg.norm(ball_xy - delivery_xy) >= minimum_separation:
            break
    else:
        raise RuntimeError("Could not sample separated ball and bowl locations.")

    mass_step = int(rng.integers(1, 11))
    ball_color = BALL_COLORS[int(rng.integers(len(BALL_COLORS)))]
    bowl_color = BOWL_COLORS[int(rng.integers(len(BOWL_COLORS)))]
    # Drawn last so a seed keeps the layout and colours it had with a static bowl.
    bowl_mass_kg = round(float(rng.uniform(*BOWL_MASS_RANGE)), 3)
    return ScenarioSpecification(
        seed=seed,
        sample_attempt=attempt,
        ball_mass_kg=round(mass_step * 0.05, 2),
        ball_color=ball_color,
        bowl_color=bowl_color,
        bowl_shape=bowl_shape,
        ball_xy=(float(ball_xy[0]), float(ball_xy[1])),
        bowl_xy=(float(bowl_xy[0]), float(bowl_xy[1])),
        bowl_mass_kg=bowl_mass_kg,
        bowl_target_xy=None
        if target_xy is None
        else (float(target_xy[0]), float(target_xy[1])),
    )


BotInfo = EmbodimentModel


@dataclass
class Workcell:
    desk_actors: list[physics.Actor]
    bowl: physics.Actor
    ball: physics.Actor


def contact_params(friction: float) -> physics.ContactParams:
    """Contact settings shared by a physical material in this workcell."""
    return physics.ContactParams(
        penalty_smoothing_half_distance=CONTACT_SMOOTHING,
        penalty_threshold_default=CONTACT_THRESHOLD,
        viscous_friction_coefficient=0.0,
        coulomb_friction_coefficient=friction,
        friction_falloff_vel=FRICTION_FALLOFF_SPEED,
        normal_viscous_damping_coefficient=CONTACT_NORMAL_DAMPING,
    )


def _load_shape(asset: str, scale: npt.ArrayLike) -> physics.ShapeHandle:
    return physics.load_shape_from_file(
        file_path=str(resolve_asset(asset)), bake_scale=np.asarray(scale, dtype=float)
    )


def create_workcell(
    scene: physics.Scene,
    specification: ScenarioSpecification | None = None,
) -> Workcell:
    """Build the wooden desk, the bowl (dynamic unless ``bowl_static``), and the dynamic ball."""
    specification = specification or ScenarioSpecification.fixed()
    block_asset = "prefabs/box_and_blocks/collision/block.mochi.h5"
    # The source block is a 25 mm cube with its local origin at a lower corner.
    top_shape = _load_shape(block_asset, DESK_SIZE / 0.025)
    top = scene.create_rigid_actor(
        name="wooden_desk/top",
        layer="desk",
        shape=top_shape,
        is_static=True,
        contact=contact_params(WOOD_FRICTION),
        world_from_local=physics.TransformRT(translation=DESK_MIN),
    )
    physics.release_shape(top_shape)

    leg_shape = _load_shape(block_asset, DESK_LEG_SIZE / 0.025)
    desk_actors = [top]
    for x in (DESK_MIN[0] + 0.035, DESK_MIN[0] + DESK_SIZE[0] - 0.09):
        for y in (DESK_MIN[1] + 0.035, DESK_MIN[1] + DESK_SIZE[1] - 0.09):
            leg = scene.create_rigid_actor(
                name=f"wooden_desk/leg_{len(desk_actors)}",
                layer="desk",
                shape=leg_shape,
                is_static=True,
                contact=contact_params(WOOD_FRICTION),
                world_from_local=physics.TransformRT(translation=[x, y, 0.0]),
            )
            desk_actors.append(leg)
    physics.release_shape(leg_shape)

    bowl_vertices, bowl_faces = bowl_asset.mesh(specification.bowl_scale)
    bowl_shape = physics.create_tri_mesh_shape(
        bowl_vertices.ravel(), bowl_faces.ravel().astype(np.int32)
    )
    bowl_body: dict[str, Any] = (
        {"is_static": True}
        if specification.bowl_static
        else {
            "collider_type": physics.ColliderType.SDF,
            "is_static": False,
            "mass": specification.bowl_mass_kg,
        }
    )
    bowl = scene.create_rigid_actor(
        name="gray_bowl",
        layer="bowl",
        shape=bowl_shape,
        contact=contact_params(CERAMIC_FRICTION),
        world_from_local=physics.TransformRT(translation=specification.bowl_start),
        **bowl_body,
    )
    physics.release_shape(bowl_shape)

    # The authored sphere radius is 15 mm.
    sphere_scale = np.full(3, BALL_RADIUS / 0.015, dtype=float)
    ball_shape = _load_shape("prefabs/sphere/collision/sphere.mochi.h5", sphere_scale)
    ball = scene.create_rigid_actor(
        name="blue_ball",
        layer="ball",
        shape=ball_shape,
        collider_type=physics.ColliderType.SPHERE,
        mass=specification.ball_mass_kg,
        center_of_mass=[0.0, 0.0, 0.0],
        moment_of_inertia=[
            specification.ball_shell_inertia,
            0.0,
            0.0,
            specification.ball_shell_inertia,
            0.0,
            specification.ball_shell_inertia,
        ],
        contact=contact_params(BALL_FRICTION),
        world_from_local=physics.TransformRT(translation=specification.ball_start),
    )
    physics.release_shape(ball_shape)
    return Workcell(desk_actors=desk_actors, bowl=bowl, ball=ball)


@dataclass(frozen=True)
class EmbodimentSpec:
    """How to build one embodiment into the shared workcell, and who drives it."""

    embodiment_id: str
    scene_name: str
    render_manifest: str
    solver_max_iter: int
    build: Callable[[physics.Scene, robotics.RoboticsContext], EmbodimentModel]
    destroy: Callable[[physics.Scene, EmbodimentModel], None]
    kinematics: Callable[
        [robotics.RoboticsContext, EmbodimentModel, ScenarioSpecification],
        ArmKinematics,
    ]
    policy: str
    """``"module:Class"`` implementing :class:`episode.EpisodePolicy`."""
    left_kinematics: (
        Callable[
            [robotics.RoboticsContext, EmbodimentModel, ScenarioSpecification],
            ArmKinematics,
        ]
        | None
    ) = None
    """Second kinematic twin of a two-armed embodiment."""
    move_bowl_policy: str | None = None
    """Policy for a specification with a ``bowl_target_xy``; needs two arms."""

    def policy_for(self, specification: ScenarioSpecification) -> str:
        """The registered policy that solves ``specification`` with this embodiment."""
        if specification.bowl_target_xy is None:
            return self.policy
        if self.move_bowl_policy is None:
            raise ValueError(
                f"Embodiment {self.embodiment_id!r} has no policy that moves the bowl; "
                "drop bowl_target_xy or use a two-armed embodiment."
            )
        return self.move_bowl_policy


def embodiment_spec(embodiment_id: str) -> EmbodimentSpec:
    """Look an embodiment up in :data:`embodiments.EMBODIMENTS`."""
    # Imported here: every embodiment subpackage imports this module.
    from .embodiments import EMBODIMENTS

    return EMBODIMENTS[embodiment_id]


@dataclass
class BallBowlScenario:
    """Built task state for one embodiment, independent of any controller or policy."""

    scene: physics.Scene
    specification: ScenarioSpecification
    bot_info: EmbodimentModel
    workcell: Workcell
    kinematics: ArmKinematics
    embodiment: EmbodimentSpec
    left_kinematics: ArmKinematics | None = None
    _closed: bool = field(default=False, init=False, repr=False)

    @property
    def embodiment_id(self) -> str:
        return self.embodiment.embodiment_id

    @property
    def camera_names(self) -> tuple[str, ...]:
        from .cameras import camera_specs

        return tuple(camera.name for camera in camera_specs(self.bot_info))

    @classmethod
    def build(
        cls,
        context: robotics.RoboticsContext,
        embodiment_id: str = DEFAULT_EMBODIMENT,
        specification: ScenarioSpecification | None = None,
    ) -> BallBowlScenario:
        """Construct one specified scene: embodiment, ground, workcell, kinematic twins."""
        embodiment = embodiment_spec(embodiment_id)
        specification = specification or ScenarioSpecification.fixed()
        scene = physics.create_scene(embodiment.scene_name)
        scene.set_gravity([0.0, 0.0, -9.80665])
        solver = scene.get_solver_params()
        solver.integration_method = physics.IntegrationMethod.BDF2
        solver.non_linear_solver.max_iter = embodiment.solver_max_iter
        scene.set_solver_params(solver)
        info: EmbodimentModel | None = None
        kinematics: ArmKinematics | None = None
        left_kinematics: ArmKinematics | None = None
        try:
            info = embodiment.build(scene, context)
            ground_shape = physics.create_plane_shape(
                normal=[0.0, 0.0, 1.0], distance=0.0
            )
            scene.create_rigid_actor(
                name="ground",
                layer="ground",
                shape=ground_shape,
                is_static=True,
                contact=contact_params(0.60),
            )
            physics.release_shape(ground_shape)
            workcell = create_workcell(scene, specification)
            kinematics = embodiment.kinematics(context, info, specification)
            if embodiment.left_kinematics is not None:
                left_kinematics = embodiment.left_kinematics(
                    context, info, specification
                )
            return cls(
                scene,
                specification,
                info,
                workcell,
                kinematics,
                embodiment,
                left_kinematics,
            )
        except Exception:
            if left_kinematics is not None:
                left_kinematics.close()
            if kinematics is not None:
                kinematics.close()
            if info is not None:
                embodiment.destroy(scene, info)
            physics.destroy_scene(scene)
            raise

    @classmethod
    def build_randomized(
        cls,
        context: robotics.RoboticsContext,
        seed: int,
        embodiment_id: str = DEFAULT_EMBODIMENT,
        *,
        planner: Callable[[BallBowlScenario], Any] | None = None,
        move_bowl: bool = False,
    ) -> tuple[BallBowlScenario, Any]:
        """Sample a task, build it, and plan it; resample while planning fails.

        ``move_bowl`` samples the variant whose bowl has to be moved to a target.

        ``planner`` receives the built scenario and returns the planned policy;
        it raises :class:`episode.PlanningError` for an infeasible sample, in
        which case the scene is destroyed and the next sample is tried (up to
        ``MAX_RANDOMIZATION_ATTEMPTS`` per seed).  Returns ``(scenario, policy)``.
        """
        from .episode import PlanningError

        rng = np.random.default_rng(seed)
        last_error: Exception | None = None
        for attempt in range(1, MAX_RANDOMIZATION_ATTEMPTS + 1):
            specification = sample_specification(
                rng, seed, attempt, move_bowl=move_bowl
            )
            scenario = cls.build(context, embodiment_id, specification)
            if planner is None:
                return scenario, None
            try:
                return scenario, planner(scenario)
            except PlanningError as error:
                last_error = error
                scenario.close()
                print(
                    f"Random sample {attempt} was not plannable ({error}); resampling."
                )
            except Exception:
                scenario.close()
                raise
        raise RuntimeError(
            "Could not plan a randomized scenario after "
            f"{MAX_RANDOMIZATION_ATTEMPTS} attempts for seed {seed}."
        ) from last_error

    def close(self) -> None:
        """Release scenario-owned planning and embodiment resources once."""
        if self._closed:
            return
        if self.left_kinematics is not None:
            self.left_kinematics.close()
        self.kinematics.close()
        self.embodiment.destroy(self.scene, self.bot_info)
        physics.destroy_scene(self.scene)
        self._closed = True
