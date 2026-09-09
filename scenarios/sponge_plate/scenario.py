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

"""Randomized sponge-and-plate wiping task.

A soft finite-element sponge lies on the desk next to a rigid, scanned plate.
The embodiment picks the sponge up, presses it onto the dish floor, wipes the
floor in a serpentine of straight passes, lifts off, puts the sponge back where
it started, and returns home.  The plate carries a cleanliness map (see
:mod:`cleanliness`): a grid of cells over the dish floor that count as cleaned
once the sponge has slid over them under contact pressure.

The module mirrors :mod:`scenarios.ball_bowl.scenario`: one
:class:`SpongePlateScenario` holds the built task state and embodiments are
registered in :data:`EMBODIMENTS` with the policy class that drives them.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from superdex import physics, robotics

from scenarios.ball_bowl.scenario import (
    CERAMIC_FRICTION,
    CONTACT_THRESHOLD,
    DESK_LEG_SIZE,
    DESK_MIN,
    DESK_SIZE,
    DESK_TOP_Z,
    RENDER_EVERY_STEPS,
    TIME_STEP,
    WOOD_COLOR,
    WOOD_FRICTION,
    ColorChoice,
    _load_shape,
    contact_params,
)
from superdex_scenarios.embodiments.base import ArmKinematics, EmbodimentModel
from superdex_scenarios.embodiments.openarm_v2 import (
    OpenArmKinematics,
    build_openarm_v2,
    destroy_openarm_v2,
)
from superdex_scenarios.planning import TrajOptTrajectoryOptimizer

from .cleanliness import (
    CLEAN_MIN_NORMAL_FORCE,
    CLEAN_MIN_SLIDING_SPEED,
    CLEAN_SUCCESS_COVERAGE,
    DIRT_CELL,
    DIRT_RADIUS,
    CleanlinessMap,
)
from .collision import SpongePlateCollisionModel
from .plate import SolidPlate, solidify_plate

SCENARIO_ROOT = Path(__file__).resolve().parent
PLATE_ASSET = SCENARIO_ROOT / "assets" / "ycb_029_plate" / "nontextured.stl"

# Plate: a solidified YCB 029 scan (about 26 cm across).  The dish floor is flat
# to roughly 8 cm from the centre; the rim rises beyond 9 cm.  The cleanliness
# map (see cleanliness.py) covers the inner 7 cm so the sponge's corners can
# ride the gentle slope.
PLATE_XY = np.array([-0.05, 0.0])
PLATE_X_RANGE = (-0.080, -0.030)
PLATE_Y_RANGE = (-0.025, 0.020)

# Sponge: a neo-Hookean tetrahedral block.  X is its length, Y its width (the
# gripper's jaw axis), Z its height.  Young's modulus is randomized over a soft
# cellulose range; density gives about 30 g.
SPONGE_SIZE = np.array([0.10, 0.065, 0.045])
SPONGE_CELL = 0.01
SPONGE_YOUNGS_MODULUS = 8.0e3
SPONGE_YOUNGS_MODULUS_RANGE = (5.0e3, 1.5e4)
SPONGE_POISSON_RATIO = 0.30
SPONGE_DENSITY = 100.0
SPONGE_FRICTION = 0.85
SPONGE_XY = np.array([0.0, -0.23])
SPONGE_X_RANGE = (-0.015, 0.030)
SPONGE_Y_RANGE = (-0.27, -0.21)

# Grasp and wipe geometry.  The gripper pinches the upper half of the sponge
# (GRASP_UP above its centre) with the nose pitched down so the jaw pads stay
# above the dish while the sponge bottom is pressed PRESS_DEPTH into the floor.
GRASP_UP = 0.030
APPROACH_PITCH = np.radians(30.0)
APPROACH_DISTANCE = 0.13
LIFT_HEIGHT = 0.10
HOVER_HEIGHT = 0.05
PRESS_DEPTH = 0.006
WIPE_PASS_OFFSETS_Y = (-0.040, 0.0, 0.040)
WIPE_HALF_LENGTH_X = 0.030
FINGERS_WIPE_CLOSED = -0.20

MAX_RANDOMIZATION_ATTEMPTS = 24

PLATE_COLOR = np.array([0.92, 0.91, 0.86])
TRAJECTORY_COLOR = np.array([0.0, 0.85, 0.95])

SPONGE_COLORS = (
    ColorChoice("yellow", (0.96, 0.80, 0.15)),
    ColorChoice("green", (0.20, 0.65, 0.25)),
    ColorChoice("blue", (0.15, 0.40, 0.85)),
    ColorChoice("pink", (0.92, 0.35, 0.55)),
    ColorChoice("orange", (0.95, 0.45, 0.10)),
)


def approach_direction_world() -> npt.NDArray[np.float64]:
    """Unit vector along the pitched gripper's approach axis (nose direction)."""
    return np.array([np.cos(APPROACH_PITCH), 0.0, -np.sin(APPROACH_PITCH)])


_PLATE_CACHE: dict[str, SolidPlate] = {}


def plate_model() -> SolidPlate:
    """The solidified plate mesh and heightfield, built once per process."""
    key = str(PLATE_ASSET)
    if key not in _PLATE_CACHE:
        _PLATE_CACHE[key] = solidify_plate(PLATE_ASSET)
    return _PLATE_CACHE[key]


@dataclass(frozen=True)
class ScenarioSpecification:
    """All randomized values needed to reproduce one task instance exactly."""

    seed: int | None
    sample_attempt: int
    sponge_youngs_modulus_pa: float
    sponge_color: ColorChoice
    sponge_xy: tuple[float, float]
    plate_xy: tuple[float, float]

    @classmethod
    def fixed(cls) -> ScenarioSpecification:
        """Deterministic regression configuration."""
        return cls(
            seed=None,
            sample_attempt=1,
            sponge_youngs_modulus_pa=SPONGE_YOUNGS_MODULUS,
            sponge_color=SPONGE_COLORS[0],
            sponge_xy=(float(SPONGE_XY[0]), float(SPONGE_XY[1])),
            plate_xy=(float(PLATE_XY[0]), float(PLATE_XY[1])),
        )

    # -- plate geometry (world) -------------------------------------------

    @property
    def plate_origin(self) -> npt.NDArray[np.float64]:
        """World position of the plate-local origin (its lowest point sits on the desk)."""
        return np.array([*self.plate_xy, DESK_TOP_Z], dtype=float)

    @property
    def plate_center_xy(self) -> npt.NDArray[np.float64]:
        return np.asarray(self.plate_xy, dtype=float)

    @property
    def plate_rim_radius(self) -> float:
        return plate_model().outer_radius

    @property
    def plate_rim_z(self) -> float:
        return DESK_TOP_Z + plate_model().top_z

    @property
    def dish_floor_z(self) -> float:
        return DESK_TOP_Z + plate_model().floor_height(DIRT_RADIUS)

    def plate_surface_z(self, x: float, y: float) -> float:
        """World height of the plate's top surface under a world XY."""
        local = plate_model().center_xy + (np.array([x, y]) - self.plate_center_xy)
        return DESK_TOP_Z + plate_model().surface_height(float(local[0]), float(local[1]))

    # -- sponge and grasp targets (world, grasp point = jaw midpoint) ------

    @property
    def sponge_start(self) -> npt.NDArray[np.float64]:
        """World position of the sponge centre when it rests on the desk."""
        return np.array(
            [*self.sponge_xy, DESK_TOP_Z + 0.5 * SPONGE_SIZE[2] + CONTACT_THRESHOLD],
            dtype=float,
        )

    @property
    def pick(self) -> npt.NDArray[np.float64]:
        return self.sponge_start + np.array([0.0, 0.0, GRASP_UP])

    @property
    def pre_pick(self) -> npt.NDArray[np.float64]:
        return self.pick - APPROACH_DISTANCE * approach_direction_world()

    @property
    def pick_lift(self) -> npt.NDArray[np.float64]:
        return self.pick + np.array([0.0, 0.0, LIFT_HEIGHT])

    def wipe_grasp_z(self, x: float, y: float) -> float:
        """Grasp-point height that presses the sponge PRESS_DEPTH into the surface at XY."""
        surface = self.plate_surface_z(x, y)
        if not np.isfinite(surface):
            surface = self.dish_floor_z
        return surface + 0.5 * SPONGE_SIZE[2] - PRESS_DEPTH + GRASP_UP

    @property
    def wipe_passes(self) -> list[npt.NDArray[np.float64]]:
        """Serpentine wiping passes: each is a (2, 3) start/end of grasp positions."""
        passes = []
        for index, offset in enumerate(WIPE_PASS_OFFSETS_Y):
            y = self.plate_center_xy[1] + offset
            xs = self.plate_center_xy[0] + np.array([-WIPE_HALF_LENGTH_X, WIPE_HALF_LENGTH_X])
            if index % 2:
                xs = xs[::-1]
            passes.append(
                np.array([[x, y, self.wipe_grasp_z(x, y)] for x in xs], dtype=float)
            )
        return passes

    @property
    def wipe_start(self) -> npt.NDArray[np.float64]:
        return self.wipe_passes[0][0]

    @property
    def wipe_end(self) -> npt.NDArray[np.float64]:
        return self.wipe_passes[-1][-1]

    @property
    def wipe_hover(self) -> npt.NDArray[np.float64]:
        return self.wipe_start + np.array([0.0, 0.0, HOVER_HEIGHT])

    @property
    def wipe_lift_off(self) -> npt.NDArray[np.float64]:
        return self.wipe_end + np.array([0.0, 0.0, HOVER_HEIGHT])

    @property
    def place(self) -> npt.NDArray[np.float64]:
        return self.pick

    @property
    def place_safe(self) -> npt.NDArray[np.float64]:
        return self.pick_lift

    # -- success criteria --------------------------------------------------

    def sponge_returned(self, position: npt.ArrayLike) -> bool:
        """Sponge centre rests on the desk near where it started."""
        position = np.asarray(position, dtype=float)
        return bool(
            np.linalg.norm(position[:2] - np.asarray(self.sponge_xy)) < 0.06
            and abs(position[2] - self.sponge_start[2]) < 0.015
        )

    def to_dict(self) -> dict[str, Any]:
        model = plate_model()
        return {
            "seed": self.seed,
            "sample_attempt": self.sample_attempt,
            "sponge": {
                "size_m": SPONGE_SIZE.tolist(),
                "youngs_modulus_pa": self.sponge_youngs_modulus_pa,
                "poisson_ratio": SPONGE_POISSON_RATIO,
                "density_kg_m3": SPONGE_DENSITY,
                "color": self.sponge_color.name,
                "color_rgb": list(self.sponge_color.rgb),
                "position_m": self.sponge_start.tolist(),
            },
            "plate": {
                "asset": "ycb_029_plate",
                "position_m": self.plate_origin.tolist(),
                "outer_radius_m": model.outer_radius,
                "rim_z_m": self.plate_rim_z,
                "dish_floor_z_m": self.dish_floor_z,
                "dirt_radius_m": DIRT_RADIUS,
                "dirt_cell_m": DIRT_CELL,
            },
            "targets": {
                "pick_m": self.pick.tolist(),
                "wipe_passes_m": [segment.tolist() for segment in self.wipe_passes],
                "press_depth_m": PRESS_DEPTH,
                "place_m": self.place.tolist(),
            },
        }


def sample_specification(
    rng: np.random.Generator, seed: int, attempt: int
) -> ScenarioSpecification:
    """Sample one candidate task while keeping the sponge clear of the plate."""
    rim = plate_model().outer_radius
    for _ in range(128):
        sponge_xy = np.array(
            [rng.uniform(*SPONGE_X_RANGE), rng.uniform(*SPONGE_Y_RANGE)], dtype=float
        )
        plate_xy = np.array(
            [rng.uniform(*PLATE_X_RANGE), rng.uniform(*PLATE_Y_RANGE)], dtype=float
        )
        separation = float(np.linalg.norm(sponge_xy - plate_xy))
        if separation >= rim + 0.5 * float(np.max(SPONGE_SIZE[:2])) + 0.02:
            break
    else:
        raise RuntimeError("Could not sample separated sponge and plate locations.")
    return ScenarioSpecification(
        seed=seed,
        sample_attempt=attempt,
        sponge_youngs_modulus_pa=float(
            round(rng.uniform(*SPONGE_YOUNGS_MODULUS_RANGE), -2)
        ),
        sponge_color=SPONGE_COLORS[int(rng.integers(len(SPONGE_COLORS)))],
        sponge_xy=(float(sponge_xy[0]), float(sponge_xy[1])),
        plate_xy=(float(plate_xy[0]), float(plate_xy[1])),
    )


# -- workcell --------------------------------------------------------------


BotInfo = EmbodimentModel


@dataclass
class Workcell:
    desk_actors: list[physics.Actor]
    plate: physics.Actor
    sponge: physics.Actor
    plate_model: SolidPlate
    task_object_label: str = "sponge"

    @property
    def task_object(self) -> physics.Actor:
        """The manipulated actor, for embodiment-neutral recording code."""
        return self.sponge


def sponge_tet_mesh(
    size: npt.ArrayLike = SPONGE_SIZE, cell: float = SPONGE_CELL
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]:
    """Kuhn-subdivided box tetrahedral mesh centred at the origin.

    Each grid cube is split into six tetrahedra along its main diagonal, which
    tiles the box conformingly.  Returns node coordinates (n, 3) and
    connectivity (m, 4) with positive orientation.
    """
    size = np.asarray(size, dtype=float)
    counts = np.maximum(1, np.round(size / cell).astype(int))
    axes = [np.linspace(-0.5 * s, 0.5 * s, n + 1) for s, n in zip(size, counts)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    nodes = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    strides = np.array([(counts[1] + 1) * (counts[2] + 1), counts[2] + 1, 1])

    def node(i: int, j: int, k: int) -> int:
        return int(i * strides[0] + j * strides[1] + k)

    orders = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    tets: list[list[int]] = []
    for i in range(counts[0]):
        for j in range(counts[1]):
            for k in range(counts[2]):
                base = np.array([i, j, k])
                for order in orders:
                    corners = [base.copy()]
                    for axis in order:
                        corners.append(corners[-1].copy())
                        corners[-1][axis] += 1
                    tets.append([node(*c) for c in corners])
    connectivity = np.asarray(tets, dtype=np.int32)
    a, b, c, d = (nodes[connectivity[:, i]] for i in range(4))
    volume = np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a)
    flip = volume < 0.0
    connectivity[flip] = connectivity[flip][:, [0, 2, 1, 3]]
    return nodes, connectivity


def sponge_material(youngs_modulus: float) -> physics.SoftMaterialParams:
    material = physics.SoftMaterialParams()
    material.type = physics.SoftMaterialType.NEO_HOOKEAN
    material.neo_hookean.youngs_modulus = float(youngs_modulus)
    material.neo_hookean.poisson_ratio = SPONGE_POISSON_RATIO
    material.density = SPONGE_DENSITY
    return material


def create_full_robot(
    scene: physics.Scene, context: robotics.RoboticsContext
) -> BotInfo:
    return build_openarm_v2(scene, context, contact_params)


def create_workcell(
    scene: physics.Scene,
    specification: ScenarioSpecification | None = None,
) -> Workcell:
    """Build the wooden desk, the solid plate, and the soft sponge."""
    specification = specification or ScenarioSpecification.fixed()
    block_asset = "prefabs/box_and_blocks/collision/block.mochi.h5"
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

    model = plate_model()
    # Centre the plate on its own footprint so plate_xy is the dish centre.
    vertices = model.vertices - np.array([*model.center_xy, 0.0])
    plate_shape = physics.create_tri_mesh_shape(
        vertices.ravel(), model.faces.ravel().astype(np.int32)
    )
    plate = scene.create_rigid_actor(
        name="ceramic_plate",
        layer="plate",
        shape=plate_shape,
        is_static=True,
        contact=contact_params(CERAMIC_FRICTION),
        world_from_local=physics.TransformRT(translation=specification.plate_origin),
    )
    physics.release_shape(plate_shape)

    nodes, connectivity = sponge_tet_mesh()
    sponge_shape = physics.create_tet_mesh_shape(nodes.ravel(), connectivity.ravel())
    sponge = scene.create_soft_actor(
        name="soft_sponge",
        layer="sponge",
        shape=sponge_shape,
        material=sponge_material(specification.sponge_youngs_modulus_pa),
        contact=contact_params(SPONGE_FRICTION),
        world_from_local=physics.TransformRT(translation=specification.sponge_start),
    )
    physics.release_shape(sponge_shape)
    return Workcell(desk_actors=desk_actors, plate=plate, sponge=sponge, plate_model=model)


def sponge_position(sponge: physics.Actor) -> npt.NDArray[np.float64]:
    """World centre of the deformed sponge (bounding-box midpoint)."""
    bounds = sponge.get_aabb_world()
    return 0.5 * (np.asarray(bounds.min, dtype=float) + np.asarray(bounds.max, dtype=float))


# -- kinematics and planning ------------------------------------------------


class RightArmKinematics(OpenArmKinematics):
    """OpenArm kinematic twin with the pitched approach and plate obstacles."""

    def __init__(
        self,
        context: robotics.RoboticsContext,
        reference: BotInfo,
        specification: ScenarioSpecification | None = None,
    ) -> None:
        specification = specification or ScenarioSpecification.fixed()
        super().__init__(
            context,
            reference,
            contact_params,
            SpongePlateCollisionModel(
                specification,
                DESK_MIN,
                DESK_SIZE,
                rim_radius=specification.plate_rim_radius - 0.005,
                rim_z=specification.plate_rim_z,
            ),
            approach_pitch=float(APPROACH_PITCH),
        )


@dataclass
class PlannedMotion:
    """OpenArm reference trajectory: free segments are optimized, contact ones are IK lines."""

    home_to_pre_pick: npt.NDArray[np.float64]
    pre_pick_to_pick: npt.NDArray[np.float64]
    pick_to_lift: npt.NDArray[np.float64]
    lift_to_hover: npt.NDArray[np.float64]
    hover_to_press: npt.NDArray[np.float64]
    wipe_segments: list[npt.NDArray[np.float64]]
    press_to_lift_off: npt.NDArray[np.float64]
    lift_off_to_place_safe: npt.NDArray[np.float64]
    place_safe_to_place: npt.NDArray[np.float64]
    place_to_retreat: npt.NDArray[np.float64]
    retreat_to_home: npt.NDArray[np.float64]

    def segments(self) -> Iterable[npt.NDArray[np.float64]]:
        yield self.home_to_pre_pick
        yield self.pre_pick_to_pick
        yield self.pick_to_lift
        yield self.lift_to_hover
        yield self.hover_to_press
        yield from self.wipe_segments
        yield self.press_to_lift_off
        yield self.lift_off_to_place_safe
        yield self.place_safe_to_place
        yield self.place_to_retreat
        yield self.retreat_to_home


def _cartesian_ik_reference(
    kinematics: ArmKinematics,
    start_pose: npt.ArrayLike,
    start_position: npt.ArrayLike,
    goal_position: npt.ArrayLike,
    num_knots: int,
) -> npt.NDArray[np.float64]:
    """Follow a Cartesian line with continuation IK to stay on one joint branch."""
    start_pose = np.asarray(start_pose, dtype=float)
    positions = np.linspace(start_position, goal_position, num_knots, dtype=float)
    path = [start_pose]
    seed = start_pose
    for position in positions[1:]:
        seed = kinematics.solve(position, seed)
        path.append(seed)
    return np.asarray(path, dtype=float)


def plan_motion(
    kinematics: ArmKinematics,
    reference: BotInfo | EmbodimentModel,
    specification: ScenarioSpecification | None = None,
    *,
    optimize_trajectory: bool = True,
) -> PlannedMotion:
    """Build the OpenArm reference trajectory; optimize the free-space segments."""
    spec = specification or ScenarioSpecification.fixed()
    home = reference.default_pose[reference.right_arm_dofs]
    pre_pick = kinematics.solve(spec.pre_pick, home)
    home_to_pre_pick = np.linspace(home, pre_pick, 12, dtype=float)
    pre_pick_to_pick = _cartesian_ik_reference(kinematics, pre_pick, spec.pre_pick, spec.pick, 8)
    pick_to_lift = _cartesian_ik_reference(
        kinematics, pre_pick_to_pick[-1], spec.pick, spec.pick_lift, 8
    )
    lift_to_hover = _cartesian_ik_reference(
        kinematics, pick_to_lift[-1], spec.pick_lift, spec.wipe_hover, 16
    )
    hover_to_press = _cartesian_ik_reference(
        kinematics, lift_to_hover[-1], spec.wipe_hover, spec.wipe_start, 8
    )
    wipe_segments: list[npt.NDArray[np.float64]] = []
    pose = hover_to_press[-1]
    position = spec.wipe_start
    for index, segment in enumerate(spec.wipe_passes):
        if index:
            # Shift sideways to the next pass while staying pressed on the plate.
            shift = _cartesian_ik_reference(kinematics, pose, position, segment[0], 6)
            wipe_segments.append(shift)
            pose, position = shift[-1], segment[0]
        stroke = _cartesian_ik_reference(kinematics, pose, position, segment[-1], 12)
        wipe_segments.append(stroke)
        pose, position = stroke[-1], segment[-1]
    press_to_lift_off = _cartesian_ik_reference(
        kinematics, pose, spec.wipe_end, spec.wipe_lift_off, 8
    )
    lift_off_to_place_safe = _cartesian_ik_reference(
        kinematics, press_to_lift_off[-1], spec.wipe_lift_off, spec.place_safe, 16
    )
    place_safe_to_place = _cartesian_ik_reference(
        kinematics, lift_off_to_place_safe[-1], spec.place_safe, spec.place, 8
    )
    place_to_retreat = _cartesian_ik_reference(
        kinematics, place_safe_to_place[-1], spec.place, spec.pre_pick, 8
    )
    retreat_to_home = np.linspace(place_to_retreat[-1], home, 12, dtype=float)

    motion = PlannedMotion(
        home_to_pre_pick,
        pre_pick_to_pick,
        pick_to_lift,
        lift_to_hover,
        hover_to_press,
        wipe_segments,
        press_to_lift_off,
        lift_off_to_place_safe,
        place_safe_to_place,
        place_to_retreat,
        retreat_to_home,
    )
    if not optimize_trajectory:
        print("Trajectory optimization disabled: using direct Cartesian references.")
        return motion

    print("TrajOpt-style collision-aware optimization of the free-space segments:")
    optimizer = TrajOptTrajectoryOptimizer(kinematics)
    motion.home_to_pre_pick = optimizer.optimize(home_to_pre_pick, max_iterations=28)
    motion.lift_to_hover = optimizer.optimize(lift_to_hover, max_iterations=28)
    motion.lift_off_to_place_safe = optimizer.optimize(
        lift_off_to_place_safe, max_iterations=28
    )
    motion.retreat_to_home = optimizer.optimize(retreat_to_home, max_iterations=28)
    return motion


# -- embodiment registry ----------------------------------------------------


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
        [robotics.RoboticsContext, EmbodimentModel, ScenarioSpecification], ArmKinematics
    ]
    policy: str
    """``"module:Class"`` implementing :class:`episode.EpisodePolicy`."""


def _openarm_kinematics(
    context: robotics.RoboticsContext,
    info: EmbodimentModel,
    specification: ScenarioSpecification,
) -> ArmKinematics:
    return RightArmKinematics(context, info, specification)


EMBODIMENTS: dict[str, EmbodimentSpec] = {
    "openarm_v2": EmbodimentSpec(
        embodiment_id="openarm_v2",
        scene_name="OpenArm v2: sponge wipes plate",
        render_manifest="",
        solver_max_iter=8,
        build=create_full_robot,
        destroy=destroy_openarm_v2,
        kinematics=_openarm_kinematics,
        policy="scenarios.sponge_plate.openarm_policy:OpenArmPolicy",
    ),
}
DEFAULT_EMBODIMENT = "openarm_v2"


@dataclass
class SpongePlateScenario:
    """Built task state for one embodiment, independent of any controller or policy."""

    scene: physics.Scene
    specification: ScenarioSpecification
    bot_info: EmbodimentModel
    workcell: Workcell
    kinematics: ArmKinematics
    embodiment: EmbodimentSpec
    _closed: bool = field(default=False, init=False, repr=False)

    @property
    def embodiment_id(self) -> str:
        return self.embodiment.embodiment_id

    @classmethod
    def build(
        cls,
        context: robotics.RoboticsContext,
        embodiment_id: str = DEFAULT_EMBODIMENT,
        specification: ScenarioSpecification | None = None,
    ) -> SpongePlateScenario:
        """Construct one specified scene: embodiment, ground, workcell, kinematic twin."""
        embodiment = EMBODIMENTS[embodiment_id]
        specification = specification or ScenarioSpecification.fixed()
        scene = physics.create_scene(embodiment.scene_name)
        scene.set_gravity([0.0, 0.0, -9.80665])
        solver = scene.get_solver_params()
        solver.integration_method = physics.IntegrationMethod.BDF2
        solver.non_linear_solver.max_iter = embodiment.solver_max_iter
        scene.set_solver_params(solver)
        info: EmbodimentModel | None = None
        kinematics: ArmKinematics | None = None
        try:
            info = embodiment.build(scene, context)
            ground_shape = physics.create_plane_shape(normal=[0.0, 0.0, 1.0], distance=0.0)
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
            return cls(scene, specification, info, workcell, kinematics, embodiment)
        except Exception:
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
        planner: Callable[[SpongePlateScenario], Any] | None = None,
    ) -> tuple[SpongePlateScenario, Any]:
        """Sample a task, build it, and plan it; resample while planning fails."""
        from .episode import PlanningError

        rng = np.random.default_rng(seed)
        last_error: Exception | None = None
        for attempt in range(1, MAX_RANDOMIZATION_ATTEMPTS + 1):
            specification = sample_specification(rng, seed, attempt)
            scenario = cls.build(context, embodiment_id, specification)
            if planner is None:
                return scenario, None
            try:
                return scenario, planner(scenario)
            except PlanningError as error:
                last_error = error
                scenario.close()
                print(f"Random sample {attempt} was not plannable ({error}); resampling.")
            except Exception:
                scenario.close()
                raise
        raise RuntimeError(
            "Could not plan a randomized scenario after "
            f"{MAX_RANDOMIZATION_ATTEMPTS} attempts for seed {seed}."
        ) from last_error

    def close(self) -> None:
        if self._closed:
            return
        self.kinematics.close()
        self.embodiment.destroy(self.scene, self.bot_info)
        physics.destroy_scene(self.scene)
        self._closed = True


__all__ = [
    "APPROACH_PITCH",
    "CLEAN_MIN_NORMAL_FORCE",
    "CLEAN_MIN_SLIDING_SPEED",
    "CLEAN_SUCCESS_COVERAGE",
    "DEFAULT_EMBODIMENT",
    "DESK_TOP_Z",
    "DIRT_CELL",
    "DIRT_RADIUS",
    "EMBODIMENTS",
    "FINGERS_WIPE_CLOSED",
    "GRASP_UP",
    "PLATE_COLOR",
    "RENDER_EVERY_STEPS",
    "SPONGE_SIZE",
    "TIME_STEP",
    "TRAJECTORY_COLOR",
    "WOOD_COLOR",
    "CleanlinessMap",
    "EmbodimentSpec",
    "PlannedMotion",
    "RightArmKinematics",
    "ScenarioSpecification",
    "SpongePlateScenario",
    "Workcell",
    "contact_params",
    "create_workcell",
    "plan_motion",
    "plate_model",
    "sample_specification",
    "sponge_position",
    "sponge_tet_mesh",
]
