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
twin.  Embodiments are registered in :data:`EMBODIMENTS`; each names the policy
class (see :mod:`episode`) that plans and executes the task for it.  The
OpenArm entry is the complete reference; the human entry's policy is the
project deliverable.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from superdex import physics, robotics
from superdex.physics.paths import resolve_asset

from superdex_scenarios.embodiments.base import ArmKinematics, EmbodimentModel
from superdex_scenarios.embodiments.human_right_arm import (
    HumanArmKinematics,
    build_human_right_arm,
    destroy_human_right_arm,
)
from superdex_scenarios.embodiments.openarm_v2 import (
    OpenArmKinematics,
    build_openarm_v2,
    destroy_openarm_v2,
)
from superdex_scenarios.planning import TrajOptTrajectoryOptimizer

from .collision import BallBowlCollisionModel

# Workcell dimensions.  SuperDex uses metres and a Z-up world.
DESK_MIN = np.array([-0.36, -0.38, 0.338], dtype=float)
DESK_SIZE = np.array([1.02, 0.76, 0.05], dtype=float)
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

BOWL_CENTER_XY = np.array([-0.04, 0.02])
BOWL_SCALE = np.array([2.4, 2.4, 0.35])
BOWL_OUTER_RADIUS = 0.047 * BOWL_SCALE[0]
BOWL_RIM_Z = DESK_TOP_Z + 0.113 * BOWL_SCALE[2]

# The points below refer to GRASP_POINT_EE, which coincides with the ball centre
# while held.  Keeping the same local point for every IK target preserves the
# authored gripper orientation.  The gripper's local -Z approach axis points
# mostly along world +X, so pre-grasp retracts along that axis instead of dropping
# vertically onto the ball.
PICK = BALL_START.copy()
GRIPPER_APPROACH_WORLD = np.array([1.0, 0.0, 0.0])
PRE_PICK = PICK - 0.13 * GRIPPER_APPROACH_WORLD
PICK_LIFT = PICK + np.array([0.0, 0.0, 0.115])
PLACE_SAFE = np.array([BOWL_CENTER_XY[0], BOWL_CENTER_XY[1], 0.52])
PLACE = np.array([BOWL_CENTER_XY[0], BOWL_CENTER_XY[1], 0.48])
BALL_RELEASE_CLEARANCE = PLACE[2] - BOWL_RIM_Z - BALL_RADIUS

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


@dataclass(frozen=True)
class ColorChoice:
    """Named RGB color available to the randomized scene."""

    name: str
    rgb: tuple[float, float, float]


@dataclass(frozen=True)
class BowlShapeChoice:
    """A collision/render scale variant of the open paper-cup bowl mesh."""

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
    BowlShapeChoice("shallow_round", (2.40, 2.40, 0.35)),
    BowlShapeChoice("deep_round", (2.55, 2.55, 0.55)),
    BowlShapeChoice("wide_oval_x", (2.90, 2.40, 0.34)),
    BowlShapeChoice("wide_oval_y", (2.40, 2.90, 0.34)),
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

    @classmethod
    def fixed(cls) -> ScenarioSpecification:
        """Return the original deterministic scenario for regression comparisons."""
        return cls(
            seed=None,
            sample_attempt=1,
            ball_mass_kg=BALL_MASS,
            ball_color=BALL_COLORS[0],
            bowl_color=BOWL_COLORS[0],
            bowl_shape=BOWL_SHAPES[0],
            ball_xy=(float(BALL_XY[0]), float(BALL_XY[1])),
            bowl_xy=(float(BOWL_CENTER_XY[0]), float(BOWL_CENTER_XY[1])),
        )

    @property
    def ball_start(self) -> npt.NDArray[np.float64]:
        return np.array(
            [*self.ball_xy, DESK_TOP_Z + BALL_RADIUS + CONTACT_THRESHOLD],
            dtype=float,
        )

    @property
    def bowl_scale(self) -> npt.NDArray[np.float64]:
        return np.asarray(self.bowl_shape.scale, dtype=float)

    @property
    def bowl_outer_radii(self) -> npt.NDArray[np.float64]:
        return 0.047 * self.bowl_scale[:2]

    @property
    def bowl_rim_z(self) -> float:
        return DESK_TOP_Z + 0.113 * float(self.bowl_scale[2])

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
        return np.array([*self.bowl_xy, release_z], dtype=float)

    @property
    def place_safe(self) -> npt.NDArray[np.float64]:
        return np.array([*self.bowl_xy, self.place[2] + 0.04], dtype=float)

    @property
    def ball_shell_inertia(self) -> float:
        return (2.0 / 3.0) * self.ball_mass_kg * BALL_RADIUS**2

    def contains_ball(self, position: npt.ArrayLike) -> bool:
        """Use the randomized elliptical rim and height for success validation."""
        position = np.asarray(position, dtype=float)
        normalized_xy = (
            position[:2] - np.asarray(self.bowl_xy)
        ) / self.bowl_outer_radii
        return bool(
            np.dot(normalized_xy, normalized_xy) < 1.0
            and position[2] < self.bowl_rim_z + 2.0 * BALL_RADIUS
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
                "position_m": [*self.bowl_xy, DESK_TOP_Z],
                "outer_radii_m": self.bowl_outer_radii.tolist(),
                "rim_z_m": self.bowl_rim_z,
            },
            "targets": {
                "pick_m": self.pick.tolist(),
                "place_m": self.place.tolist(),
            },
        }


def sample_specification(
    rng: np.random.Generator, seed: int, attempt: int
) -> ScenarioSpecification:
    """Sample one candidate while enforcing tabletop and object separation."""
    bowl_shape = BOWL_SHAPES[int(rng.integers(len(BOWL_SHAPES)))]
    outer_radii = 0.047 * np.asarray(bowl_shape.scale[:2], dtype=float)
    for _ in range(128):
        ball_xy = np.array(
            [rng.uniform(*BALL_X_RANGE), rng.uniform(*BALL_Y_RANGE)], dtype=float
        )
        bowl_xy = np.array(
            [rng.uniform(*BOWL_X_RANGE), rng.uniform(*BOWL_Y_RANGE)], dtype=float
        )
        minimum_separation = float(np.max(outer_radii)) + BALL_RADIUS + 0.045
        if np.linalg.norm(ball_xy - bowl_xy) >= minimum_separation:
            break
    else:
        raise RuntimeError("Could not sample separated ball and bowl locations.")

    mass_step = int(rng.integers(1, 11))
    return ScenarioSpecification(
        seed=seed,
        sample_attempt=attempt,
        ball_mass_kg=round(mass_step * 0.05, 2),
        ball_color=BALL_COLORS[int(rng.integers(len(BALL_COLORS)))],
        bowl_color=BOWL_COLORS[int(rng.integers(len(BOWL_COLORS)))],
        bowl_shape=bowl_shape,
        ball_xy=(float(ball_xy[0]), float(ball_xy[1])),
        bowl_xy=(float(bowl_xy[0]), float(bowl_xy[1])),
    )


BotInfo = EmbodimentModel


@dataclass
class Workcell:
    desk_actors: list[physics.Actor]
    bowl: physics.Actor
    ball: physics.Actor


@dataclass
class PlannedMotion:
    """OpenArm reference trajectory produced by the included robot planner."""

    home_to_pre_pick: npt.NDArray[np.float64]
    pre_pick_to_pick: npt.NDArray[np.float64]
    pick_to_lift: npt.NDArray[np.float64]
    lift_to_place_safe: npt.NDArray[np.float64]
    place_safe_to_place: npt.NDArray[np.float64]
    place_to_retreat: npt.NDArray[np.float64]
    retreat_to_home: npt.NDArray[np.float64]

    def segments(self) -> Iterable[npt.NDArray[np.float64]]:
        return (
            self.home_to_pre_pick,
            self.pre_pick_to_pick,
            self.pick_to_lift,
            self.lift_to_place_safe,
            self.place_safe_to_place,
            self.place_to_retreat,
            self.retreat_to_home,
        )


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


def create_full_robot(
    scene: physics.Scene, context: robotics.RoboticsContext
) -> BotInfo:
    """Compatibility name for the scenario's selected OpenArm embodiment."""
    return build_openarm_v2(scene, context, contact_params)


def _load_shape(asset: str, scale: npt.ArrayLike) -> physics.ShapeHandle:
    return physics.load_shape_from_file(
        file_path=str(resolve_asset(asset)), bake_scale=np.asarray(scale, dtype=float)
    )


def create_workcell(
    scene: physics.Scene,
    specification: ScenarioSpecification | None = None,
) -> Workcell:
    """Build the wooden desk and the specified bowl and dynamic ball."""
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

    bowl_shape = _load_shape(
        "prefabs/paper_cups/collision/paper_cup.mochi.h5",
        specification.bowl_scale,
    )
    bowl = scene.create_rigid_actor(
        name="gray_bowl",
        layer="bowl",
        shape=bowl_shape,
        is_static=True,
        contact=contact_params(CERAMIC_FRICTION),
        world_from_local=physics.TransformRT(
            translation=[*specification.bowl_xy, DESK_TOP_Z]
        ),
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


class RightArmKinematics(OpenArmKinematics):
    """Scenario adapter supplying ball-and-bowl obstacles to OpenArm IK."""

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
            BallBowlCollisionModel(specification, DESK_MIN, DESK_SIZE),
        )


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
    """Build and optimize the included OpenArm reference trajectory."""
    specification = specification or ScenarioSpecification.fixed()
    home = reference.default_pose[reference.right_arm_dofs]
    if (
        reference.approach_direction_world is None
        or reference.pregrasp_distance is None
    ):
        raise ValueError("The OpenArm reference requires pregrasp metadata.")
    approach = np.asarray(reference.approach_direction_world, dtype=float)
    pre_pick_position = specification.pick - reference.pregrasp_distance * approach
    pre_pick = kinematics.solve(pre_pick_position, home)
    home_to_pre_pick = np.linspace(home, pre_pick, 12, dtype=float)
    pre_pick_to_pick = _cartesian_ik_reference(
        kinematics,
        pre_pick,
        pre_pick_position,
        specification.pick,
        8,
    )
    pick_to_lift = _cartesian_ik_reference(
        kinematics,
        pre_pick_to_pick[-1],
        specification.pick,
        specification.pick_lift,
        8,
    )
    lift_to_place_safe = _cartesian_ik_reference(
        kinematics,
        pick_to_lift[-1],
        specification.pick_lift,
        specification.place_safe,
        16,
    )
    place_safe_to_place = _cartesian_ik_reference(
        kinematics,
        lift_to_place_safe[-1],
        specification.place_safe,
        specification.place,
        7,
    )
    place_to_retreat = place_safe_to_place[::-1].copy()
    retreat_to_home = np.linspace(place_to_retreat[-1], home, 12, dtype=float)
    paths = (
        home_to_pre_pick,
        pre_pick_to_pick,
        pick_to_lift,
        lift_to_place_safe,
        place_safe_to_place,
        place_to_retreat,
        retreat_to_home,
    )

    if not optimize_trajectory:
        print("Trajectory optimization disabled: using direct Cartesian references.")
        return PlannedMotion(*paths)

    print("TrajOpt-style collision-aware trajectory optimization:")
    optimizer = TrajOptTrajectoryOptimizer(kinematics)
    return PlannedMotion(
        *(optimizer.optimize(path, max_iterations=28) for path in paths)
    )


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


def _human_kinematics(
    context: robotics.RoboticsContext,
    info: EmbodimentModel,
    specification: ScenarioSpecification,
) -> ArmKinematics:
    return HumanArmKinematics(
        context,
        info,
        contact_params,
        BallBowlCollisionModel(specification, DESK_MIN, DESK_SIZE),
    )


EMBODIMENTS: dict[str, EmbodimentSpec] = {
    "openarm_v2": EmbodimentSpec(
        embodiment_id="openarm_v2",
        scene_name="OpenArm v2: randomized ball into bowl",
        render_manifest=(
            "/scenarios/ball_bowl/studio/scene/openarm_ball_bowl_studio.mochi_scene"
        ),
        solver_max_iter=8,
        build=create_full_robot,
        destroy=destroy_openarm_v2,
        kinematics=_openarm_kinematics,
        policy="scenarios.ball_bowl.openarm_policy:OpenArmPolicy",
    ),
    "human_right_hand": EmbodimentSpec(
        embodiment_id="human_right_hand",
        scene_name="Human right hand: randomized ball into bowl",
        render_manifest=(
            "/scenarios/ball_bowl/studio/human_scene/human_ball_bowl_studio.mochi_scene"
        ),
        solver_max_iter=10,
        build=lambda scene, context: build_human_right_arm(scene, context, contact_params),
        destroy=destroy_human_right_arm,
        kinematics=_human_kinematics,
        policy="scenarios.ball_bowl.human_project:HumanPolicy",
    ),
}
DEFAULT_EMBODIMENT = "openarm_v2"


@dataclass
class BallBowlScenario:
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
        planner: Callable[[BallBowlScenario], Any] | None = None,
    ) -> tuple[BallBowlScenario, Any]:
        """Sample a task, build it, and plan it; resample while planning fails.

        ``planner`` receives the built scenario and returns the planned policy;
        it raises :class:`episode.PlanningError` for an infeasible sample, in
        which case the scene is destroyed and the next sample is tried (up to
        ``MAX_RANDOMIZATION_ATTEMPTS`` per seed).  Returns ``(scenario, policy)``.
        """
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
        """Release scenario-owned planning and embodiment resources once."""
        if self._closed:
            return
        self.kinematics.close()
        self.embodiment.destroy(self.scene, self.bot_info)
        physics.destroy_scene(self.scene)
        self._closed = True


class OpenArmBallBowlScenario(BallBowlScenario):
    """Compatibility API retaining the original eagerly planned OpenArm scenario."""

    plan: PlannedMotion

    @classmethod
    def build(  # type: ignore[override]
        cls,
        context: robotics.RoboticsContext,
        specification: ScenarioSpecification | None = None,
        *,
        optimize_trajectory: bool = True,
    ) -> OpenArmBallBowlScenario:
        scenario = super().build(context, "openarm_v2", specification)
        try:
            scenario.plan = plan_motion(
                scenario.kinematics,
                scenario.bot_info,
                scenario.specification,
                optimize_trajectory=optimize_trajectory,
            )
            return scenario
        except Exception:
            scenario.close()
            raise

    @classmethod
    def build_randomized(  # type: ignore[override]
        cls,
        context: robotics.RoboticsContext,
        seed: int,
        *,
        optimize_trajectory: bool = True,
    ) -> OpenArmBallBowlScenario:
        rng = np.random.default_rng(seed)
        last_error: Exception | None = None
        for attempt in range(1, MAX_RANDOMIZATION_ATTEMPTS + 1):
            specification = sample_specification(rng, seed, attempt)
            try:
                return cls.build(
                    context,
                    specification,
                    optimize_trajectory=optimize_trajectory,
                )
            except RuntimeError as error:
                last_error = error
                print(f"Random sample {attempt} was not plannable ({error}); resampling.")
        raise RuntimeError(
            "Could not plan a randomized scenario after "
            f"{MAX_RANDOMIZATION_ATTEMPTS} attempts for seed {seed}."
        ) from last_error


class HumanBallBowlScenario(BallBowlScenario):
    """Compatibility API for building the original unplanned human scenario."""

    @classmethod
    def build(  # type: ignore[override]
        cls,
        context: robotics.RoboticsContext,
        specification: ScenarioSpecification | None = None,
    ) -> HumanBallBowlScenario:
        return super().build(context, "human_right_hand", specification)

    @classmethod
    def build_randomized(  # type: ignore[override]
        cls,
        context: robotics.RoboticsContext,
        seed: int,
    ) -> HumanBallBowlScenario:
        specification = sample_specification(
            np.random.default_rng(seed),
            seed,
            attempt=1,
        )
        return cls.build(context, specification)
