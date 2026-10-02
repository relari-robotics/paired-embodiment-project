"""Physical organizer, dividers, parts, and the existing OpenArm tabletop."""

from dataclasses import dataclass, field
import numpy as np
from superdex import physics

from scenarios.ball_bowl.scenario import (
    DESK_MIN,
    DESK_SIZE,
    DESK_LEG_SIZE,
    DESK_TOP_Z,
    TIME_STEP as TIME_STEP,
    RENDER_EVERY_STEPS as RENDER_EVERY_STEPS,
    WOOD_COLOR,
    contact_params,
)
from scenarios.ball_bowl.collision import BallBowlCollisionModel
from superdex_scenarios.embodiments.openarm_v2 import (
    build_openarm_v2,
    destroy_openarm_v2,
    OpenArmKinematics,
)
from .geometry import (
    ScenarioSpecification,
    TRAY_SIZE,
    WALL,
    WALL_HEIGHT,
    SLOT_Y,
    SLOT_CLEARANCE,
    BIN_Y,
    PART_SIZE,
    COLORS,
    DIVIDER_BOXES,
    DIVIDER_THICKNESS,
    union_box_mesh,
)


def create_body(
    scene, name, boxes, position, *, static=False, mass=0.08, friction=0.65, sdf=None
):
    vertices, faces = union_box_mesh(boxes)
    shape = physics.create_tri_mesh_shape(vertices.ravel(), faces.ravel())
    try:
        return scene.create_rigid_actor(
            name=name,
            layer="organizer",
            shape=shape,
            is_static=static,
            mass=mass,
            contact=contact_params(friction),
            world_from_local=physics.TransformRT(translation=position),
            **({"sdf": sdf} if sdf is not None else {}),
        )
    finally:
        physics.release_shape(shape)


class CollisionModel:
    """Desk, task objects, and parked-arm proxies for free-space paths."""

    def __init__(self, scene):
        self.scene = scene
        self.obstacles = []
        self.active = None

    def evaluate(self, kinematics, arm_pose):
        kinematics.set_arm_pose(arm_pose)
        clearances = []
        boxes = [(DESK_MIN, DESK_MIN + DESK_SIZE)]
        for actor, components in self.obstacles:
            if actor is self.active:
                continue
            transform = actor.get_root_transform()
            for centre, size in components:
                half = np.asarray(size) / 2
                corners = [
                    np.asarray(
                        (
                            transform
                            * physics.TransformRT(
                                translation=np.asarray(centre) + half * [x, y, z]
                            )
                        ).translation
                    )
                    for x in (-1, 1)
                    for y in (-1, 1)
                    for z in (-1, 1)
                ]
                boxes.append((np.min(corners, axis=0), np.max(corners, axis=0)))
        for actor in kinematics.moving_proxy_actors:
            clearances.extend(
                BallBowlCollisionModel.actor_box_clearance(actor, lo, hi)
                for lo, hi in boxes
            )
        for p, r in kinematics.world_proxy_spheres("moving")[2:]:
            for s, t in kinematics.world_proxy_spheres("static")[2:]:
                clearances.append(float(np.linalg.norm(p - s) - r - t))
        values = np.asarray(clearances)
        return float(np.square(np.maximum(0.005 - values, 0)).sum()), float(
            values.min()
        )


@dataclass
class OrganizerScenario:
    scene: object
    specification: ScenarioSpecification
    bot_info: object
    kinematics: object
    organizer: object
    dividers: list
    parts: list
    desk_actors: list
    colors: dict
    left_kinematics: object = None
    _closed: bool = field(default=False, init=False)

    @classmethod
    def build(cls, context, specification=None):
        spec = specification or ScenarioSpecification.fixed()
        spec.validate(DESK_MIN, DESK_SIZE)
        scene = physics.create_scene("OpenArm: assemble and pack organizer")
        scene.set_gravity([0, 0, -9.80665])
        solver = scene.get_solver_params()
        solver.integration_method = physics.IntegrationMethod.BDF2
        solver.non_linear_solver.max_iter = 12
        scene.set_solver_params(solver)
        info = kin = None
        try:
            info = build_openarm_v2(
                scene, context, contact_params, sides=("right", "left")
            )
            desk = [
                create_body(
                    scene,
                    "wooden_desk/top",
                    [(DESK_SIZE / 2, DESK_SIZE)],
                    DESK_MIN,
                    static=True,
                )
            ]
            for x in (DESK_MIN[0] + 0.035, DESK_MIN[0] + DESK_SIZE[0] - 0.09):
                for y in (DESK_MIN[1] + 0.035, DESK_MIN[1] + DESK_SIZE[1] - 0.09):
                    desk.append(
                        create_body(
                            scene,
                            f"wooden_desk/leg_{len(desk)}",
                            [(DESK_LEG_SIZE / 2, DESK_LEG_SIZE)],
                            [x, y, 0],
                            static=True,
                        )
                    )
            boxes = [((0, 0, TRAY_SIZE[2] / 2), TRAY_SIZE)]
            for axis in (0, 1):
                for sign in (-1, 1):
                    centre = np.array([0.0, 0.0, TRAY_SIZE[2] + WALL_HEIGHT / 2])
                    centre[axis] = sign * (TRAY_SIZE[axis] - WALL) / 2
                    size = np.array([*TRAY_SIZE[:2], WALL_HEIGHT])
                    size[axis] = WALL
                    boxes.append((centre, size))
            # Four short guides at each divider's ends leave an open vertical slot.
            for y in SLOT_Y:
                for x in (-0.065, 0.065):
                    for sign in (-1, 1):
                        guide_y = y + sign * (
                            (DIVIDER_THICKNESS + SLOT_CLEARANCE) / 2 + 0.004
                        )
                        boxes.append(((x, guide_y, 0.020), (0.014, 0.008, 0.024)))
            organizer = create_body(
                scene,
                "organizer/tray",
                boxes,
                [*spec.organizer_xy, DESK_TOP_Z],
                static=True,
            )
            dividers = [
                create_body(
                    scene,
                    f"organizer/divider_{i}",
                    DIVIDER_BOXES,
                    [*xy, DESK_TOP_Z + 0.0003],
                    mass=0.09,
                )
                for i, xy in enumerate(spec.divider_xy)
            ]
            parts = [
                create_body(
                    scene,
                    f"organizer/part_{i}",
                    [((0, 0, PART_SIZE[2] / 2), PART_SIZE)],
                    [*xy, DESK_TOP_Z + 0.0003],
                    mass=0.045,
                    friction=0.8,
                )
                for i, xy in enumerate(spec.part_xy)
            ]
            colors = {a.get_name(): WOOD_COLOR.tolist() for a in desk}
            colors[organizer.get_name()] = [0.24, 0.30, 0.35]
            colors.update({a.get_name(): [0.74, 0.79, 0.80] for a in dividers})
            colors.update({a.get_name(): list(c) for a, c in zip(parts, COLORS)})
            collisions = CollisionModel(scene)
            collisions.obstacles = [(organizer, boxes)] + [
                (a, DIVIDER_BOXES) for a in dividers
            ]
            collisions.obstacles += [
                (a, [((0, 0, PART_SIZE[2] / 2), PART_SIZE)]) for a in parts
            ]
            kin = OpenArmKinematics(
                context, info, contact_params, collisions, approach_pitch=np.radians(30)
            )
            left = OpenArmKinematics(
                context,
                info,
                contact_params,
                collisions,
                approach_pitch=np.radians(30),
                side="left",
            )
            return cls(
                scene, spec, info, kin, organizer, dividers, parts, desk, colors, left
            )
        except Exception:
            if kin is not None:
                kin.close()
            if info is not None:
                destroy_openarm_v2(scene, info)
            physics.destroy_scene(scene)
            raise

    def target(self, kind, index):
        local = (
            [0, SLOT_Y[index], TRAY_SIZE[2]]
            if kind == "divider"
            else [-0.022, BIN_Y[index], TRAY_SIZE[2]]
        )
        transform = self.organizer.get_root_transform() * physics.TransformRT(
            translation=local
        )
        return np.asarray(transform.translation, dtype=float)

    def outcome(self):
        """Require actual seated dividers and whole settled parts in their assigned bins."""
        result = {"dividers": [], "parts": []}
        for kind, actors in (("divider", self.dividers), ("part", self.parts)):
            for i, actor in enumerate(actors):
                pose = actor.get_root_transform()
                p = np.asarray(pose.translation)
                target = self.target(kind, i)
                rotation = physics.TransformRT(
                    rotation=pose.rotation, translation=[0, 0, 0]
                )
                up = np.asarray(
                    (rotation * physics.TransformRT(translation=[0, 0, 1])).translation
                )
                speed = float(np.linalg.norm(actor.get_linear_velocity()))
                angular_speed = float(np.linalg.norm(actor.get_angular_velocity()))
                if kind == "divider":
                    error = p - target
                    forward = np.asarray(
                        (
                            rotation * physics.TransformRT(translation=[1, 0, 0])
                        ).translation
                    )
                    passed = (
                        abs(error[0]) < 0.007
                        and abs(error[1]) < SLOT_CLEARANCE / 2
                        and abs(error[2]) < 0.004
                        and up[2] > 0.99
                        and forward[0] > 0.99
                    )
                else:
                    bounds = actor.get_aabb_world()
                    lo, hi = np.asarray(bounds.min), np.asarray(bounds.max)
                    origin = np.asarray(self.organizer.get_root_transform().translation)
                    ylo = (
                        -TRAY_SIZE[1] / 2 + WALL
                        if i == 0
                        else SLOT_Y[i - 1] + DIVIDER_THICKNESS / 2
                    )
                    yhi = (
                        TRAY_SIZE[1] / 2 - WALL
                        if i == 2
                        else SLOT_Y[i] - DIVIDER_THICKNESS / 2
                    )
                    passed = (
                        lo[0] > origin[0] - TRAY_SIZE[0] / 2 + WALL - 0.001
                        and hi[0] < origin[0] + TRAY_SIZE[0] / 2 - WALL + 0.001
                        and lo[1] > origin[1] + ylo - 0.001
                        and hi[1] < origin[1] + yhi + 0.001
                        and abs(lo[2] - target[2]) < 0.004
                    )
                result[kind + "s"].append(
                    {
                        "name": actor.get_name(),
                        "position_m": p.tolist(),
                        "target_m": target.tolist(),
                        "speed_m_s": speed,
                        "angular_speed_rad_s": angular_speed,
                        "success": bool(
                            passed and speed < 0.015 and angular_speed < 0.15
                        ),
                    }
                )
        result["success"] = all(
            item["success"]
            for group in (result["dividers"], result["parts"])
            for item in group
        )
        return result

    def close(self):
        if not self._closed:
            self.kinematics.close()
            self.left_kinematics.close()
            destroy_openarm_v2(self.scene, self.bot_info)
            physics.destroy_scene(self.scene)
            self._closed = True
