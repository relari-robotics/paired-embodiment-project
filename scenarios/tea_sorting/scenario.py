"""A real contact-physics tea task on the existing OpenArm desk."""

import numpy as np
from scipy.spatial.transform import Rotation
from superdex import physics
from scenarios.organizer.scenario import (
    OrganizerScenario as BaseScenario,
    CollisionModel,
    create_body,
    DESK_MIN,
    DESK_SIZE,
    DESK_LEG_SIZE,
    DESK_TOP_Z,
    WOOD_COLOR,
    contact_params,
    build_openarm_v2,
    destroy_openarm_v2,
    OpenArmKinematics,
    TIME_STEP as TIME_STEP,
    RENDER_EVERY_STEPS as RENDER_EVERY_STEPS,
)
from .geometry import (
    ScenarioSpecification as ScenarioSpecification,
    BOX_SIZE,
    WALL,
    HEIGHT,
    DIVIDER,
    SLOT_Y,
    BIN_Y,
    PACKET_SIZE,
    TEAS,
    COLORS,
    box_components,
    holder_components,
    packet_surface,
)

CAMERAS = {
    "detail": {
        "kind": "fixed",
        "look_from": [0.36, -0.46, 0.91],
        "look_at": [-0.025, 0.0, 0.412],
    },
    "tea_top": {
        "kind": "fixed",
        "look_from": [-0.025, 0.0, 1.10],
        "look_at": [-0.025, 0.0, 0.40],
        "up_world": [1, 0, 0],
    },
    "table": {
        "kind": "fixed",
        "look_from": [0.66, -0.72, 1.02],
        "look_at": [-0.055, 0.0, 0.40],
    },
}


class OrganizerScenario(BaseScenario):
    """Common runner protocol, with tea-specific construction and outcome."""

    @classmethod
    def build(cls, context, specification=None):
        spec = specification or ScenarioSpecification.fixed()
        spec.validate(DESK_MIN, DESK_SIZE)
        scene = physics.create_scene("OpenArm: sort wrapped tea by color")
        scene.set_gravity([0, 0, -9.80665])
        solver = scene.get_solver_params()
        solver.integration_method = physics.IntegrationMethod.BDF2
        solver.non_linear_solver.max_iter = 12
        scene.set_solver_params(solver)
        info = kin = left = None
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
            boxes = box_components()
            # Mean-edge automatic SDF spacing loses these 4 mm dividers and the
            # 8 mm floor. Resolve the actual thin walls at a fixed 1 mm pitch.
            fine_sdf = physics.GridSdfParams(
                resolution_mode=physics.GridSdfResolutionMode.EXPLICIT,
                resolution_delta=[0.001, 0.001, 0.001],
            )
            organizer = create_body(
                scene,
                "tea/box",
                boxes,
                [*spec.organizer_xy, DESK_TOP_Z],
                static=True,
                sdf=fine_sdf,
            )
            parts, holders = [], []
            for i, (xy, tea) in enumerate(zip(spec.part_xy, spec.tea_order)):
                vertices, quads, _ = packet_surface()
                faces = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
                shape = physics.create_tri_mesh_shape(vertices.ravel(), faces.ravel())
                try:
                    parts.append(
                        scene.create_rigid_actor(
                            name=f"tea/packet_{i}_{TEAS[tea]}",
                            layer="organizer",
                            shape=shape,
                            mass=0.004,
                            contact=contact_params(0.6),
                            world_from_local=physics.TransformRT(
                                translation=[
                                    *xy,
                                    DESK_TOP_Z + PACKET_SIZE[2] / 2 + 0.0003,
                                ]
                            ),
                        )
                    )
                finally:
                    physics.release_shape(shape)
                holders.append(
                    create_body(
                        scene,
                        f"tea/source_cradle_{i}",
                        holder_components(),
                        [*xy, DESK_TOP_Z],
                        static=True,
                        sdf=fine_sdf,
                    )
                )
            colors = {a.get_name(): WOOD_COLOR.tolist() for a in desk}
            colors[organizer.get_name()] = [0.45, 0.26, 0.11]
            colors.update(
                {a.get_name(): list(COLORS[t]) for a, t in zip(parts, spec.tea_order)}
            )
            colors.update({a.get_name(): [0.23, 0.25, 0.24] for a in holders})
            collisions = CollisionModel(scene)
            collisions.obstacles = [(organizer, boxes)]
            collisions.obstacles += [(a, holder_components()) for a in holders]
            collisions.obstacles += [(a, [((0, 0, 0), PACKET_SIZE)]) for a in parts]
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
            return cls(scene, spec, info, kin, organizer, [], parts, desk, colors, left)
        except Exception:
            for k in (kin, left):
                if k is not None:
                    k.close()
            if info is not None:
                destroy_openarm_v2(scene, info)
            physics.destroy_scene(scene)
            raise

    def target(self, kind, index):
        tea = self.specification.tea_order[index]
        transform = self.organizer.get_root_transform() * physics.TransformRT(
            translation=[-0.022, BIN_Y[tea], BOX_SIZE[2]]
        )
        return np.asarray(transform.translation, dtype=float)

    def outcome(self):
        result = {"dividers": [], "parts": []}
        origin = np.asarray(self.organizer.get_root_transform().translation)
        for i, actor in enumerate(self.parts):
            tea = self.specification.tea_order[i]
            pose = actor.get_root_transform()
            vertices, _, _ = packet_surface()
            world = Rotation.from_quat(np.asarray(pose.rotation)).apply(
                vertices
            ) + np.asarray(pose.translation)
            # A transformed local bounding box includes nonexistent pillow
            # corners. Test the actual collision surface, not that loose proxy.
            lo, hi = world.min(axis=0), world.max(axis=0)
            ylo = -BOX_SIZE[1] / 2 + WALL if tea == 0 else SLOT_Y[tea - 1] + DIVIDER / 2
            yhi = BOX_SIZE[1] / 2 - WALL if tea == 2 else SLOT_Y[tea] - DIVIDER / 2
            speed = float(np.linalg.norm(actor.get_linear_velocity()))
            angular = float(np.linalg.norm(actor.get_angular_velocity()))
            contained = bool(
                lo[0] > origin[0] - BOX_SIZE[0] / 2 + WALL - 0.001
                and hi[0] < origin[0] + BOX_SIZE[0] / 2 - WALL + 0.001
                and lo[1] > origin[1] + ylo - 0.001
                and hi[1] < origin[1] + yhi + 0.001
                and lo[2] >= origin[2] + BOX_SIZE[2] - 0.002
                and hi[2] < origin[2] + BOX_SIZE[2] + HEIGHT + PACKET_SIZE[2]
                and lo[2] < origin[2] + BOX_SIZE[2] + 0.005
            )
            result["parts"].append(
                {
                    "name": actor.get_name(),
                    "tea": TEAS[tea],
                    "bin": tea,
                    "position_m": list(actor.get_root_transform().translation),
                    "target_m": self.target("part", i).tolist(),
                    "speed_m_s": speed,
                    "angular_speed_rad_s": angular,
                    "contained": contained,
                    "success": contained and speed < 0.015 and angular < 0.15,
                    "surface_min_m": lo.tolist(),
                    "surface_max_m": hi.tolist(),
                }
            )
        result["success"] = all(p["success"] for p in result["parts"])
        return result
