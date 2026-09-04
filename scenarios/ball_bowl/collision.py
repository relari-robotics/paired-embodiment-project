"""Ball-and-bowl obstacle model consumed by embodiment kinematics."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from superdex import physics


class BallBowlCollisionModel:
    """Evaluate embodiment-provided collision proxies against the workcell."""

    def __init__(
        self,
        specification: object,
        desk_min: npt.ArrayLike,
        desk_size: npt.ArrayLike,
    ) -> None:
        self.specification = specification
        self.desk_min = np.asarray(desk_min, dtype=float)
        self.desk_max = self.desk_min + np.asarray(desk_size, dtype=float)

    @staticmethod
    def actor_box_clearance(
        actor: physics.Actor,
        box_min: npt.NDArray[np.float64],
        box_max: npt.NDArray[np.float64],
    ) -> float:
        bounds = actor.get_aabb_world()
        actor_min = np.asarray(bounds.min, dtype=float)
        actor_max = np.asarray(bounds.max, dtype=float)
        separation = np.maximum(
            np.maximum(box_min - actor_max, actor_min - box_max),
            0.0,
        )
        if np.any(separation > 0.0):
            return float(np.linalg.norm(separation))
        overlap = np.minimum(actor_max - box_min, box_max - actor_min)
        return -float(np.min(overlap))

    def evaluate(
        self,
        kinematics: object,
        arm_pose: npt.ArrayLike,
    ) -> tuple[float, float]:
        kinematics.set_arm_pose(arm_pose)
        moving = kinematics.world_proxy_spheres("moving")
        static = kinematics.world_proxy_spheres("static")
        clearances = [
            self.actor_box_clearance(actor, self.desk_min, self.desk_max)
            for actor in kinematics.moving_proxy_actors
        ]

        spec = self.specification
        bowl_xy = np.asarray(spec.bowl_xy, dtype=float)
        rim_radius = 0.010
        for theta in np.linspace(0.0, 2.0 * np.pi, 20, endpoint=False):
            rim = np.array(
                [
                    bowl_xy[0] + spec.bowl_outer_radii[0] * np.cos(theta),
                    bowl_xy[1] + spec.bowl_outer_radii[1] * np.sin(theta),
                    spec.bowl_rim_z,
                ]
            )
            clearances.extend(
                float(np.linalg.norm(center - rim)) - radius - rim_radius
                for center, radius in moving
            )

        for moving_center, moving_radius in moving[2:]:
            for static_center, static_radius in static[2:]:
                clearances.append(
                    float(np.linalg.norm(moving_center - static_center))
                    - moving_radius
                    - static_radius
                )

        if not clearances:
            return 0.0, float("inf")
        clearance = np.asarray(clearances, dtype=float)
        penetration = np.maximum(0.008 - clearance, 0.0)
        return float(np.dot(penetration, penetration)), float(np.min(clearance))
