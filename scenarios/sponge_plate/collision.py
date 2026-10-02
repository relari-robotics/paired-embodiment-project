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

"""Clearance model used by trajectory optimization for the sponge-and-plate task.

The moving right arm and gripper are approximated by bounding spheres (see
``OpenArmKinematics``).  Obstacles are the desk top (axis-aligned box) and the
plate rim, sampled as a ring of small spheres; the parked left arm supplies
self-collision spheres.  Only free-space segments are optimized against this
model; contact segments (grasp, wipe, release) are executed as direct inverse
kinematics references because touching the plate is the point of the task.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from scenarios.ball_bowl.collision import BallBowlCollisionModel


class SpongePlateCollisionModel:
    """Desk-box, plate-rim, and self-collision clearance for the OpenArm proxies."""

    def __init__(
        self,
        specification: object,
        desk_min: npt.ArrayLike,
        desk_size: npt.ArrayLike,
        *,
        rim_radius: float,
        rim_z: float,
        rim_thickness: float = 0.010,
    ) -> None:
        self.specification = specification
        self.desk_min = np.asarray(desk_min, dtype=float)
        self.desk_max = self.desk_min + np.asarray(desk_size, dtype=float)
        self.rim_radius = float(rim_radius)
        self.rim_z = float(rim_z)
        self.rim_thickness = float(rim_thickness)

    def evaluate(self, kinematics: object, arm_pose: npt.ArrayLike) -> tuple[float, float]:
        kinematics.set_arm_pose(arm_pose)
        moving = kinematics.world_proxy_spheres("moving")
        static = kinematics.world_proxy_spheres("static")
        clearances = [
            BallBowlCollisionModel.actor_box_clearance(actor, self.desk_min, self.desk_max)
            for actor in kinematics.moving_proxy_actors
        ]

        plate_xy = np.asarray(self.specification.plate_xy, dtype=float)
        for theta in np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False):
            rim = np.array(
                [
                    plate_xy[0] + self.rim_radius * np.cos(theta),
                    plate_xy[1] + self.rim_radius * np.sin(theta),
                    self.rim_z,
                ]
            )
            clearances.extend(
                float(np.linalg.norm(center - rim)) - radius - self.rim_thickness
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


__all__ = ["SpongePlateCollisionModel"]
