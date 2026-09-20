"""How the OpenArm v2 is built into the ball-and-bowl workcell."""

from __future__ import annotations

from superdex_scenarios.embodiments.openarm_v2 import (
    build_openarm_v2,
    destroy_openarm_v2,
)

from ...scenario import EmbodimentSpec, contact_params
from .kinematics import left_arm, right_arm

_PACKAGE = "scenarios.ball_bowl.embodiments.openarm_v2"
RENDER_MANIFEST = (
    "/scenarios/ball_bowl/embodiments/openarm_v2/studio/openarm_ball_bowl_studio.mochi_scene"
)

EMBODIMENTS: dict[str, EmbodimentSpec] = {
    "openarm_v2": EmbodimentSpec(
        embodiment_id="openarm_v2",
        scene_name="OpenArm v2: randomized ball into bowl",
        render_manifest=RENDER_MANIFEST,
        solver_max_iter=8,
        build=lambda scene, context: build_openarm_v2(scene, context, contact_params),
        destroy=destroy_openarm_v2,
        kinematics=right_arm,
        policy=f"{_PACKAGE}.policy:OpenArmPolicy",
    ),
    # The default: both arms motor-controlled.  The single-arm reference policy
    # drives the right arm and leaves the left parked; the bimanual one, replayed
    # trajectory files, and teleoperation command either or both arms.
    "openarm_v2_bimanual": EmbodimentSpec(
        embodiment_id="openarm_v2_bimanual",
        scene_name="OpenArm v2 both arms: randomized ball into bowl",
        render_manifest=RENDER_MANIFEST,
        solver_max_iter=8,
        build=lambda scene, context: build_openarm_v2(
            scene, context, contact_params, sides=("right", "left")
        ),
        destroy=destroy_openarm_v2,
        kinematics=right_arm,
        left_kinematics=left_arm,
        policy=f"{_PACKAGE}.policy:OpenArmPolicy",
        move_bowl_policy=f"{_PACKAGE}.bimanual_policy:BimanualOpenArmPolicy",
    ),
}

__all__ = ["EMBODIMENTS", "RENDER_MANIFEST"]
