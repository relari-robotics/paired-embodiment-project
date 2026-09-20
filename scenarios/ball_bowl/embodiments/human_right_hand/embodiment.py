"""How the human right arm is built into the ball-and-bowl workcell."""

from __future__ import annotations

from superdex import robotics

from superdex_scenarios.embodiments.base import ArmKinematics, EmbodimentModel
from superdex_scenarios.embodiments.human_right_arm import (
    HumanArmKinematics,
    build_human_right_arm,
    destroy_human_right_arm,
)

from ...collision import BallBowlCollisionModel
from ...scenario import (
    DESK_MIN,
    DESK_SIZE,
    EmbodimentSpec,
    ScenarioSpecification,
    contact_params,
)


def _kinematics(
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
    "human_right_hand": EmbodimentSpec(
        embodiment_id="human_right_hand",
        scene_name="Human right hand: randomized ball into bowl",
        render_manifest=(
            "/scenarios/ball_bowl/embodiments/human_right_hand/studio/"
            "human_ball_bowl_studio.mochi_scene"
        ),
        solver_max_iter=10,
        build=lambda scene, context: build_human_right_arm(
            scene, context, contact_params
        ),
        destroy=destroy_human_right_arm,
        kinematics=_kinematics,
        policy="scenarios.ball_bowl.embodiments.human_right_hand.policy:HumanPolicy",
    ),
}

__all__ = ["EMBODIMENTS"]
