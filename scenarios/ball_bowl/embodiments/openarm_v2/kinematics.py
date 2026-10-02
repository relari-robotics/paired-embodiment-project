"""OpenArm v2 kinematic twins for the ball-and-bowl workcell."""

from __future__ import annotations

import numpy as np
from superdex import robotics

from superdex_scenarios.embodiments.base import ArmKinematics, EmbodimentModel
from superdex_scenarios.embodiments.openarm_v2 import OpenArmKinematics

from ...collision import BallBowlCollisionModel
from ...scenario import DESK_MIN, DESK_SIZE, ScenarioSpecification, contact_params

RIM_PITCH = np.radians(35.0)
"""Nose-down pitch of the left gripper for the rim pinch (keeps the wrist above the rim)."""


class OpenArmTwin(OpenArmKinematics):
    """Scenario adapter supplying ball-and-bowl obstacles to either OpenArm side."""

    def __init__(
        self,
        context: robotics.RoboticsContext,
        reference: EmbodimentModel,
        specification: ScenarioSpecification | None = None,
        *,
        side: str = "right",
        avoid_bowl: bool = True,
        approach_pitch: float = 0.0,
    ) -> None:
        specification = specification or ScenarioSpecification.fixed()
        super().__init__(
            context,
            reference,
            contact_params,
            BallBowlCollisionModel(
                specification, DESK_MIN, DESK_SIZE, avoid_bowl=avoid_bowl
            ),
            side=side,
            approach_pitch=approach_pitch,
        )


def right_arm(
    context: robotics.RoboticsContext,
    info: EmbodimentModel,
    specification: ScenarioSpecification,
) -> ArmKinematics:
    """The arm that delivers the ball; it plans around the bowl rim."""
    return OpenArmTwin(context, info, specification)


def left_arm(
    context: robotics.RoboticsContext,
    info: EmbodimentModel,
    specification: ScenarioSpecification,
) -> ArmKinematics:
    """The second arm: parked, or the one that drags the bowl to its target.

    Dragging means pinching the rim, so that twin ignores the bowl and pitches
    the gripper nose-down: with a level gripper the wrist flange sits at the
    height of the jaw pads and fouls the bowl wall behind the jaws.
    """
    if specification.bowl_target_xy is None:
        return OpenArmTwin(context, info, specification, side="left")
    return OpenArmTwin(
        context,
        info,
        specification,
        side="left",
        avoid_bowl=False,
        approach_pitch=float(RIM_PITCH),
    )


__all__ = ["RIM_PITCH", "OpenArmTwin", "left_arm", "right_arm"]
