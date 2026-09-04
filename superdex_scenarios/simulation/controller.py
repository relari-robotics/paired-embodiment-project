"""Generic compliant articulated-pose controller construction."""

from __future__ import annotations

from superdex import physics, robotics

from superdex_scenarios.embodiments import EmbodimentModel


def create_pose_controller(
    info: EmbodimentModel,
) -> tuple[robotics.ControllerBase, robotics.ControllerMochiArticulatedPoseTarget]:
    params = physics.PoseControllerParams(len(info.prefab.links))
    controlled_links = []
    for index, (link, joint) in enumerate(zip(info.prefab.links, info.prefab.joints)):
        tracking = info.tracking.get(link.name)
        if tracking is None or joint.type == physics.ArticulatedJointType.HARD:
            continue
        params.joint_tracking[index] = physics.PoseTrackingParams(
            stiffness=tracking.stiffness,
            damping=tracking.damping,
            saturation=tracking.saturation,
        )
        controlled_links.append(link.name)

    controller = info.bot.create_controller("MOCHI_ARTICULATED_POSE")
    controller.set_params(
        robotics.ControllerMochiArticulatedPoseParams(pose_controller_params=params)
    )
    controller.initialize(True)
    target = robotics.ControllerMochiArticulatedPoseTarget()
    target.world_from_root = info.actor.get_root_transform()
    print(
        f"Controller: {info.display_name} "
        f"({len(controlled_links)} controlled moving links)"
    )
    return controller, target
