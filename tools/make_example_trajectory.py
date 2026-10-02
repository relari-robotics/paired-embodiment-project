#!/usr/bin/env python
"""Generate the neutral example trajectories shipped under ``examples/``.

The motion is deliberately not a task solution: both grippers start open,
close, the right end effector traces a horizontal circle and the left one a
vertical circle above the desk while the grippers stay closed, and the
grippers open again. It exercises the file
format, the gripper convention, both arms, and the timing the controller
follows, without touching any object.

    python tools/make_example_trajectory.py --output-dir examples

writes ``examples/circles_right/trajectory.json`` (right arm only, for the
single-arm scenarios) and ``examples/circles_both/trajectory.json`` (both
arms, for the bimanual scenario). Positions are solved with the simulator's
own inverse kinematics at the level gripper orientation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from superdex_scenarios.replay.trajectory import JointTrajectory, PhaseMark  # noqa: E402

RATE_HZ = 50.0
OPEN_HOLD_S = 1.0
CLOSE_S = 1.0
CIRCLE_S = 6.0
OPEN_S = 1.0
CIRCLE_RADIUS = 0.06
CIRCLE_TURNS = 1.0
# Circle centres: in front of each parked gripper, well above the ball, bowl,
# plate, and sponge of every fixed layout (desk top is z = 0.388).
CIRCLE_CENTER = {"right": np.array([-0.04, -0.26, 0.54]), "left": np.array([-0.04, 0.26, 0.54])}


CIRCLE_PLANE = {"right": "horizontal", "left": "vertical"}


def _circle(side: str, phase: np.ndarray) -> np.ndarray:
    """Points on the side's circle; both start at the same relative point."""
    center = CIRCLE_CENTER[side]
    if CIRCLE_PLANE[side] == "horizontal":
        offsets = np.column_stack([np.cos(phase) - 1.0, np.sin(phase), np.zeros_like(phase)])
    else:  # vertical circle in the X-Z plane, in front of the robot
        offsets = np.column_stack([np.cos(phase) - 1.0, np.zeros_like(phase), np.sin(phase)])
    return center + CIRCLE_RADIUS * offsets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY_ROOT / "examples")
    args = parser.parse_args()

    from superdex import physics, robotics
    from tools.openarm_kinematics import _build

    physics.initialize(num_worker_threads=0)
    context = robotics.create_context()
    scene, info, twins = _build(context, ("right", "left"))
    try:
        home = info.default_pose[info.arm_dofs].copy()
        # Bring each arm from its parked pose to the circle start with a short
        # blend so the file begins at a pose the controller is already in.
        starts = {}
        for side, twin in twins.items():
            starts[side] = twin.solve(_circle(side, np.zeros(1))[0], home)

        total = OPEN_HOLD_S + CLOSE_S + CIRCLE_S + OPEN_S
        times = np.arange(0.0, total + 1e-9, 1.0 / RATE_HZ)
        arms = {side: np.zeros((len(times), 7)) for side in twins}
        aperture = np.zeros(len(times))
        groups = {side: np.isin(info.arm_dofs, info.dof_groups[f"{side}_arm"]) for side in twins}

        seed = {side: home.copy() for side in twins}
        for index, t in enumerate(times):
            if t < OPEN_HOLD_S:
                # Blend from the parked pose to the circle start with the gripper open.
                fraction = t / OPEN_HOLD_S
                fraction = fraction * fraction * (3.0 - 2.0 * fraction)
                pose = {side: (1.0 - fraction) * home + fraction * starts[side] for side in twins}
                aperture[index] = 1.0
            elif t < OPEN_HOLD_S + CLOSE_S:
                pose = {side: starts[side] for side in twins}
                aperture[index] = 1.0 - (t - OPEN_HOLD_S) / CLOSE_S
            elif t < OPEN_HOLD_S + CLOSE_S + CIRCLE_S:
                progress = (t - OPEN_HOLD_S - CLOSE_S) / CIRCLE_S
                phase = 2.0 * np.pi * CIRCLE_TURNS * progress
                pose = {}
                for side, twin in twins.items():
                    target = _circle(side, np.array([phase]))[0]
                    seed[side] = twin.solve(target, seed[side])
                    pose[side] = seed[side]
                aperture[index] = 0.0
            else:
                pose = {side: seed[side] for side in twins}
                aperture[index] = min(1.0, (t - OPEN_HOLD_S - CLOSE_S - CIRCLE_S) / OPEN_S)
            for side in twins:
                arms[side][index] = pose[side][groups[side]]

        phases = [
            PhaseMark("open", 0.0),
            PhaseMark("close", OPEN_HOLD_S),
            PhaseMark("circle", OPEN_HOLD_S + CLOSE_S),
            PhaseMark("release", OPEN_HOLD_S + CLOSE_S + CIRCLE_S),
        ]
        metadata = {
            "source": "tools/make_example_trajectory.py",
            "note": (
                "Neutral motion demo: open, close, one circle of each end effector "
                "above the desk (right horizontal, left vertical), open. Not a task solution."
            ),
            "circle_radius_m": CIRCLE_RADIUS,
            "circle_planes": CIRCLE_PLANE,
            "circle_centers_m": {side: c.tolist() for side, c in CIRCLE_CENTER.items()},
        }
        from superdex_scenarios.replay.trajectory import finger_joints_from_aperture

        both = JointTrajectory(
            times,
            {side: arms[side] for side in ("right", "left")},
            {side: finger_joints_from_aperture(aperture, side) for side in ("right", "left")},
            phases,
            dict(metadata, arms="both"),
        )
        right = JointTrajectory(
            times,
            {"right": arms["right"]},
            {"right": finger_joints_from_aperture(aperture, "right")},
            phases,
            dict(metadata, arms="right"),
        )
        for name, trajectory in (("circles_both", both), ("circles_right", right)):
            output = args.output_dir / name / "trajectory.json"
            trajectory.save(output)
            print(f"Wrote {output}: {trajectory.summary()}")
    finally:
        for twin in twins.values():
            twin.close()
        physics.destroy_scene(scene)
        physics.shutdown()


if __name__ == "__main__":
    main()
