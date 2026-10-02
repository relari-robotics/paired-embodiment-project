# Variant: move the bowl, then the ball into it

The bowl is a dynamic rigid body in every ball-and-bowl run except the
`--fixed` regression scene, which bolts it to the desk. This variant makes
moving it part of the task (so its bowl is movable even with `--fixed`): the bowl starts on the robot's left, the ball rests
on its right, the **left** arm has to bring the bowl to a target region near
the middle of the desk, and the **right** arm then has to put the ball into the
relocated bowl. It mirrors the recorded `move-the-bowl-and-ball` demonstration,
in which the person drags the bowl by its rim with one hand and drops the ball
in with the other.

It is the same scenario, scene, and runner. A specification with a
`bowl_target_xy` selects it; `--move-bowl` provides one (fixed or sampled), and
so does a `--layout` file with that key.

Success requires all of:

- the bowl centre within 6 cm of `bowl_target_xy`, resting on the desk
  (`|z - desk| < 2 cm`) and upright (tilt below 15 degrees);
- the ball inside the bowl *as it actually sits* (not at the nominal target);
- the episode completed.

Without a target only the last two apply.

## Running

```bash
python scenarios/ball_bowl/runner.py --fixed --move-bowl --plan-only
python scenarios/ball_bowl/runner.py --fixed --move-bowl --dry-run
python scenarios/ball_bowl/runner.py --move-bowl --headless --skip-video --seed 1234
python scenarios/ball_bowl/runner.py --fixed --move-bowl --skip-video \
  --export-dir scenarios/ball_bowl/exports/debug
python scenarios/ball_bowl/runner.py --fixed --move-bowl --dry-run --replay my_trajectory.json
python scenarios/ball_bowl/runner.py --fixed --move-bowl --replay my_trajectory.json --camera left
```

Every other runner option applies unchanged. A replay file for this variant
commands `right_arm`, `right_gripper`, `left_arm`, and `left_gripper` (see
[`TRAJECTORY.md`](../../TRAJECTORY.md)).

## Embodiment

The variant needs two arms, so it runs on `openarm_v2_bimanual` (the default):
the full OpenArm v2 with **both** seven-DOF arms and two-jaw grippers
gravity-enabled and motor-controlled (18 DOFs). `arm_dofs` is
`[right 7, left 7]` and `hand_dofs` is `[right 2, left 2]`; `dof_groups` names
them. `openarm_v2` and `human_right_hand` have no policy that moves the bowl
and refuse `--move-bowl`.

The scenario builds two kinematic twins
([`embodiments/openarm_v2/kinematics.py`](embodiments/openarm_v2/kinematics.py)):
`scenario.kinematics` (right arm, collision model with the bowl rim at its
*target*) and `scenario.left_kinematics` (left arm, desk and self-collision
only, gripper pitched 35 degrees nose-down). The embodiment's primary grasp
point stays on the right gripper; `runner.grasp_point_pose(link)` measures the
left one.

## Reference policy

[`embodiments/openarm_v2/bimanual_policy.py`](embodiments/openarm_v2/bimanual_policy.py)
is scripted, not retargeted:

1. Left arm: rise straight up from home, travel above the rim on its own
   (+Y) side while pitching the gripper 35 degrees nose-down, descend so the
   open jaws straddle the rim, pinch it, drag the bowl in a
   straight line to the target with the bowl still resting on the desk,
   open, rise, and return home.
2. Right arm: the single-arm reference motion planned against the bowl at
   its target (collision-aware TrajOpt), then the same grasp, lift, carry,
   release, and return.

The nose-down pitch matters: with the level gripper the wrist flange sits at
the same height as the jaw pads, so a level rim pinch fouls the bowl wall
behind the jaws.

## Phases

`episode.MOVE_BOWL_PHASES`: `home, preshape, bowl_approach, bowl_grasp,
bowl_move, bowl_release, bowl_retreat, approach, pre_grasp, grasp, lift, carry,
lower, release, retreat, return_home`, plus the point events `bowl_at_target`
and `grasp_verified`.

## Randomization

| Parameter | Distribution |
| --- | --- |
| Ball mass | Uniform over 50, 100, …, 500 g |
| Bowl mass | Continuous uniform 150–400 g |
| Ball / bowl colour | As without a target |
| Bowl shape | `measured` (the recorded bowl: 17.5 cm across, 7 cm tall) |
| Ball X/Y | Continuous uniform in `[-0.015, 0.030] × [-0.310, -0.190] m` |
| Bowl start X/Y | Continuous uniform in `[0.03, 0.08] × [0.17, 0.23] m` |
| Bowl target X/Y | Continuous uniform in `[-0.08, 0.01] × [-0.02, 0.05] m` |

The bowl start band keeps the bowl clear of the parked left gripper, which sits
at rim height; the target band is reachable by both arms. Samples the left arm
cannot reach raise `PlanningError` and are resampled.

## Layout file

`--layout FILE` accepts `bowl_target_xy` and `bowl_mass_kg` next to the usual
`ball_xy`, `bowl_xy` (where the bowl starts), `ball_mass_kg`, `ball_color`,
`bowl_color`, and `bowl_shape`. With a `bowl_target_xy` the remaining defaults
are those of `--fixed --move-bowl`.

## Exports

The usual bundle with both arms' joints. `scenario.json` records the bowl
start, mass, and target, and the bowl's transform is part of the replay.
