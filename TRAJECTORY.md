# Open-loop trajectory file format

A retargeting pipeline hands the simulator one JSON file per demonstration.
Every scenario runner replays it with `--replay FILE` through the same
compliant controller, physics, recording, and success check that drive the
scripted reference policies. The file is executed as given: there is no
feedback, no time warping, and no smoothing beyond linear interpolation
between samples at the 400 Hz physics rate.

The parser is [`superdex_scenarios/replay/trajectory.py`](superdex_scenarios/replay/trajectory.py)
(NumPy only). [`tools/trajectory_from_export.py`](tools/trajectory_from_export.py)
writes a valid file from any exported episode bundle and is the worked
example.

## Schema

```json
{
  "format": "openarm-joint-trajectory-v1",
  "time_s": [0.0, 0.02, 0.04],
  "right_arm": [[q1, q2, q3, q4, q5, q6, q7], "..."],
  "right_gripper": [1.0, 0.9, 0.8],
  "left_arm": [[q1, q2, q3, q4, q5, q6, q7], "..."],
  "left_gripper": [0.45, 0.45, 0.45],
  "phases": [{"name": "approach", "time_s": 0.0}, {"name": "grasp", "time_s": 1.3}],
  "metadata": {"anything": "you like"}
}
```

| Key | Required | Meaning |
| --- | --- | --- |
| `format` | yes | Must be `openarm-joint-trajectory-v1`. |
| `time_s` | yes | `N` strictly increasing sample times in seconds. The first value may be anything; the runner rebases it to zero. Any sample rate is accepted; 30 to 100 Hz is typical. |
| `right_arm`, `left_arm` | at least one | `N × 7` joint positions in radians, in joint order `arm_openarm_<side>_joint1` … `joint7`. |
| `right_gripper`, `left_gripper` | optional | `N` normalized apertures in `[0, 1]`: `0` is fully closed, `1` fully open. A gripper key needs the matching arm key. |
| `right_finger_joints`, `left_finger_joints` | optional | `N × 2` raw finger joint values in radians, an alternative to the aperture (never both). |
| `phases` | optional | Named marks with non-decreasing `time_s` inside the time range and unique names. They are logged to `phases.json` and printed with `SUPERDEX_PHASE_DEBUG=1`; they do not affect execution. |
| `metadata` | optional | Free-form object copied into `result.json`. |

Joints that a file does not command hold their default (parked) value: both
arms are motor-controlled in every scenario, so a file may command the right
arm, the left arm, or both, and an arm it does not mention stays parked. Values
outside the joint limits are clipped and reported in `result.json` under
`joint_limit_violations_rad`.

## Gripper convention

Both grippers are two-jaw. The aperture maps linearly to the finger joints:

```text
joint = open_sign[side] * 0.78 rad * aperture
open_sign = {"right": -1, "left": +1}
```

Useful reference apertures for the OpenArm v2 gripper:

| Aperture | Meaning |
| --- | --- |
| 1.00 | Fully open (about 75 mm between the jaw pads). |
| 0.45 | The parked "home" opening. |
| 0.26 | The reference policies' commanded closure on the 67 mm ball. The commanded position is deliberately inside the object; the compliant jaws stop on contact and squeeze. |
| 0.05 | Pinching a thin bowl rim. |
| 0.00 | Jaws touching. |

Closing an empty gripper below the object's size is how a grasp force is
produced; there is no separate force command.

## Timing

The runner holds the first sample for `--replay-settle` seconds (default 0.5)
so the arm settles at its start pose, tracks the file at `--replay-speed`
(default 1.0), then holds the last sample for `--replay-tail` seconds
(default 0.5). Large jumps between consecutive samples are executed as such:
the controller is compliant and effort-limited, so an unreachable rate shows
up as tracking error, not as an exception.

## What the runner reports

`result.json` (printed, written to `--export-dir` and to `--result PATH`):

```json
{
  "task": "ball_bowl",
  "mode": "replay",
  "kinematic": false,
  "objects_present": true,
  "completed": true,
  "success": true,
  "in_bowl": true,
  "final_ball_position_m": [-0.088, 0.023, 0.421],
  "replay": {
    "trajectory": {"samples": 676, "duration_s": 13.5, "sides": ["right"], "phases": ["..."]},
    "joint_limit_violations_rad": {},
    "tracking_error": {"right_arm": {"rms_rad": 0.004, "max_abs_rad": 0.018}}
  }
}
```

`success` is the task's own check (ball in bowl; bowl at target and ball in
bowl; plate coverage and sponge returned). It is `null` when it cannot be
evaluated (`--no-objects` or `--kinematic`). A replay whose `success` is
`false` exits with status 2.

## Coordinate frames

The simulation world is metres, X forward from the robot toward the far edge
of the desk, Y to the robot's left, Z up. The robot torso stands at
`(-0.45, 0, 0)`; the desk top is the plane `z = 0.388` spanning
`x ∈ [-0.36, 0.66]`, `y ∈ [-0.38, 0.38]`. The right arm reaches the region
around `y ≈ -0.23`, the left arm its mirror. The grasp point of each gripper
is 0.155 m from the wrist flange along the finger axis and coincides with the
centre of a held ball. `tools/openarm_kinematics.py` exports the joint tree and
evaluates forward/inverse kinematics for these frames.
