# Developing and debugging episode policies

This document is for maintainers and anyone modifying the scenarios, the
embodiments, or their control policies, and it collects the diagnostic
workflows for trajectory replays. The retargeting challenge itself needs only
the root [`README.md`](../../README.md), [`PROJECT.md`](../../PROJECT.md),
and [`TRAJECTORY.md`](../../TRAJECTORY.md).

All commands assume the native setup from the root README: `.venv` exists in
the package root and the SuperDex assets are next to the package or named by
`SUPERDEX_ASSETS_PATH`. They are written for `scenarios/ball_bowl`; the
`sponge_plate` runner accepts the same options.

## The episode policy contract

One runner per scenario drives every embodiment through the same pipeline:
scene construction, the compliant pose controller, physics stepping,
transform and telemetry recording, a semantic phase log, the task's success
check, and export. What differs per embodiment is an **episode policy** -- how
the grasp is solved, how the trajectory is planned, how the jaws close, and
how the object is carried and released.

The contract is `EpisodePolicy` in [`episode.py`](episode.py): `plan()`,
`trajectory_points()`, `home_pose()`, `preshape_pose()`, and `run(runner)`,
where the runner offers `follow`, `hold`, `phase`, `check_grasp_alignment`,
and `verify_physical_grasp`. Embodiments are registered in `EMBODIMENTS`
([`scenario.py`](scenario.py)) with their builder, kinematic twin, and policy
class, so `runner.py --embodiment <id>` runs any of them. The policy receives
the built scenario (live `scene`, sampled `specification`, physical
`bot_info`, `workcell`, and the neutral `kinematics` twin) plus the runner
options; once its `plan()` succeeds the shared pipeline records, checks, and
exports the episode. Raising `PlanningError` from `plan()` makes a randomized
run resample the task.

- **OpenArm v2 reference**
  ([`embodiments/openarm_v2/policy.py`](embodiments/openarm_v2/policy.py)):
  plans with collision-aware trajectory optimization and closes two parallel
  jaws. [`bimanual_policy.py`](embodiments/openarm_v2/bimanual_policy.py)
  drives both arms and is selected when the specification has a
  `bowl_target_xy` (see [`MOVE_BOWL.md`](MOVE_BOWL.md));
  [`../sponge_plate/openarm_policy.py`](../sponge_plate/openarm_policy.py)
  adds the pressed wiping strokes.
- **Trajectory replay** (`superdex_scenarios/replay/policy.py`): the policy
  every runner instantiates for `--replay FILE`. It expands the file into full
  articulation poses, clips to joint limits, tracks the file at the physics
  rate through `PoseExecutor.track`, logs the file's phase marks, and reports
  tracking error. `--kinematic` swaps the executor for `KinematicExecutor`,
  which sets poses without stepping physics.
- **Human right hand**
  ([`embodiments/human_right_hand/policy.py`](embodiments/human_right_hand/policy.py)): a blank
  scaffold kept from an earlier project; not used by the challenge.
- **Embodiments.** `openarm_v2_bimanual` (the default everywhere) motor-controls
  both arms; the single-arm reference policies drive the right arm and the
  named gripper poses leave the left gripper parked. `openarm_v2` (right side
  only, `--embodiment openarm_v2`) is kept for the original single-arm exports.

Everything specific to one body lives in its subfolder of
[`embodiments/`](embodiments): `embodiment.py` (builder, kinematic twins,
render manifest, policy classes), the policies, the OpenArm teleoperation
mapping, and the Studio render scene. `scenario.py`, `runner.py`,
`episode.py`, `collision.py`, and `cameras.py` are shared by all of them.

## Use the runner in stages

Start with the cheapest check and add physics only when the previous stage
passes:

| Goal | Command or option | What it isolates |
| --- | --- | --- |
| Validate a trajectory file | `--replay F --fixed --plan-only` | Parsing, joint groups, joint limits, and the grasp-point route without physics. |
| Watch a file | `--replay F --fixed` (no output mode) | The interactive viewer, useful for judging approach direction and gripper closure by eye. `--video PATH` renders a photorealistic MP4 without a display; `--camera` picks the viewpoint. |
| Check reachability and tracking | `--replay F --fixed --dry-run --no-objects` | Controller tracking with no contact; read `tracking_error` in the result. |
| Check the task | `--replay F --fixed --dry-run` | Contact, grasp, release, and the success check. |
| Validate the robot planner | `--fixed --plan-only` | IK, collision proxies, TrajOpt, and trajectory validation without controllers or contact physics. |
| Validate complete physics | `--fixed --dry-run` | Controller tracking, grasp contact, lift, release, and the final check without writing files. |
| Reproduce a layout | `--seed N` or `--layout FILE` | Reconstruct the exact sampled task printed by an earlier run, or place the objects explicitly. |
| Bypass trajectory optimization | `--no-trajopt` | Use direct Cartesian references to distinguish an optimizer failure from IK, scene, or physics problems. |
| Finish a failed reference episode | `--allow-failed-grasp` | Continue after a failed check so later state can be inspected. Replays never stop early. This flag must not be used to claim success. |

```bash
# Validate the deterministic plan without stepping physics.
uv run python scenarios/ball_bowl/runner.py --fixed --plan-only

# Execute the deterministic episode without writing an export.
uv run python scenarios/ball_bowl/runner.py --fixed --dry-run

# Reproduce and export one randomized task.
uv run python scenarios/ball_bowl/runner.py --headless --skip-video --seed 1234

# Write a deterministic diagnostic bundle to a known directory.
uv run python scenarios/ball_bowl/runner.py --fixed --skip-video \
  --export-dir scenarios/ball_bowl/exports/debug

# Turn that bundle's commanded joints into a trajectory file and replay it.
uv run python tools/trajectory_from_export.py scenarios/ball_bowl/exports/debug
uv run python scenarios/ball_bowl/runner.py --fixed --dry-run \
  --replay scenarios/ball_bowl/exports/debug/trajectory.json

# Stop at exceptions and inspect a replay with pdb.
uv run python -m pdb scenarios/ball_bowl/runner.py --fixed --dry-run --replay my.json
```

`--fixed` selects the regression layout of each scenario. Without `--fixed`, a
new seed is printed and included in the export. See
[`RANDOMIZATION.md`](RANDOMIZATION.md), [`MOVE_BOWL.md`](MOVE_BOWL.md),
and [`../sponge_plate/README.md`](../sponge_plate/README.md) for every sampled
parameter and the `--layout` keys.

## Reading the console output

The console output is itself a diagnostic report:

- Each TrajOpt line reports clearance before and after optimization and joint
  travel. A negative final clearance indicates a colliding plan.
- `arm tracking error` compares the simulated arm against the planned
  pre-grasp pose; a replay prints RMS and maximum tracking error per joint
  group at the end.
- `grasp: ball/grasp point` shows task-space alignment before closing;
  `bowl grasp: rim/left grasp point` is the bimanual task's equivalent.
- `physical grasp verified` proves that contact lifted the object above the
  desk; `bowl moved to ... at_target=True` proves the drag.
- The final episode line reports the object positions, the verdict, and
  unintended drift of any uncontrolled joints.
- Exports include `phases.json`, the semantic phase log every policy writes
  (set `SUPERDEX_PHASE_DEBUG=1` to print each phase boundary to the console),
  and `result.json` for replays.

## Kinematics tools

```bash
# Joint tree, link frames, axes, limits, and gripper conventions as JSON
# (a copy ships in superdex_scenarios/embodiments/config/openarm_v2_kinematics.json).
uv run python tools/openarm_kinematics.py export --output openarm_v2_kinematics.json

# Grasp-point poses along a trajectory file, one line per second.
uv run python tools/openarm_kinematics.py fk my.json --every 1.0

# Inverse kinematics for one grasp-point pose (level gripper if no quaternion).
uv run python tools/openarm_kinematics.py ik --side right --position 0.0 -0.23 0.42
```

From Python, `OpenArmKinematics` in
`superdex_scenarios/embodiments/openarm_v2.py` offers `solve(position, seed)`,
`solve_pose(position, quaternion_xyzw, seed)`, `grasp_point_pose(arm_pose)`,
and `link_transforms(pose)`; a built scenario exposes one twin per controlled
arm (`scenario.kinematics`, `scenario.left_kinematics`).
`superdex_scenarios/retargeting/demo.py` loads a recorded demonstration into
NumPy arrays.

## Optional tooling

- **PBR videos.** `--video PATH` (default renderer) and the camera MP4s of a
  full `--export-dir` bundle render with Three.js in headless Firefox:
  `firefox` and [`geckodriver`](https://github.com/mozilla/geckodriver/releases)
  on `PATH`, Node for a one-time `npm ci` in `superdex_scenarios/rendering/pbr`
  (run automatically), and `ffmpeg` (the bundled `imageio-ffmpeg` build is used
  when none is on `PATH`). Without them, pass `--video-renderer viewer` and
  `--skip-video`. `superdex_scenarios/rendering/pbr/export_video.py` renders any
  `episode.json` directly, from a calibrated camera (`--camera`) or a free
  viewpoint (`--look-from`, `--look-at`, `--fov`).
- **Rerun.** `superdex_scenarios/recording/export_rerun.py` needs
  `rerun-sdk==0.34.0`, which the root README installs.
- **Blender renders** of the sponge task: see
  [`../sponge_plate/README.md`](../sponge_plate/README.md).
- **Studio scenes.** See [`studio/README.md`](studio/README.md).

## Maintainer image build

To publish a new headless runtime, build [`Dockerfile`](../../Dockerfile)
from a parent directory containing both this package (as
`robotics-challenge`) and a neighboring `project_superdex` checkout; the
SuperDex assets are copied into the image at publication time.

```bash
cd ..
docker build -f robotics-challenge/Dockerfile \
  --target minimal -t robotics-challenge:local .
```

Physics assumptions and limitations are documented in
[`PHYSICS.md`](PHYSICS.md).
