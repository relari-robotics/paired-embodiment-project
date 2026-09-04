# Export bundle

`runner.py --headless` creates a unique complete bundle. `--export-dir` uses
`exports/latest`, and `--export-dir PATH` selects a destination. Add
`--skip-video` to retain physics data while skipping slow camera rendering.
`--dry-run` is the only headless validation mode that writes nothing.

```bash
uv run --no-project scenarios/ball_bowl/runner.py --headless
```

These commands describe the included OpenArm reference. The recording classes
are embodiment-neutral and available to human projects, but the blank human
hook does not prescribe when or how a project records its result.

## Files

| File | Contents |
| --- | --- |
| `scenario.json` | Sampled task and `embodiment` identifier |
| `episode.json` | Self-describing transform replay, cameras, manifest, and colors/scales |
| `telemetry.csv` | Dense 400 Hz state, controller force, contact wrench, and grip channels |
| `contacts.h5` | Sparse nonzero individual contact samples at physics rate |
| `telemetry_metadata.json` | DOF map, contact groups, definitions, cameras, and provenance |
| `telemetry_summary.json` | Peak torque/contact/grip values |
| `phases.json` | Semantic task phases (`home` … `return_home`) with step, time, grasp-point pose, and ball position at each boundary, plus point events such as `grasp_verified`; identical schema for every embodiment so paired episodes align phase by phase |
| `<camera>.mp4` | Deterministic PBR view for each available camera |
| `<camera>.json` | Intrinsics, pose/extrinsics, renderer, and encoding metadata |
| `ball_bowl.rrd` | Optional self-contained Rerun recording |

OpenArm exports `desk_zed`, `wrist_right`, and `wrist_left`. The human scaffold
exposes only `desk_zed`, because that embodiment has no modeled cameras.

## Torque and force definitions

`joint/*/motor_torque_nm` is the generalized joint-side effort applied by
Mochi's articulated pose controller. It is not electrical current and is not a
pre-gearbox shaft torque. Spherical human-hand joints are expanded into `/rx`,
`/ry`, and `/rz` DOFs; the metadata maps every column to its source joint.

`contact/<actor>/*` is the aggregate world-frame contact wrench. Force is in
newtons. Torque is in newton-metres about that actor's center of mass.
`contacts.h5` contains the lossless surface samples: query actor, actor pair,
world points, normal, force on `actor_a`, separation, point velocities,
quadrature sample index, and integration weight. Exact-zero broad-phase
candidates are omitted.

Logical ball grip channels are calculated from physical link-to-ball queries:

- OpenArm: force vector/norm for each jaw, two-jaw magnitude sum, single-jaw
  pinch equivalent, both hinge torques, and a unit-ratio coupled torque.
- Human: force vector/norm for each digit, aggregate magnitude sum, opposition
  equivalent `min(thumb, all opposing digits)`, all individual hand-joint
  torques, and their absolute sum.

## Cameras

The desk camera is a measured ZED-M calibration in
[`camera_calibration.json`](camera_calibration.json). The calibrated OpenCV
optical frame is transformed into the simulation world while preserving its
right/down/forward convention.

OpenArm wrist-camera metadata lives with the embodiment at
`superdex_scenarios/embodiments/config/openarm_v2_wrist_cameras.json`. Mount
extrinsics come from the imported camera-site actors. Lens intrinsics are
modeled from the upstream 66-degree vertical field of view and are explicitly
marked as modeled rather than photogrammetrically measured.

## Video and Rerun

Render one replay camera directly:

```bash
uv run --no-project superdex_scenarios/rendering/pbr/export_video.py \
  --episode scenarios/ball_bowl/exports/latest/episode.json \
  --camera desk_zed \
  --output scenarios/ball_bowl/exports/latest/desk_zed.mp4
```

The replay itself lists its available cameras, so the exporter has no
embodiment-specific camera list.

Build an RRD after generating a bundle with videos:

```bash
uv run --no-project scenarios/ball_bowl/export_rerun.py \
  --export-dir scenarios/ball_bowl/exports/latest
rerun scenarios/ball_bowl/exports/latest/ball_bowl.rrd
```

The Rerun exporter embeds the correct Studio render assets, actor transforms,
available videos, dense torque/contact series, and optional 3D force arrows. It
also works with `--skip-video` bundles; camera panes are simply omitted.
