# Assemble and pack an organizer

A rigid organizer sits on the same wooden table as `sponge_plate`. OpenArm
inserts two removable dividers into end guides, then sorts three colored parts
into the resulting three compartments. Both arms are motor-controlled: the
right handles the right and centre jobs, the left handles the left jobs. One
arm works at a time and returns home before the other starts.

The tray is fixed to the desk, matching the fixed plate in the sponge task.
Dividers and parts are dynamic bodies moved exclusively by jaw contact,
friction, and gravity. The broad divider tabs are an explicit gripper-compatible
design. There are no attachment constraints, scripted object motion, invisible
retention forces, or success flags set by the policy.

## Run

From the repository root, using the existing SuperDex Python environment:

```bash
# Inspect inverse-kinematics feasibility without stepping physics.
.venv/bin/python -m scenarios.organizer --fixed --plan-only

# Complete physical episode, no viewer and no output files.
.venv/bin/python -m scenarios.organizer --fixed --dry-run

# Interactive physics viewer.
.venv/bin/python -m scenarios.organizer --fixed

# Independently randomized initial positions, reproducible by seed.
.venv/bin/python -m scenarios.organizer --seed 1 --dry-run

# Supply explicit positions; see layouts/shifted.json.
.venv/bin/python -m scenarios.organizer \
  --layout scenarios/organizer/layouts/shifted.json --dry-run

# Record joints, contact forces, object poses, phases, and Blender assets.
.venv/bin/python -m scenarios.organizer --fixed \
  --export-dir scenarios/organizer/exports/fixed

# A still of the initial physical scene; front and top views are available.
.venv/bin/python -m scenarios.organizer --fixed --camera front \
  --snapshot scenarios/organizer/exports/initial.png
```

The existing container can run the same mounted source:

```bash
docker compose run --rm --entrypoint python superdex \
  -m scenarios.organizer --fixed --dry-run
```

`--headless` creates a timestamped export unless `--dry-run` is also supplied.
`--record-blender DIR` aliases `--export-dir DIR`. Failed physical episodes exit
with status 1 and, when exporting, retain their result and recordings. A valid
`--plan-only` result is an IK check, not a successful physics episode.

## Position-aware control

The scripted reference policy uses simulator object poses as privileged state.
It is suitable as a demonstration generator or baseline, not a trained visual
policy. No joint trajectory is replayed across layouts.

For each object it reads the current grasp site, constructs Cartesian
approach/lift/carry paths, and solves continuation IK from the measured robot
configuration. At contact targets, measured end-effector error corrects
compliant tracking sag. After lifting, the measured object-to-gripper offset
sets the placement target. Slot and compartment targets are transformed from
the actual tray pose. Parts are released above the compartment, with jaw
withdrawal coordinated with opening; the policy checks their settled geometry.

Free transit paths undergo conservative collision screening against the desk,
the individual tray/divider components, the other parts, and the parked arm.
Cartesian motion preserves the grasp orientation. Home departure and intended
contact segments use direct IK with physical collision response. This is a
bounded tabletop policy, not a general obstacle-routing planner. Diagnostic
`--no-collision-check` disables transit screening; the physical contacts remain.

Layouts specify world XY in metres, with +X forward and +Y toward the robot's
left. The table height stays fixed. Fields are `organizer_xy` (one pair),
`divider_xy` (two pairs), and `part_xy` (three pairs). Unknown fields, nonfinite
coordinates, overlapping initial footprints, and off-table objects are rejected.
Unreachable configurations fail explicitly. There is no silent resampling of a
failed layout. Initial rotations, different shapes, moving obstacles, and
arbitrary placements anywhere on the table are outside the validated scope.

Seeded variation independently moves the tray by ±12 mm per axis, each divider
by ±8 mm, and each part by ±10 mm. The explicit shifted layout exercises
different object and tray offsets. Targets and paths are recomputed throughout
execution; they are not restricted to those discrete examples.

## Geometry and physics

The tray is 160 × 285 mm, with an 8 mm floor and 25 mm walls. Two 140 mm
dividers have 20 mm thick low panels and exposed grip tabs. End guides provide
8 mm total clearance; this is a clearance-fit insertion, not a snap fit. The
parts are 40 × 45 × 60 mm. Divider/part masses are 90/45 g.

The task meshes are generated as closed box unions with outward-facing
triangles and no internal overlapping faces. No extra mesh downloads are
required beyond the project's existing OpenArm assets. SuperDex runs BDF2 at
400 Hz with at most 12 nonlinear solver iterations, using the shared compliant-contact
and pose-controller setup. Material/friction parameters are engineering choices;
they have not been calibrated to a purchased organizer. The current scenario
does not model divider flex, latches, a lid, or movable-tray support.

Success requires both dividers to be aligned, upright, and seated within 4 mm
vertically and within the slot clearance laterally. Each entire part bounding
box must lie inside its assigned compartment and rest on the floor. All objects
must have low linear and angular velocity. Success is checked after each
placement and again after the completed episode, so disturbing an earlier
placement causes final failure.

## Training exports and rendering

An export contains:

| File | Contents |
| --- | --- |
| `scenario.json` | Initial positions, seed, units, and modeling scope |
| `result.json` | Measured final outcomes, positions, velocities, completion/error |
| `phases.json` | Timestamped per-object phases and placement checks |
| `telemetry.npz` | Time, commanded/measured joint positions, actual object poses, net contact forces and forces from each finger |
| `scene.json`, `frames.npz` | Complete Blender scene description and rigid-body animation |

Telemetry and render frames are sampled every six physics steps (66.7 Hz).
Forces are instantaneous world-frame vectors in newtons at those samples, not
interval averages. `object_from_finger_force_world_n` has axes
`[sample, object, finger, xyz]`; object, finger, and joint names are included.
Quaternions are XYZW. Net object contact force is not the sum of opposing grip
magnitudes. Rendering can produce synchronized fixed and wrist camera images
from the recorded scene using the existing renderer:

```bash
.venv/bin/python superdex_scenarios/rendering/blender/make_video.py \
  --recording scenarios/organizer/exports/fixed \
  --output scenarios/organizer/exports/fixed.mp4 \
  --camera front --fps 24 --width 1280 --height 720 --samples 32
```

## Verification

```bash
.venv/bin/python -m unittest tests/test_organizer.py tests/test_organizer_outcome.py
.venv/bin/python tools/verify_organizer.py --seeds 0 1 2 3 4 \
  --output scenarios/organizer/exports/validation.json
```

The first command tests mesh closure/volume, invalid layouts, 100 reproducible
layout samples, and success checks using actual scene transforms. The second
runs complete physical episodes for the fixed scene, explicit shifted layout,
and each requested seed, reporting failures without discarding them.

Verified on 2026-09-14: all seven full physics cases (fixed, shifted, seeds
0–4) and all seven unit tests passed. The fixed recording also passed checks
for finite telemetry, increasing timestamps, nonzero finger-contact forces,
and final task success. This validates the tested layouts, not arbitrary table
placements or transfer to uncalibrated real hardware.
