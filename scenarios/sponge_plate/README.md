# Sponge wipes plate

A soft finite-element sponge cleans a rigid, scanned ceramic plate. The OpenArm
v2 reference picks the sponge up from the desk with its two-jaw gripper,
carries it over the plate, presses it into the dish floor, drags it through
three serpentine strokes, lifts off, puts the sponge back where it started, and
returns home. Nothing welds or parents the sponge to the gripper: it is held by
pinch friction and moved by contact alone, and it visibly squashes, bulges, and
shears while it is wiped.

Success is a **cleanliness map**: a 1 cm grid over the inner 7 cm of the dish
floor. A cell counts as cleaned once a sponge-to-plate contact sample has
crossed it with non-zero normal force while the sponge material slides faster
than 1 cm/s. The episode is clean when at least 85 % of the cells are wiped and
the sponge ends the episode resting on the desk near its start.

The scenario uses the same runner options, policy contract
(`EpisodePolicy`, `PolicyOptions`, `PlanningError`), recording classes, and
workcell desk as [`ball_bowl`](../ball_bowl); only the objects, the phases, and
the success check differ.

## Running

Natively (see the repository README for the venv and `SUPERDEX_ASSETS_PATH`):

```bash
uv run --no-project python scenarios/sponge_plate/runner.py --fixed --plan-only
uv run --no-project python scenarios/sponge_plate/runner.py --fixed --dry-run
uv run --no-project python scenarios/sponge_plate/runner.py --headless --skip-video --seed 1234
uv run --no-project python scenarios/sponge_plate/runner.py --fixed --skip-video \
  --export-dir scenarios/sponge_plate/exports/debug
```

Through the published container:

```bash
docker compose run --rm --entrypoint python superdex \
  scenarios/sponge_plate/runner.py --fixed --dry-run
```

### Watching the sponge deform

The interactive viewer renders the deforming sponge surface every frame:

```bash
uv run --no-project python scenarios/sponge_plate/runner.py --fixed
uv run --no-project python scenarios/sponge_plate/runner.py --fixed --camera plate --no-trajectory
uv run --no-project python scenarios/sponge_plate/runner.py --fixed --camera side --frames exports/frames --frame-every 8
```

`--camera` picks a preset (`workcell`, `plate`, `sponge`, `side`), `--no-trajectory`
hides the planned grasp-point curve, and `--frames DIR` saves a PNG every
`--frame-every` rendered frames so a run can be scrubbed afterwards. `--debugger`
streams the scene to the SuperDex Physics Debugger instead. Both need a
display and are not available in the headless container.

## Workcell

| Object | Model |
| --- | --- |
| Desk | Same wooden desk as the ball-and-bowl task |
| Plate | YCB object 029 scan, solidified at load time (see `plate.py` and `assets/ycb_029_plate/README.md`); static, ceramic friction 0.42 |
| Sponge | 10 × 6.5 × 4.5 cm neo-Hookean tetrahedral block (Kuhn-subdivided grid of about 1 cm, 385 nodes, 1 440 tets), Poisson 0.30, density 100 kg/m³, friction 0.85 |

A scanned plate is a shell a few millimetres thick; a soft body pressed onto it
tunnels through. `solidify_plate` samples the scan's top surface on a 3 mm grid
and extrudes it to a flat base 6 mm below the plate, keeping the dish profile and
rim while making the body thick everywhere. The heightfield also tells the
planner the floor height under every wipe point.

## Grasp and wipe

The gripper cannot reach the plate top-down, so it approaches with its nose
pitched 30° below level and pinches the upper half of the sponge (grasp point
3 cm above the sponge centre) with the jaws closed to the same angle that holds
the tennis-size ball. The jaw pads therefore stay above the dish while the
sponge bottom is driven 6 mm below the plate surface, which the compliant arm
turns into roughly 4–6 N of wiping force. Three strokes 6 cm long at Y offsets
of −4, 0, and +4 cm from the plate centre cover the inner dish; the sponge's
corners ride the gentle slope beyond it.

Free-space segments (approach, carry, carry back, return) are optimized with
the shared TrajOpt against the desk box, the plate rim ring, and the parked left
arm. Contact segments (pre-grasp, lift, press, strokes, release) are executed as
direct inverse-kinematics references so the pressing depth is honoured.

## Phases

`home, preshape, approach, pre_grasp, grasp, lift, carry, lower, wipe,
lift_off, carry_back, lower_back, release, retreat, return_home`, plus the point
events `grasp_verified`, `press_verified` (with the normal force), and one
`wipe_stroke` per stroke (with the coverage reached).

## Randomization

| Parameter | Distribution |
| --- | --- |
| Sponge Young's modulus | Continuous uniform 5–15 kPa, rounded to 100 Pa |
| Sponge colour | Uniform over yellow, green, blue, pink, orange |
| Sponge X/Y | Continuous uniform in `[-0.015, 0.030] × [-0.270, -0.210] m` |
| Plate X/Y | Continuous uniform in `[-0.080, -0.030] × [-0.025, 0.020] m` |

Samples that place the sponge inside the plate's footprint are rejected, and a
sample whose wipe points are out of the arm's reach raises `PlanningError` and
is resampled, exactly as in the ball-and-bowl task.

## Exports

`--headless` and `--export-dir` write the ball-and-bowl bundle
(`scenario.json`, `episode.json`, `telemetry.csv`, `contacts.h5`,
`telemetry_metadata.json`, `telemetry_summary.json`, `phases.json`) plus:

| File | Contents |
| --- | --- |
| `cleanliness.json` | Cell centres, which cells were cleaned and at which step, the thresholds, and a coverage-versus-step history |

Telemetry columns and summary keys that read `*_ball_*` in the ball-and-bowl
task read `*_sponge_*` here (for example `gripper/finger1_sponge_force_norm_n`).
The sponge's contact wrench columns carry its total contact force; the torque
columns are zero because soft actors report no contact torque.

`episode.json` records rigid actor transforms only. The sponge's deformed
surface is not captured, so the browser replay viewer does not show it and no
render manifest is set; use the interactive viewer or the debugger to watch the
deformation. MP4 export is not available for this scenario.

## Limitations

- The sponge is a homogeneous hyperelastic block: no fluid, no plastic set, no
  surface fibres. Its 1 cm mesh resolves gross squash and shear, not fine
  buckling.
- Cleanliness is a contact-and-motion proxy. It does not model soil, water,
  or scrubbing pressure thresholds beyond the tiny per-sample minimum.
- The plate is static; it cannot be pushed off the desk.
- Only the OpenArm embodiment is registered. The `EMBODIMENTS` table accepts a
  human hand entry with the same contract as the ball-and-bowl project.
