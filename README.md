# SuperDex scenarios

This repository contains a ball-to-bowl manipulation scenario simulated with
[SuperDex](https://github.com/facebookresearch/project_superdex). The simulation
models the workcell, robot or human hand, object contact, motion, and task
outcome using an unmodified SuperDex release.

The goal is to create matched examples of the same task performed by two
embodiments. A working OpenArm robot episode provides the reference; the project
is to make a physical human-hand model perform a comparable motion. The hand
pose and trajectory should remain comparable to the robot while still looking
reasonably natural and being fully kinematically feasible. These paired episodes
can then form a synthetic dataset for studying the same behavior across robot
and human embodiments.

The code runs from a prebuilt Docker image while the repository is mounted into
the container. Contributors can clone the repository, edit the source locally,
and run their changes without compiling SuperDex or rebuilding the image.

## Scenario: ball into bowl

The scenario places a ball and a bowl on a desk. Each embodiment must start at
home, approach the ball from the side with the gripper or palm parallel to the
table, grasp it through physical contact, carry it to the bowl, release it, and
return home. The task can use a fixed configuration for development or
randomized object positions for dataset generation and robustness checks.

One runner drives every embodiment through the same pipeline: scene
construction, the compliant pose controller, physics stepping, transform and
telemetry recording, a semantic phase log, the in-bowl success check, and
export. What differs per embodiment is an **episode policy** -- how the grasp
is solved, how the trajectory is planned, how the jaws or fingers close, and
how the ball is carried and released:

- **OpenArm v2 reference** ([`openarm_policy.py`](scenarios/ball_bowl/openarm_policy.py)):
  a complete policy that plans with collision-aware trajectory optimization and
  closes two parallel jaws.
- **Human Meta XR hand project** ([`human_project.py`](scenarios/ball_bowl/human_project.py)):
  a physical 27-DOF hand on a six-DOF Cartesian carrier in the same workcell.
  Its policy is intentionally blank; the grasp, wrist trajectory, finger
  control, and carry strategy are the project.

The contract both implement is `EpisodePolicy` in
[`episode.py`](scenarios/ball_bowl/episode.py): `plan()`, `trajectory_points()`,
`home_pose()`, `preshape_pose()`, and `run(runner)`, where the runner offers
`follow`, `hold`, `phase`, `check_grasp_alignment`, and
`verify_physical_grasp`. Embodiments are registered in `EMBODIMENTS`
([`scenario.py`](scenarios/ball_bowl/scenario.py)) with their builder,
kinematic twin, render manifest, and policy class; camera metadata comes from
the shared scenario calibration and built embodiment model. This lets
`runner.py --embodiment <id>` run either one. The contract does not prescribe
an algorithm: optimization, motion capture, learning, teleoperation, and other
approaches are all valid. See [`PROJECT.md`](PROJECT.md) for the detailed task
goals and success criteria.

The old `scenarios/openarm_ball_bowl` import and executable paths remain as
compatibility shims.

## Scenario: sponge wipes plate

[`scenarios/sponge_plate`](scenarios/sponge_plate) is a second task on the same
workcell and runner contract: a soft finite-element sponge is picked up, pressed
onto a scanned ceramic plate, dragged through three serpentine strokes until a
cleanliness map of the dish floor is wiped, and put back on the desk. The
sponge is held by pinch friction alone and visibly squashes and shears. See its
[README](scenarios/sponge_plate/README.md) for the physics, randomization,
exports, the viewer and MP4 options for watching the deformation, and the
Blender pipeline for photorealistic renders:

```bash
docker compose run --rm --entrypoint python superdex scenarios/sponge_plate/runner.py --fixed --dry-run
```

## Scenario: assemble and pack an organizer

[`scenarios/organizer`](scenarios/organizer/README.md) uses the same table and
OpenArm to insert two physical dividers and sort three parts into compartments.
Its state-based policy recomputes grasps, paths, and placement targets from
actual poses, supports seeded position variation and explicit layout files,
and exports joint trajectories, contact forces, object motion, and Blender
recordings. The tray is fixed; the dividers and parts move through contact.

```bash
docker compose run --rm --entrypoint python superdex -m scenarios.organizer --fixed --dry-run
```

## Scenario: sort wrapped tea

[`scenarios/tea_sorting`](scenarios/tea_sorting/README.md) sorts red, yellow, and
blue wrapped tea sachets into a wooden container on the same table. It combines
a pose-relative physical grasping policy with a dedicated Cycles replay renderer:
8K photographed wood, printed sachets, paper microtexture, heat seals, and creases.
Packets currently use rigid contact physics; paper bending is not simulated.

```bash
docker compose run --rm --entrypoint python superdex -m scenarios.tea_sorting --fixed --dry-run
```

## Setup

Install Git, Docker, and Docker Compose, then clone the repository and download
the published image. No Python environment, C++ compiler, SuperDex checkout,
Azure login, or local image build is needed.

```bash
git clone https://github.com/relari-robotics/superdex-scenarios.git
cd superdex-scenarios
docker compose pull superdex
```

The public image is
`relaripublic.azurecr.io/superdex/scenarios-dev:1.0.0`. It is Linux x86-64 and
runs natively on x86-64 Linux. [`compose.yaml`](compose.yaml) selects
`linux/amd64`, allowing Docker Desktop to run it through x86-64 emulation on an
Apple Silicon Mac. SuperDex 1.0.0 does not provide Linux ARM64 binaries.

## Usage

### First run

Check the installation by constructing the fixed OpenArm scene and validating
its trajectory plan without running physics:

```bash
docker compose run --rm superdex --fixed --plan-only
```

A complete headless reference episode that writes no files is:

```bash
docker compose run --rm superdex --fixed --dry-run
```

Run a randomized episode and save a complete data bundle with the image's default command:

```bash
docker compose run --rm superdex
```

### How the container command works

The general form is:

```text
docker compose run [Compose options] superdex [runner options]
```

| Part | Meaning |
| --- | --- |
| `docker compose run` | Create a new one-off container for this invocation. |
| `--rm` | Delete that container when it exits. It does not delete host files or the downloaded image. |
| `superdex` | Select the service in `compose.yaml`. It runs the OpenArm reference by default, or another registered embodiment with `--embodiment <id>` (`--human` is shorthand for the human hand). |
| runner options | Passed to `scenarios/ball_bowl/runner.py`, the image entry point. |

Compose bind-mounts the repository root at `/opt/scenarios`. The Python process therefore executes the files in the checkout, not the snapshot baked into the image. Saving a source file is enough; the next command uses that change without rebuilding or restarting anything. Export files written below `scenarios/ball_bowl/exports` also remain in the checkout after `--rm` removes the container.

With no runner options, Docker uses the image default:

```text
--headless --skip-video
```

Supplying any runner options replaces that entire default. Consequently, use an explicit output mode such as `--plan-only`, `--dry-run`, or `--headless --skip-video` in custom OpenArm commands. Running only `superdex --fixed`, for example, requests the GUI viewer and is not supported by the published headless image.

Useful container-management commands are:

```bash
# Download the pinned image, or refresh it if the registry copy changed.
docker compose pull superdex

# Show the locally downloaded image.
docker image inspect relaripublic.azurecr.io/superdex/scenarios-dev:1.0.0

# Remove stopped one-off containers and the Compose network, if present.
docker compose down --remove-orphans
```

### OpenArm examples

```bash
# Show every runner option.
docker compose run --rm superdex --help

# Validate the deterministic plan without stepping physics.
docker compose run --rm superdex --fixed --plan-only

# Execute the deterministic episode without writing an export.
docker compose run --rm superdex --fixed --dry-run

# Execute and export a new randomized episode using the image defaults.
docker compose run --rm superdex

# Reproduce and export one randomized task.
docker compose run --rm superdex --headless --skip-video --seed 1234

# Write a deterministic diagnostic bundle to a known directory.
docker compose run --rm superdex --fixed --skip-video \
  --export-dir scenarios/ball_bowl/exports/debug
```

`--fixed` selects the blue-ball/gray-bowl regression case. Without `--fixed`, a
new seed is printed and included in the export. Pass that seed back with
`--seed` to reproduce the task. See
[`RANDOMIZATION.md`](scenarios/ball_bowl/RANDOMIZATION.md) for every sampled
parameter.

### Human-hand examples

Read [`PROJECT.md`](PROJECT.md), then verify the untouched physical scaffold:

```bash
docker compose run --rm superdex --human --fixed --plan-only
```

Implement `HumanPolicy` in [`human_project.py`](scenarios/ball_bowl/human_project.py), using [`openarm_policy.py`](scenarios/ball_bowl/openarm_policy.py) as the worked reference for the same contract. Before it is implemented, `--plan-only` reports that the scene was built and the policy is unimplemented, and a normal human invocation raises `NotImplementedError` by design:

```bash
docker compose run --rm superdex --human --fixed
```

After the fixed task works, exercise the same implementation on reproducible randomized tasks:

```bash
docker compose run --rm superdex --human --seed 1234
docker compose run --rm superdex --human --seed 5678
```

The policy receives the built scenario (live `scene`, sampled `specification`,
physical `bot_info`, `workcell`, and neutral `kinematics`) plus the runner
options, and once its `plan()` succeeds the shared pipeline records, checks and
exports the human episode exactly as it does the robot's (`--headless`,
`--export-dir`, `--record-pbr`, `--snapshot`, `--debugger` all apply). Raising
`PlanningError` from `plan()` makes a randomized run resample the task. The
project does not require modifying SuperDex or rebuilding the container.

## Diagnostic tools

### Use the runner in stages

Start with the cheapest check and add physics only when the previous stage passes:

| Goal | Command or option | What it isolates |
| --- | --- | --- |
| Build the human model | `--human --fixed --plan-only` | Asset loading, workcell construction, and the project boundary: it calls `HumanPolicy.plan()` and reports "policy unimplemented" until the project provides one. |
| Validate the robot planner | `--fixed --plan-only` | IK, collision proxies, TrajOpt, and trajectory validation without controllers or contact physics. |
| Validate complete robot physics | `--fixed --dry-run` | Controller tracking, grasp contact, lift, release, and the final in-bowl check without writing files. |
| Reproduce a failure | `--seed N` | Reconstruct the exact sampled task printed by an earlier run. |
| Bypass trajectory optimization | `--no-trajopt` | Use direct Cartesian references to distinguish an optimizer failure from IK, scene, or physics problems. |
| Finish a failed diagnostic episode | `--allow-failed-grasp` | Continue after a failed lift or final placement so later state can be inspected. This flag must not be used to claim success. |

The OpenArm console output is itself a diagnostic report:

- Each TrajOpt line reports clearance before and after optimization and joint
  travel. A negative final clearance indicates a colliding plan.
- `right-arm tracking error` compares the simulated arm against the planned
  pre-grasp pose.
- `grasp: ball/gripper point` shows task-space alignment before closing.
- `physical grasp verified` proves that contact lifted the ball above the desk.
- The final episode line reports the ball position, `in_bowl`, and unintended
  drift of the parked side.
- Exports include `phases.json`, the semantic phase log both embodiments write
  (set `SUPERDEX_PHASE_DEBUG=1` to print each phase boundary to the console).

### Inspect an export bundle

Create a repeatable bundle without the unsupported video exporter:

```bash
docker compose run --rm superdex --fixed --skip-video \
  --export-dir scenarios/ball_bowl/exports/debug
```

The directory contains the sampled scenario, transform replay, dense 400 Hz telemetry, individual contact samples, the task phase log, metadata, and a summary. Pretty-print a JSON report using Python from the same image:

```bash
docker compose run --rm --entrypoint python superdex -m json.tool \
  scenarios/ball_bowl/exports/debug/telemetry_summary.json
```

See [`EXPORTS.md`](scenarios/ball_bowl/EXPORTS.md) for the file and signal schema. The most useful first-pass files are `scenario.json`,
`telemetry_summary.json`, and `telemetry.csv`; use `contacts.h5` when individual contact positions and forces matter.

### Run tests, Python's debugger, or a shell

The image entry point normally launches the scenario runner. Compose's
`--entrypoint` option replaces it so other tools can run in the same environment:

```bash
# Run the repository's dependency-free unit tests.
docker compose run --rm --entrypoint python superdex \
  -m unittest discover -s tests -v

# Stop at exceptions and inspect a human run with pdb.
docker compose run --rm --entrypoint python superdex \
  -m pdb scenarios/ball_bowl/runner.py --human --fixed

# Open an interactive shell with the checkout mounted at /opt/scenarios.
docker compose run --rm --entrypoint /bin/sh superdex
```

Useful `pdb` commands are `n` (next line), `s` (step into), `p expression`
(print), `l` (list source), `c` (continue), and `q` (quit).

### View a recorded episode in a browser

The replay viewer is static HTML and Three.js. First create the `debug` export shown above, then install its locked browser dependency using a disposable Node container:

```bash
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PWD:/workspace" \
  -w /workspace/superdex_scenarios/rendering/pbr \
  node:22-alpine npm ci --cache /tmp/npm-cache
```

Serve the checkout with the existing Python image:

```bash
docker compose run --rm -p 8765:8765 --entrypoint python superdex \
  -m http.server 8765 --bind 0.0.0.0 --directory /opt/scenarios
```

While that command is running, open:

```text
http://localhost:8765/superdex_scenarios/rendering/pbr/?recording=/scenarios/ball_bowl/exports/debug/episode.json
```

The browser provides play/pause, timeline scrubbing, playback speed, orbit controls, and the cameras present in the replay.

### Published-image limitations

The candidate image is deliberately headless and minimal:

- `--debugger`, `--snapshot`, and the native interactive viewer require a host display and are not supported.
- MP4 generation is not installed. Always combine `--headless` or   `--export-dir` with `--skip-video`.
- The optional Rerun SDK is not installed. The JSON, CSV, HDF5, console, `pdb`, and browser-replay paths above require no image rebuild.
- Adding a new Python or operating-system dependency does require a new image;  ordinary edits and new source files in this repository do not.

## Maintainer image build

Candidates should not use this workflow. To publish a new runtime, maintainers build [`Dockerfile`](Dockerfile) from a parent directory containing both this repository and a neighboring `project_superdex` checkout; the SuperDex robot assets are copied into the image at publication time.

```bash
cd ..
docker build -f superdex-scenarios/Dockerfile \
  --target minimal -t superdex-scenarios:local .
```

Physics assumptions and limitations are documented in
[`PHYSICS.md`](scenarios/ball_bowl/PHYSICS.md).
