# Sort wrapped tea by color

A separate scenario on the existing OpenArm table: red rooibos, yellow chamomile,
and blue Earl Grey sachets go into the matching compartments of a wooden tea box.
The original organizer task remains available unchanged as a separate task.

## Run

From the repository root, with the existing SuperDex environment:

```bash
.venv/bin/python -m scenarios.tea_sorting --fixed --headless --export-dir scenarios/tea_sorting/exports/demo
.venv/bin/python -m scenarios.tea_sorting --seed 3 --dry-run
.venv/bin/python -m scenarios.tea_sorting --layout scenarios/tea_sorting/layouts/shifted.json --dry-run
```

Omit `--headless` and the export directory to run the native physics viewer.
Its flat colors are for debugging; use the Blender renderer below for textured
images. Docker users can replace `.venv/bin/python` with
`docker compose run --rm --entrypoint python superdex` for simulation.

## High-fidelity rendering

The renderer replays the measured object and robot poses in `frames.npz`.
It does not reposition objects to make the outcome look successful.

```bash
/Applications/Blender.app/Contents/MacOS/Blender -b -P scenarios/tea_sorting/render.py -- \
  --recording scenarios/tea_sorting/exports/demo \
  --output scenarios/tea_sorting/exports/demo/detail \
  --camera detail --lens 65 --frame-index -1 \
  --engine CYCLES --samples 256 --width 3840 --height 2880 \
  --save-blend scenarios/tea_sorting/exports/demo/tea_sorting.blend
```

On other platforms use your `blender` executable. `--frame-index 0` shows the
initial arrangement. `--camera table` shows the workcell, `tea_top` gives an
overhead inspection view, and the recorded robot cameras remain available.
For video, omit `--frame-index`, add `--fps 24`, and encode the numbered PNGs:

```bash
ffmpeg -framerate 24 -i PATH_TO_RENDERED_FRAMES/frame_%05d.png -c:v libx264 -crf 18 -pix_fmt yuv420p tea_sorting.mp4
```

To reproduce the five review stills (initial table, grasp, transfer, sorted
detail, and overhead) from the fixed demonstration recording:

```bash
blender -b -P scenarios/tea_sorting/render_review.py -- \
  --recording scenarios/tea_sorting/exports/demo \
  --output scenarios/tea_sorting/exports/review_frames \
  --engine CYCLES --samples 128 --width 1920 --height 1440
```

Visual assets include:

- 8K photographed wood base color, roughness, and OpenGL normal maps from
  [Poly Haven](https://polyhaven.com/a/wood_table_001), mapped at physical scale.
- Original red/yellow/blue print artwork, UV-mapped without image resampling.
  The source atlas is 1942 × 809, approximately 647 × 809 per packet face.
- Pillow-shaped packet geometry, compressed perimeter seals, modeled crimp
  ridges and shallow wrinkles, plus fine paper roughness and micro-bump.
- Individually beveled box boards, color-coded enamel labels, brass pins,
  satin-coated presentation cradles, area lighting, and Cycles path tracing.
- 16-bit PNG output; a saved `.blend` packs its textures for portability.

See [assets/PROVENANCE.md](assets/PROVENANCE.md) for licenses, asset paths, and
the exact prompt used with the built-in image-generation tool. Full material
fidelity is currently in the Blender replay, not the native real-time viewer.

## Physics and policy

Packets are 62 × 75 mm with a 9.4 mm filled centre, 0.7 mm perimeter seal,
and a 4 g total mass. They use a closed, approximately 4 mm-spaced surface
mesh for contact. The wooden container has an 8 mm floor, 8 mm outer walls,
4 mm partitions, and 60 mm compartment height. Its signed-distance collision
grid is explicitly 1 mm: automatic mean-edge spacing was too coarse for these
thin walls. Low 18 mm presentation cradles keep the starting sachets accessible.

The motor policy reads each current packet pose and tea class, solves inverse
kinematics, closes the jaws, checks actual lift, compensates the measured grasp
offset, and releases over the corresponding box-relative compartment. Small
jaw opening and settling checks replace the original block-release motion.
The full physical packet surface must be contained and nearly stationary;
commanded targets are not counted as successful placement. Telemetry includes
measured/commanded joints, packet transforms, total contact forces, and contact
forces from each finger. There are no gripper attachments or object teleports.

`--seed` independently shifts the box by up to ±10 mm per axis and each packet
and its cradle by up to ±8 mm. Layout JSON accepts `organizer_xy`, `part_xy`, and
an optional `tea_order` permutation of `[0, 1, 2]`. Targets follow the actual box
pose. Off-table and overlapping layouts fail explicitly. This is bounded
translation robustness, not a guarantee for arbitrary positions or rotations;
the arm assignment assumes the three source regions shown in the default scene.

Important limits: this is a privileged-state scripted demonstration policy, not
a trained vision policy. The tea class comes from simulator metadata, not color
recognition. Packets are rigid contact bodies: paper bending, compression,
tearing, strings, and moving tea leaves are not simulated. Render wrinkles do
not deform; they differ from the physical surface by at most 0.23 mm. Friction
and packaging properties are plausible development values, not measurements
calibrated to a real packet. The box and source cradles are fixed to the table.

## Verify

```bash
.venv/bin/python -m unittest tests.test_tea_sorting tests.test_organizer tests.test_organizer_outcome
.venv/bin/python tools/verify_organizer.py --task tea_sorting --seeds 0 1 2 3 4 \
  --output scenarios/tea_sorting/exports/validation.json
```

The second command runs full contact-physics episodes for the fixed layout,
the explicitly shifted layout, and the requested seeds. It writes results as
each case completes and exits nonzero if any fails. It is not an IK-only test.

Verified locally on 2026-09-14: all 7 physics cases passed (fixed, shifted,
seeds 0–4), along with all 13 tea/organizer unit tests and Ruff checks.
Episodes took 55.6–60.1 simulated seconds. The local `exports/demo` bundle
includes initial, detail, and overhead renders plus a texture-packed `.blend`.
Exports are ignored by Git; rerun the commands above after a fresh checkout.
