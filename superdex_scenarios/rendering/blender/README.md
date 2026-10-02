# Blender rendering

Offline photorealistic renders of recorded episodes: the scenario runner
records the scene, Blender rebuilds and path-traces it, ffmpeg encodes it.

```bash
# 1. Record (any scenario runner that supports --record-blender)
uv run --no-project python scenarios/ball_bowl/runner.py --human --fixed \
  --record-blender scenarios/ball_bowl/exports/blender_human_fixed

# 2. Render and encode
uv run --no-project python superdex_scenarios/rendering/blender/make_video.py \
  --recording scenarios/ball_bowl/exports/blender_human_fixed \
  --output scenarios/ball_bowl/exports/human_ball_bowl.mp4 \
  --fps 24 --width 1280 --height 720 --samples 40 --camera hand --lens 35
```

`make_video.py` finds Blender through `--blender`, the `BLENDER` variable,
`PATH`, or the macOS application bundle, runs `render_episode.py` inside it,
and passes through every option it does not recognise. Blender 4.2 or newer.

## Files

| File | Role |
| --- | --- |
| `recorder.py` | `BlenderSceneRecorder`: describes every actor once (GLB render model, physics surface mesh, or soft-body topology) and stores per-frame root transforms and soft surface nodes. Writes `scene.json` and `frames.npz`. |
| `render_episode.py` | Blender script. Rebuilds the scene: GLB render models with PBR material replacements for the OpenArm, textured scan overrides, procedural wood, ceramic, rubber, sponge, and floor materials, an HDRI world with key, fill, and rim lights, and Cycles on the GPU (or EEVEE). |
| `human_body.py` | Scanned hand, skinned arm and stationary person (see below). |
| `make_video.py` | Orchestrates Blender and ffmpeg. |

## Render options

| Option | Meaning |
| --- | --- |
| `--camera NAME` | Camera recorded by the runner. The sponge task provides fixed `front`/`top` views and moving `wrist_right`/`wrist_left` views, as well as its legacy presets. |
| `--lens MM` | Override focal length. Fixed presets default to 40 mm; calibrated cameras use their recorded intrinsics. |
| `--engine CYCLES\|EEVEE` | Path tracing (default) or the fast rasteriser |
| `--samples N` | Cycles samples, adaptive with denoising |
| `--width/--height` | Output size, default 1920x1080 |
| `--start/--end S` | Time range to render |
| `--frame-index N` | Render one recorded frame only |
| `--environment FILE` | Blender studio-light world HDR (`interior.exr` default, also `studio.exr`, `city.exr`, ...) |
| `--environment-strength` | World light strength, default 0.6 |
| `--save-blend FILE` | Save the built scene to open interactively |
| `--no-human-body` | Use the individual recorded human link meshes; omit the person |
| `--no-human-figure` | Keep the scanned hand replacement but omit the stationary person |

## Device selection and parallel rendering

The renderer selects the best available Cycles backend automatically:

- NVIDIA on Linux or Windows: OptiX, then CUDA.
- Apple on macOS: Metal.
- No compatible GPU: CPU fallback.

OptiX and Metal are separate native backends. OptiX cannot run on an Apple
Metal GPU, but callers use the same `--device AUTO` interface on both systems.
Use `--device OPTIX` or `--device METAL` to make a deployment fail loudly when
the expected backend is unavailable.

For an animation, use one Blender process per physical GPU. Workers render
contiguous frame ranges into the same directory, with globally ordered names,
and ffmpeg runs only after all workers finish:

```bash
# One RTX GPU on Linux/Windows.
.venv/bin/python superdex_scenarios/rendering/blender/make_video.py \
  --recording scenarios/organizer/exports/fixed \
  --output scenarios/organizer/exports/fixed.mp4 \
  --device OPTIX --workers 1 --camera front --samples 48

# Two visible GPUs on one host.
.venv/bin/python superdex_scenarios/rendering/blender/make_video.py \
  --recording scenarios/organizer/exports/fixed \
  --output scenarios/organizer/exports/fixed.mp4 \
  --device OPTIX --workers 2 --devices 0,1 --camera front --samples 48
```

`--devices` is required for multiple GPU workers so accidentally launching
several full Blender processes on one GPU is rejected. For separate cloud
machines, run the underlying worker directly once per machine with a different
worker index:

```bash
blender -b -P superdex_scenarios/rendering/blender/render_episode.py -- \
  --recording scenarios/organizer/exports/fixed \
  --output scenarios/organizer/exports/fixed-frames \
  --device OPTIX --device-index 0 \
  --worker-index 2 --worker-count 6 --camera front --samples 48
```

Keep the output directory or object-storage layout shared by worker, and
encode only after all frame files exist. The same command works with
`--device METAL` and `--device-index 0` on a macOS worker.

Use `--resume` when retrying a failed render. It skips non-empty PNGs while
preserving the same frame numbering. `--overwrite` takes precedence when both
flags are supplied.

## Standing human, arm and hand

Recordings with the Meta XR right-hand links receive a stationary, clothed
standing person by default. The Artec full-body scan supplies photographic
face/clothing/shoe detail. Its forward bend is baked into the mesh for the
current low table. The body is positioned once from frame 0 and stays fixed.

The working hand uses Artec's detailed hand scan, fitted to the recorded finger
joints and driven by explicit skin weights. Its creases and nail geometry are
preserved; it is **not** voxel-remeshed or globally smoothed. The hand scan has
no color texture, so the hand and authored arm use procedural skin shading.
The arm has oval sections, muscle contours, a rounded elbow, and a curved wrist
transition. The hand/arm attachment consists of overlapping surfaces, not a
watertight remesh.

When upper-arm and forearm tracks exist, they drive the arm. For hand-only
recordings, a two-segment visual arm reaches from a fixed shoulder to the
recorded wrist. This inferred elbow is for rendering only: it adds no simulated
body, collisions, or measured body motion. Targets beyond its normal reach
extend the visual arm. Fixed body placement cannot adapt to arbitrary walking
or large workspace changes. The default person faces world -X.

The interactive PBR viewer loads the same standing figure and a reduced-detail
version of the skinned arm/hand. It uses the same placement and elbow solve,
with simpler skin shading. Blender remains the lighting/detail reference.

Sources and licenses:

- [Artec full body scan](https://www.artec3d.com/3d-models/full-body-scan):
  `embodiments/assets/artec_figure/PROVENANCE.md` and `LICENSE.txt`.
- [Artec hand](https://www.artec3d.com/3d-models/hand):
  `embodiments/assets/artec_hand/PROVENANCE.md` and `LICENSE.txt`.

Both downloaded archives include CC BY 3.0 notices, retained with attribution.
The figure's working forearm was removed and the occluded clothing reconstructed.
The baked pose and garment patch are visual approximations; close views of the
sleeve/hip are less detailed than the original untouched scan.

Rebuild the figure with `tools/build_human_figure_assets.py --source PATH/full_body.obj`.
Rebuild the interactive arm after changes to the bind geometry by opening a
saved Blender review scene containing `human_body` and `human_armature`, then
running `tools/build_human_preview.py` inside Blender. The generated
`preview.glb` and `preview_bind.json` must be kept together. Shared skin settings
live in `rendering/human_appearance.json`.
