"""Render a small material-review sequence from measured episode poses."""

import json
import sys
from pathlib import Path

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scenarios.tea_sorting import render  # noqa: E402

base = render.base
args = base.parse_args()
output = Path(args.output).resolve()
output.mkdir(parents=True, exist_ok=True)
base.clear_scene()
base.setup_render(args)
base.setup_world(args.environment, args.environment_strength * 0.45)
for name, energy in (("Key", 100), ("Fill", 30), ("Rim", 55)):
    bpy.data.lights[name].energy = energy
bpy.context.scene.view_settings.exposure = -0.7
bpy.context.scene.render.image_settings.color_depth = "16"
bpy.context.scene.cycles.adaptive_threshold = 0.008
bpy.context.scene.cycles.max_bounces = 12
episode = render.TeaEpisodeScene(Path(args.recording).resolve(), args)
base.add_floor()
shots = [
    ("01_initial_table", 0.0, episode.spec["cameras"]["table"], 40),
    (
        "02_red_tea_grasp",
        6.2,
        {"look_from": [0.20, -0.65, 0.76], "look_at": [-0.04, -0.38, 0.475]},
        58,
    ),
    (
        "03_blue_tea_transfer",
        44.3,
        {"look_from": [0.30, -0.26, 0.94], "look_at": [-0.025, 0.04, 0.47]},
        52,
    ),
    (
        "04_sorted_detail",
        60.1,
        {"look_from": [0.30, -0.16, 1.04], "look_at": [-0.025, 0.0, 0.412]},
        65,
    ),
    ("05_sorted_overhead", 60.1, episode.spec["cameras"]["tea_top"], 70),
]
manifest = []
for name, time_s, camera_spec, lens in shots:
    frame = int(np.argmin(np.abs(episode.times - time_s)))
    camera = base.EpisodeCamera(episode, name, camera_spec, lens)
    episode.apply_frame(frame)
    camera.apply_frame(frame)
    bpy.context.scene.render.filepath = str(output / f"{name}.png")
    bpy.ops.render.render(write_still=True)
    manifest.append(
        {
            "file": f"{name}.png",
            "recorded_frame": frame,
            "time_s": float(episode.times[frame]),
            "camera": camera_spec,
            "lens_mm": lens,
        }
    )
    print(f"REVIEW FRAME COMPLETE: {name} at {episode.times[frame]:.3f}s", flush=True)
    bpy.data.objects.remove(camera.camera, do_unlink=True)
(output / "frames.json").write_text(json.dumps(manifest, indent=2) + "\n")
