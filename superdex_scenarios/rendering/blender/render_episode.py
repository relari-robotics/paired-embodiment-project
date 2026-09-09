# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Render a recorded episode with Blender (run inside Blender's Python).

    blender -b -P render_episode.py -- --recording DIR --output DIR [options]

Reads ``scene.json`` and ``frames.npz`` written by
:class:`superdex_scenarios.rendering.blender.recorder.BlenderSceneRecorder`,
rebuilds the scene with the robot's GLB render models, physically based
materials, image-based lighting, and a Cycles (or EEVEE) camera, then renders
one PNG per output frame.  ``make_video.py`` wraps this and encodes the MP4.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import bpy  # type: ignore[import-not-found]
import numpy as np
from mathutils import Matrix, Quaternion, Vector  # type: ignore[import-not-found]

BLENDER_RESOURCES = Path(bpy.utils.resource_path("LOCAL")) / "datafiles" / "studiolights" / "world"


# --------------------------------------------------------------------------
# arguments
# --------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", required=True, help="directory with scene.json and frames.npz")
    parser.add_argument("--output", required=True, help="directory for frame_%%05d.png")
    parser.add_argument("--camera", default="plate", help="camera preset name from scene.json")
    parser.add_argument("--fps", type=float, default=24.0, help="output frame rate")
    parser.add_argument("--start", type=float, default=0.0, help="start time [s]")
    parser.add_argument("--end", type=float, default=None, help="end time [s] (default: all)")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--engine", choices=("CYCLES", "EEVEE"), default="CYCLES")
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--environment", default="interior.exr", help="Blender studio-light world HDR")
    parser.add_argument("--environment-strength", type=float, default=0.7)
    parser.add_argument("--lens", type=float, default=40.0, help="camera focal length [mm]")
    parser.add_argument("--frame-index", type=int, default=None, help="render only this recorded frame")
    parser.add_argument("--save-blend", default=None, help="also save the built scene as a .blend")
    return parser.parse_args(argv)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def matrix_from_vector(vector: np.ndarray) -> Matrix:
    """4x4 Blender matrix from ``[tx, ty, tz, qx, qy, qz, qw]``."""
    tx, ty, tz, qx, qy, qz, qw = (float(v) for v in vector)
    matrix = Quaternion((qw, qx, qy, qz)).to_matrix().to_4x4()
    matrix.translation = Vector((tx, ty, tz))
    return matrix


def new_material(name: str) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    return material


def principled(material: bpy.types.Material) -> bpy.types.Node:
    return material.node_tree.nodes["Principled BSDF"]


def set_input(node: bpy.types.Node, name: str, value) -> None:
    if name in node.inputs:
        node.inputs[name].default_value = value


def make_wood_material() -> bpy.types.Material:
    """Dark walnut: streaky noise stretched along the grain, light coat."""
    material = new_material("SuperDex Wood")
    tree = material.node_tree
    bsdf = principled(material)
    coords = tree.nodes.new("ShaderNodeTexCoord")
    mapping = tree.nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (1.0, 25.0, 1.0)
    grain = tree.nodes.new("ShaderNodeTexNoise")
    grain.inputs["Scale"].default_value = 4.0
    grain.inputs["Detail"].default_value = 10.0
    grain.inputs["Roughness"].default_value = 0.65
    ramp = tree.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.35
    ramp.color_ramp.elements[0].color = (0.11, 0.055, 0.022, 1.0)
    ramp.color_ramp.elements[1].position = 0.70
    ramp.color_ramp.elements[1].color = (0.30, 0.16, 0.07, 1.0)
    pores = tree.nodes.new("ShaderNodeTexNoise")
    pores.inputs["Scale"].default_value = 120.0
    pores.inputs["Detail"].default_value = 4.0
    bump = tree.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.08
    links = tree.links
    links.new(coords.outputs["Object"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], grain.inputs["Vector"])
    links.new(grain.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(pores.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    set_input(bsdf, "Roughness", 0.5)
    set_input(bsdf, "Coat Weight", 0.15)
    set_input(bsdf, "Coat Roughness", 0.2)
    return material


def make_ceramic_material(texture: Path | None) -> bpy.types.Material:
    material = new_material("SuperDex Ceramic")
    tree = material.node_tree
    bsdf = principled(material)
    if texture is not None and texture.exists():
        image = tree.nodes.new("ShaderNodeTexImage")
        image.image = bpy.data.images.load(str(texture))
        tree.links.new(image.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        set_input(bsdf, "Base Color", (0.92, 0.91, 0.86, 1.0))
    set_input(bsdf, "Roughness", 0.18)
    set_input(bsdf, "Coat Weight", 0.6)
    set_input(bsdf, "Coat Roughness", 0.05)
    return material


def make_sponge_material(color: list[float] | None) -> bpy.types.Material:
    """Yellow cellulose body with a darker green scouring layer on the top face."""
    material = new_material("SuperDex Sponge")
    tree = material.node_tree
    bsdf = principled(material)
    base = tuple(color) if color else (0.96, 0.72, 0.10)
    set_input(bsdf, "Roughness", 0.95)
    set_input(bsdf, "Subsurface Weight", 0.02)
    set_input(bsdf, "Subsurface Radius", (0.004, 0.003, 0.002))
    noise = tree.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 420.0
    noise.inputs["Detail"].default_value = 10.0
    noise.inputs["Roughness"].default_value = 0.8
    bump = tree.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 1.0
    bump.inputs["Distance"].default_value = 0.006
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    separate = tree.nodes.new("ShaderNodeSeparateXYZ")
    threshold = tree.nodes.new("ShaderNodeMath")
    threshold.operation = "GREATER_THAN"
    threshold.inputs[1].default_value = 0.85
    mix = tree.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.inputs["A"].default_value = (*base, 1.0)
    mix.inputs["B"].default_value = (0.12, 0.42, 0.18, 1.0)
    links = tree.links
    links.new(geometry.outputs["Normal"], separate.inputs["Vector"])
    links.new(separate.outputs["Z"], threshold.inputs[0])
    links.new(threshold.outputs["Value"], mix.inputs["Factor"])
    links.new(mix.outputs["Result"], bsdf.inputs["Base Color"])
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return material


def make_floor_material() -> bpy.types.Material:
    material = new_material("SuperDex Floor")
    bsdf = principled(material)
    set_input(bsdf, "Base Color", (0.42, 0.42, 0.43, 1.0))
    set_input(bsdf, "Roughness", 0.75)
    return material


def make_flat_material(name: str, color: list[float] | None) -> bpy.types.Material:
    material = new_material(name)
    bsdf = principled(material)
    rgb = tuple(color) if color else (0.6, 0.6, 0.6)
    set_input(bsdf, "Base Color", (*rgb, 1.0))
    set_input(bsdf, "Roughness", 0.5)
    return material


def mesh_object(name: str, vertices: np.ndarray, faces: np.ndarray) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(
        [tuple(map(float, v)) for v in vertices],
        [],
        [tuple(map(int, f)) for f in faces],
    )
    mesh.validate()
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def shade_smooth(obj: bpy.types.Object) -> None:
    for polygon in obj.data.polygons:
        polygon.use_smooth = True


def import_glb(path: str, name: str) -> bpy.types.Object:
    """Import a GLB and return an empty that parents everything it created."""
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    created = [o for o in bpy.data.objects if o not in before]
    root = bpy.data.objects.new(f"{name}/root", None)
    bpy.context.scene.collection.objects.link(root)
    for obj in created:
        if obj.parent is None or obj.parent not in created:
            obj.parent = root
    return root


def import_obj(path: str, name: str) -> bpy.types.Object:
    before = set(bpy.data.objects)
    bpy.ops.wm.obj_import(filepath=path, forward_axis="Y", up_axis="Z")
    created = [o for o in bpy.data.objects if o not in before]
    if len(created) == 1:
        created[0].name = name
        return created[0]
    root = bpy.data.objects.new(f"{name}/root", None)
    bpy.context.scene.collection.objects.link(root)
    for obj in created:
        obj.parent = root
    return root


# --------------------------------------------------------------------------
# scene construction
# --------------------------------------------------------------------------


def clear_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def setup_world(environment: str, strength: float) -> None:
    world = bpy.data.worlds.new("SuperDex World")
    bpy.context.scene.world = world
    world.use_nodes = True
    tree = world.node_tree
    background = tree.nodes["Background"]
    hdr = BLENDER_RESOURCES / environment
    if hdr.exists():
        env = tree.nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(str(hdr))
        mapping = tree.nodes.new("ShaderNodeMapping")
        mapping.inputs["Rotation"].default_value = (0.0, 0.0, math.radians(200.0))
        coords = tree.nodes.new("ShaderNodeTexCoord")
        tree.links.new(coords.outputs["Generated"], mapping.inputs["Vector"])
        tree.links.new(mapping.outputs["Vector"], env.inputs["Vector"])
        tree.links.new(env.outputs["Color"], background.inputs["Color"])
    else:
        background.inputs["Color"].default_value = (0.7, 0.72, 0.75, 1.0)
    background.inputs["Strength"].default_value = strength
    # Key light: a large soft area light above and to the camera's right.
    light_data = bpy.data.lights.new("Key", type="AREA")
    light_data.energy = 170.0
    light_data.size = 1.2
    light_data.color = (1.0, 0.97, 0.92)
    key = bpy.data.objects.new("Key", light_data)
    bpy.context.scene.collection.objects.link(key)
    key.location = (0.6, -0.9, 1.9)
    key.rotation_euler = (math.radians(35.0), 0.0, math.radians(30.0))
    fill_data = bpy.data.lights.new("Fill", type="AREA")
    fill_data.energy = 60.0
    fill_data.size = 2.0
    fill = bpy.data.objects.new("Fill", fill_data)
    bpy.context.scene.collection.objects.link(fill)
    fill.location = (-0.8, 1.2, 1.6)
    fill.rotation_euler = (math.radians(-40.0), 0.0, math.radians(-150.0))


def setup_camera(look_from: list[float], look_at: list[float], lens: float) -> bpy.types.Object:
    camera_data = bpy.data.cameras.new("Camera")
    camera_data.lens = lens
    camera_data.sensor_width = 36.0
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.scene.collection.objects.link(camera)
    eye = Vector(look_from)
    camera.location = eye
    direction = Vector(look_at) - eye
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = camera
    return camera


def setup_render(args: argparse.Namespace) -> None:
    scene = bpy.context.scene
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    if args.engine == "CYCLES":
        scene.render.engine = "CYCLES"
        preferences = bpy.context.preferences.addons["cycles"].preferences
        try:
            preferences.compute_device_type = "METAL"
        except TypeError:
            pass
        preferences.refresh_devices()
        gpu = False
        for device in preferences.devices:
            device.use = device.type != "CPU"
            gpu = gpu or device.use
        scene.cycles.device = "GPU" if gpu else "CPU"
        scene.cycles.samples = args.samples
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.adaptive_threshold = 0.02
        scene.cycles.use_denoising = True
        scene.cycles.max_bounces = 8
        scene.cycles.caustics_reflective = False
        scene.cycles.caustics_refractive = False
        print(
            "Cycles device:", scene.cycles.device,
            [(d.name, d.type, d.use) for d in preferences.devices],
        )
    else:
        scene.render.engine = "BLENDER_EEVEE"
        scene.eevee.taa_render_samples = max(16, args.samples)
        try:
            scene.eevee.use_raytracing = True
        except AttributeError:
            pass
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"


class EpisodeScene:
    """Blender objects built from a recording, updated per frame."""

    def __init__(self, recording: Path, args: argparse.Namespace) -> None:
        self.recording = recording
        self.spec = json.loads((recording / "scene.json").read_text(encoding="utf-8"))
        self.frames = np.load(recording / self.spec["frames_file"])
        self.transforms = self.frames["transforms"]
        self.times = self.frames["times"]
        self.objects: dict[int, bpy.types.Object] = {}
        self.local: dict[int, Matrix] = {}
        self.soft: dict[int, tuple[bpy.types.Object, np.ndarray]] = {}
        self.materials = {
            "wood": make_wood_material(),
            "floor": make_floor_material(),
        }
        self._build(args)

    def _material_for(self, entry: dict, color: list[float] | None) -> bpy.types.Material:
        name = entry.get("material") or "generic"
        if name == "wood":
            return self.materials["wood"]
        if name == "floor":
            return self.materials["floor"]
        if name == "ceramic":
            override = entry.get("override") or {}
            texture = override.get("texture")
            key = f"ceramic:{texture}"
            if key not in self.materials:
                self.materials[key] = make_ceramic_material(
                    self.recording / texture if texture and not Path(texture).is_absolute()
                    else (Path(texture) if texture else None)
                )
            return self.materials[key]
        if name == "sponge":
            key = f"sponge:{color}"
            if key not in self.materials:
                self.materials[key] = make_sponge_material(color)
            return self.materials[key]
        key = f"flat:{entry['name']}"
        if key not in self.materials:
            self.materials[key] = make_flat_material(key, color)
        return self.materials[key]

    def _build(self, args: argparse.Namespace) -> None:
        for entry in self.spec["actors"]:
            if entry.get("hidden"):
                continue
            index = entry["index"]
            kind = entry["kind"]
            name = entry["name"]
            override = entry.get("override") or {}
            if kind == "glb":
                glb = entry["glb"]
                obj = import_glb(glb["path"], name)
                self.local[index] = matrix_from_vector(np.asarray(glb["local_transform"]))
                sx, sy, sz = glb["scale"]
                if not np.allclose([sx, sy, sz], 1.0):
                    # GLB Y-up -> Z-up import permutes the scale axes.
                    obj.scale = (sx, sz, sy)
            elif kind == "mesh" or (kind == "empty" and "obj" in override):
                if "obj" in override:
                    obj_path = override["obj"]
                    if not Path(obj_path).is_absolute():
                        obj_path = str(self.recording / obj_path)
                    obj = import_obj(obj_path, name)
                    offset = override.get("offset", [0.0, 0.0, 0.0])
                    self.local[index] = Matrix.Translation(Vector(offset))
                    shade_smooth(obj) if obj.type == "MESH" else None
                else:
                    mesh = entry["mesh"]
                    obj = mesh_object(
                        name, self.frames[mesh["vertices"]], self.frames[mesh["faces"]]
                    )
                    self.local[index] = Matrix.Identity(4)
                material = self._material_for(entry, entry.get("color"))
                for target in ([obj] if obj.type == "MESH" else [c for c in obj.children if c.type == "MESH"]):
                    if target.data.materials:
                        target.data.materials[0] = material
                    else:
                        target.data.materials.append(material)
                if entry.get("material") == "floor":
                    obj.scale = (1.0, 1.0, 1.0)
            elif kind == "soft":
                soft = entry["soft"]
                nodes = self.frames[soft["nodes"]]
                faces = self.frames[soft["faces"]]
                obj = mesh_object(name, nodes[0], faces)
                shade_smooth(obj)
                subdivision = obj.modifiers.new("Smooth", type="SUBSURF")
                subdivision.levels = 2
                subdivision.render_levels = 2
                obj.data.materials.append(self._material_for(entry, entry.get("color")))
                self.soft[index] = (obj, nodes)
                continue  # world-space vertices: no per-frame transform
            else:
                continue
            self.objects[index] = obj

    def apply_frame(self, frame_index: int) -> None:
        row = self.transforms[frame_index]
        for index, obj in self.objects.items():
            obj.matrix_world = matrix_from_vector(row[index]) @ self.local[index]
        for index, (obj, nodes) in self.soft.items():
            obj.data.vertices.foreach_set("co", nodes[frame_index].astype(np.float32).ravel())
            obj.data.update()

    def frame_indices(self, fps: float, start: float, end: float | None) -> list[int]:
        duration = float(self.times[-1]) if len(self.times) else 0.0
        end = duration if end is None else min(end, duration)
        count = int(math.floor((end - start) * fps)) + 1
        targets = start + np.arange(count) / fps
        return [int(np.argmin(np.abs(self.times - t))) for t in targets]


def add_floor() -> None:
    bpy.ops.mesh.primitive_plane_add(size=12.0, location=(0.0, 0.0, -0.001))
    floor = bpy.context.active_object
    floor.name = "floor"
    floor.data.materials.append(make_floor_material())


def main() -> None:
    args = parse_args()
    recording = Path(args.recording).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    clear_scene()
    setup_render(args)
    setup_world(args.environment, args.environment_strength)
    episode = EpisodeScene(recording, args)
    camera = episode.spec["cameras"][args.camera]
    setup_camera(camera["look_from"], camera["look_at"], args.lens)
    add_floor()
    if args.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=str(Path(args.save_blend).resolve()))

    if args.frame_index is not None:
        indices = [args.frame_index]
    else:
        indices = episode.frame_indices(args.fps, args.start, args.end)
    print(f"Rendering {len(indices)} frames with {args.engine} at {args.width}x{args.height}")
    started = time.time()
    for output_index, frame_index in enumerate(indices):
        episode.apply_frame(frame_index)
        bpy.context.scene.render.filepath = str(output / f"frame_{output_index:05d}.png")
        bpy.ops.render.render(write_still=True)
        if output_index % 10 == 0 or output_index == len(indices) - 1:
            elapsed = time.time() - started
            print(
                f"  frame {output_index + 1}/{len(indices)} (recorded {frame_index}) "
                f"{elapsed:.0f}s elapsed, {elapsed / (output_index + 1):.1f}s/frame"
            )
    print(f"Frames written to {output}")


if __name__ == "__main__":
    main()
