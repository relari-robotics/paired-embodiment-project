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

try:
    from .device import DEVICE_AUTO, SUPPORTED_DEVICES, configure_cycles_device
    from .render_plan import partition_frames
except ImportError:  # Blender executes this file as a standalone script.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from device import DEVICE_AUTO, SUPPORTED_DEVICES, configure_cycles_device
    from render_plan import partition_frames

BLENDER_RESOURCES = Path(bpy.utils.resource_path("LOCAL")) / "datafiles" / "studiolights" / "world"


# --------------------------------------------------------------------------
# arguments
# --------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", required=True, help="directory with scene.json and frames.npz")
    parser.add_argument("--output", required=True, help="directory for frame_%%05d.png")
    parser.add_argument("--camera", default="front", help="camera name from scene.json")
    parser.add_argument("--fps", type=float, default=24.0, help="output frame rate")
    parser.add_argument("--start", type=float, default=0.0, help="start time [s]")
    parser.add_argument("--end", type=float, default=None, help="end time [s] (default: all)")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--engine", choices=("CYCLES", "EEVEE"), default="CYCLES")
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--environment", default="interior.exr", help="Blender studio-light world HDR")
    parser.add_argument("--environment-strength", type=float, default=0.7)
    parser.add_argument(
        "--device",
        choices=SUPPORTED_DEVICES,
        default=DEVICE_AUTO,
        type=str.upper,
        help="Cycles backend: AUTO, OPTIX, CUDA, METAL, or CPU",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=None,
        help="GPU index for this worker; omit to use all visible GPUs",
    )
    parser.add_argument(
        "--lens",
        type=float,
        default=None,
        help=(
            "override camera focal length [mm] "
            "(default: 40 for presets, calibrated for robot cameras)"
        ),
    )
    parser.add_argument("--frame-index", type=int, default=None, help="render only this recorded frame")
    parser.add_argument("--worker-index", type=int, default=0, help="zero-based frame worker index")
    parser.add_argument("--worker-count", type=int, default=1, help="number of frame workers")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip output frames that already exist and are non-empty",
    )
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing output frames")
    parser.add_argument(
        "--hide-robot",
        action="store_true",
        help=(
            "hide articulated robot render meshes while retaining their recorded "
            "transforms for actor-mounted cameras"
        ),
    )
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
    mapping.inputs["Scale"].default_value = (1.0, 12.0, 1.0)
    grain = tree.nodes.new("ShaderNodeTexNoise")
    grain.inputs["Scale"].default_value = 6.0
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
    """Uniformly coloured cellulose sponge material."""
    material = new_material("SuperDex Sponge")
    tree = material.node_tree
    bsdf = principled(material)
    base = tuple(color) if color else (0.96, 0.72, 0.10)
    set_input(bsdf, "Base Color", (*base, 1.0))
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
    links = tree.links
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


def _noise_roughness(tree, bsdf, base: float, spread: float, scale: float) -> None:
    """Modulate roughness with fine noise so metal and plastic are not perfectly uniform."""
    noise = tree.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = scale
    noise.inputs["Detail"].default_value = 6.0
    ramp = tree.nodes.new("ShaderNodeMapRange")
    ramp.inputs["From Min"].default_value = 0.35
    ramp.inputs["From Max"].default_value = 0.65
    ramp.inputs["To Min"].default_value = base - spread
    ramp.inputs["To Max"].default_value = base + spread
    tree.links.new(noise.outputs["Fac"], ramp.inputs["Value"])
    tree.links.new(ramp.outputs["Result"], bsdf.inputs["Roughness"])


def _micro_bump(tree, bsdf, scale: float, strength: float, distance: float = 0.0004) -> None:
    noise = tree.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = scale
    noise.inputs["Detail"].default_value = 8.0
    noise.inputs["Roughness"].default_value = 0.75
    bump = tree.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = strength
    bump.inputs["Distance"].default_value = distance
    tree.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    tree.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])


def make_robot_materials() -> dict[str, bpy.types.Material]:
    """Physically based replacements for the flat OpenArm GLB materials, by name prefix."""
    plastic = new_material("OpenArm Plastic")
    bsdf = principled(plastic)
    set_input(bsdf, "Base Color", (0.020, 0.020, 0.023, 1.0))
    set_input(bsdf, "Specular IOR Level", 0.45)
    _noise_roughness(plastic.node_tree, bsdf, 0.52, 0.10, 90.0)
    _micro_bump(plastic.node_tree, bsdf, 900.0, 0.12)

    anodized = new_material("OpenArm Anodized Aluminium")
    bsdf = principled(anodized)
    set_input(bsdf, "Base Color", (0.040, 0.040, 0.044, 1.0))
    set_input(bsdf, "Metallic", 0.85)
    set_input(bsdf, "Coat Weight", 0.3)
    set_input(bsdf, "Coat Roughness", 0.12)
    _noise_roughness(anodized.node_tree, bsdf, 0.30, 0.06, 140.0)
    _micro_bump(anodized.node_tree, bsdf, 700.0, 0.18)

    raw = new_material("OpenArm Machined Aluminium")
    bsdf = principled(raw)
    set_input(bsdf, "Base Color", (0.80, 0.81, 0.83, 1.0))
    set_input(bsdf, "Metallic", 1.0)
    _noise_roughness(raw.node_tree, bsdf, 0.30, 0.08, 120.0)
    _micro_bump(raw.node_tree, bsdf, 600.0, 0.22)

    steel = new_material("OpenArm Black Steel")
    bsdf = principled(steel)
    set_input(bsdf, "Base Color", (0.06, 0.06, 0.065, 1.0))
    set_input(bsdf, "Metallic", 1.0)
    _noise_roughness(steel.node_tree, bsdf, 0.40, 0.06, 100.0)
    return {
        "M_OpenArm_Plastic_Black": plastic,
        "M_OpenArm_Aluminum_Black": anodized,
        "M_OpenArm_Aluminum_Raw": raw,
        "M_OpenArm_Steel_Black": steel,
    }


_ROBOT_MATERIALS: dict[str, bpy.types.Material] = {}


def finish_robot_mesh(obj: bpy.types.Object) -> None:
    """Smooth curved surfaces, keep machined edges sharp, bevel them to catch highlights."""
    if obj.type != "MESH":
        return
    if not _ROBOT_MATERIALS:
        _ROBOT_MATERIALS.update(make_robot_materials())
    for slot in obj.material_slots:
        if slot.material is None:
            continue
        for prefix, replacement in _ROBOT_MATERIALS.items():
            if slot.material.name.startswith(prefix):
                slot.material = replacement
                break
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    split = obj.modifiers.new("Sharp edges", type="EDGE_SPLIT")
    split.split_angle = math.radians(32.0)
    split.use_edge_angle = True
    split.use_edge_sharp = True
    bevel = obj.modifiers.new("Edge bevel", type="BEVEL")
    bevel.width = 0.0006
    bevel.segments = 2
    bevel.limit_method = "ANGLE"
    bevel.angle_limit = math.radians(32.0)
    bevel.harden_normals = False


def import_glb(path: str, name: str) -> bpy.types.Object:
    """Import a GLB and return an empty that parents everything it created."""
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    created = [o for o in bpy.data.objects if o not in before]
    root = bpy.data.objects.new(f"{name}/root", None)
    bpy.context.scene.collection.objects.link(root)
    for obj in created:
        finish_robot_mesh(obj)
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
    light_data.energy = 200.0
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
    # Rim light behind the robot so the dark actuator housings separate from the room.
    rim_data = bpy.data.lights.new("Rim", type="AREA")
    rim_data.energy = 140.0
    rim_data.size = 0.8
    rim_data.color = (0.92, 0.95, 1.0)
    rim = bpy.data.objects.new("Rim", rim_data)
    bpy.context.scene.collection.objects.link(rim)
    rim.location = (-1.3, 0.6, 1.5)
    direction = Vector((-0.3, 0.0, 0.6)) - rim.location
    rim.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def new_camera(lens: float) -> bpy.types.Object:
    camera_data = bpy.data.cameras.new("Camera")
    camera_data.lens = lens
    camera_data.sensor_width = 36.0
    # Wrist cameras routinely observe the gripper and manipulated object from
    # only a few centimetres away. Blender's 10 cm default clips straight
    # through them, so use a near plane suitable for robot-mounted cameras.
    camera_data.clip_start = 0.005
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.scene.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    return camera


def setup_camera(
    look_from: list[float],
    look_at: list[float],
    lens: float,
    up_world: list[float] | None = None,
) -> bpy.types.Object:
    camera = new_camera(lens)
    eye = Vector(look_from)
    camera.location = eye
    direction = Vector(look_at) - eye
    if up_world is None:
        camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    else:
        forward = direction.normalized()
        right = forward.cross(Vector(up_world)).normalized()
        up = right.cross(forward).normalized()
        rotation = Matrix((right, up, -forward)).transposed().to_4x4()
        rotation.translation = eye
        camera.matrix_world = rotation
    return camera


def calibrated_lens(spec: dict) -> float:
    """Convert pixel focal length to Blender's 36 mm horizontal sensor model."""
    intrinsics = spec.get("intrinsics") or {}
    width = float(intrinsics.get("width_px", 0.0))
    fx = float(intrinsics.get("fx_px", 0.0))
    return 36.0 * fx / width if width > 0.0 and fx > 0.0 else 40.0


class EpisodeCamera:
    """A fixed look-at camera or a calibrated camera attached to a recorded actor."""

    def __init__(
        self,
        episode: "EpisodeScene",
        name: str,
        spec: dict,
        lens_override: float | None,
    ) -> None:
        lens = lens_override if lens_override is not None else calibrated_lens(spec)
        self.episode = episode
        self.actor_index: int | None = None
        if "look_from" in spec and "look_at" in spec:
            up_world = spec.get("up_world")
            if up_world is None and name == "top":
                up_world = [1.0, 0.0, 0.0]
            self.camera = setup_camera(
                spec["look_from"], spec["look_at"], lens, up_world
            )
            return

        self.camera = new_camera(lens)
        kind = spec.get("kind")
        if kind == "fixed":
            pose = spec.get("world_from_camera_cv")
            if not isinstance(pose, dict):
                raise RuntimeError(f"Fixed camera {name!r} has no world pose")
            right = Vector(pose["right_world"])
            up = Vector(pose["up_world"])
            forward = Vector(pose["forward_world"])
            position = Vector(pose["position_m"])
            self.camera.matrix_world = Matrix(
                (
                    (right.x, up.x, -forward.x, position.x),
                    (right.y, up.y, -forward.y, position.y),
                    (right.z, up.z, -forward.z, position.z),
                    (0.0, 0.0, 0.0, 1.0),
                )
            )
            return
        if kind != "actor":
            raise RuntimeError(f"Unsupported camera kind {kind!r} for {name!r}")

        suffix = str(spec.get("actor_suffix", ""))
        matches = [
            int(entry["index"])
            for entry in episode.spec["actors"]
            if entry["name"] == suffix or entry["name"].endswith(suffix)
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Camera {name!r} expected one recorded actor matching {suffix!r}; "
                f"found {len(matches)}"
            )
        self.actor_index = matches[0]
        self.apply_frame(0)

    def apply_frame(self, frame_index: int) -> None:
        if self.actor_index is None:
            return
        # Recorded camera actors use OpenCV axes (+X right, +Y down, +Z
        # forward). Blender cameras use +X right, +Y up, -Z forward.
        cv_to_blender = Matrix.Diagonal((1.0, -1.0, -1.0, 1.0))
        self.camera.matrix_world = (
            matrix_from_vector(self.episode.transforms[frame_index, self.actor_index])
            @ cv_to_blender
        )


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
        selection = configure_cycles_device(
            preferences,
            scene,
            requested=args.device,
            device_index=args.device_index,
        )
        scene.cycles.samples = args.samples
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.adaptive_threshold = 0.02
        scene.cycles.use_denoising = True
        try:
            scene.render.use_persistent_data = True
        except AttributeError:
            pass
        scene.cycles.max_bounces = 8
        scene.cycles.caustics_reflective = False
        scene.cycles.caustics_refractive = False
        print(
            "Cycles device:", scene.cycles.device,
            selection.backend,
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
    scene.view_settings.exposure = -0.45


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
                if args.hide_robot and "bots" in Path(glb["path"]).parts:
                    # Keep the actor root in ``self.objects`` so wrist cameras
                    # can still follow the recorded link pose. Hiding every
                    # descendant removes both the robot and its shadows.
                    stack = [obj]
                    while stack:
                        target = stack.pop()
                        target.hide_render = True
                        stack.extend(target.children)
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
    if args.camera not in episode.spec["cameras"]:
        available = ", ".join(sorted(episode.spec["cameras"]))
        raise RuntimeError(f"Unknown camera {args.camera!r}; available cameras: {available}")
    camera = EpisodeCamera(
        episode, args.camera, episode.spec["cameras"][args.camera], args.lens
    )
    add_floor()
    if args.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=str(Path(args.save_blend).resolve()))

    if args.worker_count < 1:
        raise ValueError("--worker-count must be positive")
    if not 0 <= args.worker_index < args.worker_count:
        raise ValueError("--worker-index must be within --worker-count")
    if args.frame_index is not None and args.worker_count != 1:
        raise ValueError("--frame-index cannot be combined with multiple workers")

    if args.frame_index is not None:
        indices = [args.frame_index]
        output_offset = 0
    else:
        all_indices = episode.frame_indices(args.fps, args.start, args.end)
        partition = partition_frames(len(all_indices), args.worker_index, args.worker_count)
        indices = all_indices[partition.start : partition.stop]
        output_offset = partition.start
    print(
        f"Rendering {len(indices)} frames with {args.engine} at {args.width}x{args.height} "
        f"(worker {args.worker_index + 1}/{args.worker_count})"
    )
    started = time.time()
    for output_index, frame_index in enumerate(indices):
        episode.apply_frame(frame_index)
        camera.apply_frame(frame_index)
        global_output_index = output_offset + output_index
        frame_path = output / f"frame_{global_output_index:05d}.png"
        if args.resume and not args.overwrite and frame_path.exists() and frame_path.stat().st_size > 0:
            print(f"  frame {global_output_index + 1}: already exists, skipping")
            continue
        bpy.context.scene.render.filepath = str(frame_path)
        bpy.ops.render.render(write_still=True)
        if output_index % 10 == 0 or output_index == len(indices) - 1:
            elapsed = time.time() - started
            print(
                f"  frame {global_output_index + 1} (recorded {frame_index}) "
                f"{elapsed:.0f}s elapsed, {elapsed / (output_index + 1):.1f}s/frame"
            )
    print(f"Frames written to {output}")


if __name__ == "__main__":
    main()
