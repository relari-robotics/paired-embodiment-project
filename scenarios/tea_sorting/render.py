"""Cycles replay with photographed wood and detailed sealed-paper packet meshes.

Run inside Blender: blender -b -P scenarios/tea_sorting/render.py -- --recording DIR
--output DIR --camera detail --frame-index 0 --samples 256 --width 2560 --height 1920
All task object poses come from the physics recording, including the box/cradles.
"""

import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from superdex_scenarios.rendering.blender import render_episode as base  # noqa: E402
from scenarios.tea_sorting.geometry import (  # noqa: E402
    BOX_SIZE,
    HEIGHT,
    BIN_Y,
    TEAS,
    COLORS,
    box_components,
    holder_components,
    packet_surface,
)

ASSETS = Path(__file__).with_name("assets")


def image_node(material, filename, *, data=False):
    node = material.node_tree.nodes.new("ShaderNodeTexImage")
    node.image = bpy.data.images.load(str(ASSETS / filename), check_existing=True)
    if data:
        node.image.colorspace_settings.name = "Non-Color"
    node.interpolation = "Linear"
    return node


def wood_material():
    material = base.new_material("Photographed varnished wood — Poly Haven 8K")
    tree, bsdf = material.node_tree, base.principled(material)
    diffuse = image_node(material, "wood_table_001_diff_8k.jpg")
    rough = image_node(material, "wood_table_001_rough_8k.jpg", data=True)
    normal = image_node(material, "wood_table_001_nor_gl_8k.jpg", data=True)
    normal_map = tree.nodes.new("ShaderNodeNormalMap")
    normal_map.inputs["Strength"].default_value = 0.35
    finish = tree.nodes.new("ShaderNodeHueSaturation")
    finish.inputs["Saturation"].default_value = 0.65
    tree.links.new(diffuse.outputs["Color"], finish.inputs["Color"])
    tree.links.new(finish.outputs["Color"], bsdf.inputs["Base Color"])
    matte = tree.nodes.new("ShaderNodeMapRange")
    matte.inputs["To Min"].default_value = 0.36
    matte.inputs["To Max"].default_value = 0.65
    tree.links.new(rough.outputs["Color"], matte.inputs["Value"])
    tree.links.new(matte.outputs["Result"], bsdf.inputs["Roughness"])
    tree.links.new(normal.outputs["Color"], normal_map.inputs["Color"])
    tree.links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])
    base.set_input(bsdf, "Coat Weight", 0.12)
    base.set_input(bsdf, "Coat Roughness", 0.28)
    return material


def paper_material():
    material = base.new_material(
        "Printed matte paper laminate — red / yellow / blue atlas"
    )
    tree, bsdf = material.node_tree, base.principled(material)
    print_map = image_node(material, "tea_wrappers.png")
    tree.links.new(print_map.outputs["Color"], bsdf.inputs["Base Color"])
    coordinates = tree.nodes.new("ShaderNodeTexCoord")
    fibre = tree.nodes.new("ShaderNodeTexNoise")
    fibre.inputs["Scale"].default_value = 6200
    fibre.inputs["Detail"].default_value = 3
    tree.links.new(coordinates.outputs["Object"], fibre.inputs["Vector"])
    rough = tree.nodes.new("ShaderNodeMapRange")
    rough.inputs["To Min"].default_value = 0.42
    rough.inputs["To Max"].default_value = 0.64
    tree.links.new(fibre.outputs["Fac"], rough.inputs["Value"])
    tree.links.new(rough.outputs["Result"], bsdf.inputs["Roughness"])
    bump = tree.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.28
    bump.inputs["Distance"].default_value = 0.000035
    tree.links.new(fibre.outputs["Fac"], bump.inputs["Height"])
    tree.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    base.set_input(bsdf, "Coat Weight", 0.08)
    base.set_input(bsdf, "Coat Roughness", 0.36)
    return material


def simple_material(name, color, roughness, metallic=0):
    material = base.new_material(name)
    bsdf = base.principled(material)
    base.set_input(bsdf, "Base Color", (*color, 1))
    base.set_input(bsdf, "Roughness", roughness)
    base.set_input(bsdf, "Metallic", metallic)
    return material


def wood_uv(obj, offset=(0.0, 0.0)):
    """Planar board mapping in metres: the scanned wood covers 1.5 m."""
    uv = obj.data.uv_layers.new(name="Physical scale wood grain")
    for poly in obj.data.polygons:
        axis = int(np.argmax(np.abs(poly.normal)))
        axes = ((1, 2), (0, 2), (0, 1))[axis]
        for loop_index in poly.loop_indices:
            vertex = obj.data.vertices[obj.data.loops[loop_index].vertex_index].co
            uv.data[loop_index].uv = (
                vertex[axes[0]] / 1.5 + offset[0],
                vertex[axes[1]] / 1.5 + offset[1],
            )


def board(name, centre, size, material, parent, *, bevel=0.0007, uv_offset=(0, 0)):
    bpy.ops.mesh.primitive_cube_add(size=1)
    obj = bpy.context.object
    obj.name = name
    # Bake local geometry, keeping actual actor coordinates on the parent.
    for v in obj.data.vertices:
        v.co = Vector([v.co[j] * size[j] + centre[j] for j in range(3)])
    obj.data.update()
    obj.parent = parent
    obj.data.materials.append(material)
    wood_uv(obj, uv_offset)
    edge = obj.modifiers.new("Submillimetre manufactured edge radius", "BEVEL")
    edge.width = bevel
    edge.segments = 3
    obj.modifiers.new("Weighted corner normals", "WEIGHTED_NORMAL")
    return obj


def packet_mesh(name, tea, material):
    """Pillow, compressed perimeter seals, crimp lines, and shallow real creases.

    Render envelope stays inside the 62 x 10 x 75 mm contact proxy. Texture
    coordinates select one third of the source atlas without resampling artwork.
    """
    vertices, faces, uvs = packet_surface(96, 112, tea, detail=True)
    uvs[:, 0] = (tea + 0.004 + uvs[:, 0] * 0.992) / 3
    uvs[:, 1] = 0.004 + uvs[:, 1] * 0.992
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    uv_layer = mesh.uv_layers.new(name="Wrapper print")
    for loop in mesh.loops:
        uv_layer.data[loop.index].uv = uvs[loop.vertex_index]
    for poly in mesh.polygons:
        poly.use_smooth = True
    obj.data.materials.append(material)
    return obj


def label(parent, tea, enamel, brass, ink):
    # Inlaid top-facing strip on the front rim: physical width 8 mm.
    x, y, z = BOX_SIZE[0] / 2 - 0.004, BIN_Y[tea], BOX_SIZE[2] + HEIGHT + 0.0002
    board(
        f"{TEAS[tea]} enamel label",
        (x, y, z),
        (0.006, 0.072, 0.00035),
        enamel,
        parent,
        bevel=0.0002,
    )
    bpy.ops.object.text_add()
    obj = bpy.context.object
    obj.name = f"{TEAS[tea]} engraved label type"
    obj.parent = parent
    obj.location = (x, y, z + 0.00021)
    obj.rotation_euler[2] = math.pi / 2
    obj.data.body = TEAS[tea].replace("_", " ").upper()
    obj.data.align_x = "CENTER"
    obj.data.align_y = "CENTER"
    obj.data.size = 0.0036
    obj.data.extrude = 0.00001
    obj.data.materials.append(ink)
    for sign in (-1, 1):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=12, ring_count=6, radius=0.0007)
        rivet = bpy.context.object
        rivet.name = "Inset brass pin"
        rivet.parent = parent
        rivet.location = (x, y + sign * 0.033, z + 0.00012)
        rivet.scale.z = 0.35
        rivet.data.materials.append(brass)


class TeaEpisodeScene(base.EpisodeScene):
    def __init__(self, recording, args):
        super().__init__(recording, args)
        if self.spec.get("metadata", {}).get("task") != "tea_sorting":
            raise ValueError("This renderer requires a tea_sorting recording")
        wood, paper = wood_material(), paper_material()
        cradle = simple_material(
            "Satin powder-coated presentation cradles", (0.045, 0.052, 0.048), 0.47
        )
        brass = simple_material(
            "Brushed brass inlay pins", (0.48, 0.30, 0.085), 0.28, 0.75
        )
        ink = simple_material("Warm ivory label ink", (0.86, 0.82, 0.69), 0.55)
        enamels = [
            simple_material(f"{tea} enamel", color, 0.3)
            for tea, color in zip(TEAS, COLORS)
        ]
        for entry in self.spec["actors"]:
            index, name = entry["index"], entry["name"]
            obj = self.objects.get(index)
            if obj is None:
                continue
            if name.startswith("wooden_desk/"):
                obj.data.materials.clear()
                obj.data.materials.append(wood)
                wood_uv(obj)
                mod = obj.modifiers.new("Softly worn desk edges", "BEVEL")
                mod.width, mod.segments = 0.0018, 4
                obj.modifiers.new("Desk weighted normals", "WEIGHTED_NORMAL")
            elif name.startswith("tea/"):
                bpy.data.objects.remove(obj, do_unlink=True)
                if "packet_" in name:
                    tea = next(
                        t
                        for t, label_name in enumerate(TEAS)
                        if name.endswith(label_name)
                    )
                    root = packet_mesh(name, tea, paper)
                else:
                    root = bpy.data.objects.new(name, None)
                    bpy.context.collection.objects.link(root)
                    components = (
                        box_components() if name == "tea/box" else holder_components()
                    )
                    for i, (centre, size) in enumerate(components):
                        board(
                            f"{name}/board_{i}",
                            centre,
                            size,
                            wood if name == "tea/box" else cradle,
                            root,
                            uv_offset=(0.24 + i * 0.073, 0.37 + i * 0.119),
                        )
                    if name == "tea/box":
                        for tea in range(3):
                            label(root, tea, enamels[tea], brass, ink)
                self.objects[index] = root
                self.local[index] = Matrix.Identity(4)


def main():
    args = base.parse_args()
    recording, output = Path(args.recording).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    base.clear_scene()
    base.setup_render(args)
    base.setup_world(args.environment, args.environment_strength * 0.45)
    # Large daylight source; restrained exposure retains ink and wood detail.
    bpy.data.lights["Key"].energy = 100
    bpy.data.lights["Fill"].energy = 30
    bpy.data.lights["Rim"].energy = 55
    bpy.context.scene.view_settings.exposure = -0.7
    bpy.context.scene.render.image_settings.color_depth = "16"
    if args.engine == "CYCLES":
        bpy.context.scene.cycles.adaptive_threshold = 0.008
        bpy.context.scene.cycles.max_bounces = 12
    episode = TeaEpisodeScene(recording, args)
    if args.camera not in episode.spec["cameras"]:
        raise ValueError(f"Unknown camera: {args.camera}")
    camera = base.EpisodeCamera(
        episode, args.camera, episode.spec["cameras"][args.camera], args.lens
    )
    base.add_floor()
    indices = (
        [args.frame_index]
        if args.frame_index is not None
        else episode.frame_indices(args.fps, args.start, args.end)
    )
    for output_index, frame_index in enumerate(indices):
        episode.apply_frame(frame_index)
        camera.apply_frame(frame_index)
        if args.save_blend and output_index == 0:
            bpy.ops.file.pack_all()
            bpy.ops.wm.save_as_mainfile(filepath=str(Path(args.save_blend).resolve()))
        bpy.context.scene.render.filepath = str(
            output / f"frame_{output_index:05d}.png"
        )
        bpy.ops.render.render(write_still=True)
        print(
            f"Tea replay frame {output_index + 1}/{len(indices)}: recorded frame {frame_index}",
            flush=True,
        )


if __name__ == "__main__":
    main()
