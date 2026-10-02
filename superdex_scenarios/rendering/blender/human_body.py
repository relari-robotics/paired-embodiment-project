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

"""Scanned hand, skinned arm and stationary textured person for Blender.

Recorded hand links drive the scan. Hand-only recordings use visual two-bone
IK from a fixed shoulder; the figure adds no physics or recorded body motion.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from dataclasses import dataclass

import bpy  # type: ignore[import-not-found]
import numpy as np
from mathutils import Matrix, Vector  # type: ignore[import-not-found]

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from superdex_scenarios.rendering.arm_surface import HUMAN_APPEARANCE, cross_section

UPPER_ARM = "human_upper_arm"
FOREARM = "human_forearm"
WRIST_ROOT = "bone_00_wrist_root"

# Finger chains in the Meta XR hand asset (proximal to distal).
HAND_CHAINS = (
    ("bone_02_thumb_metacarpal", "bone_03_thumb_proximal", "bone_04_thumb_medial", "bone_05_thumb_distal"),
    ("bone_06_index_proximal", "bone_07_index_medial", "bone_08_index_distal"),
    ("bone_09_middle_proximal", "bone_10_middle_medial", "bone_11_middle_distal"),
    ("bone_12_ring_proximal", "bone_13_ring_medial", "bone_14_ring_distal"),
    ("bone_15_pinky_metacarpal", "bone_16_pinky_proximal", "bone_17_pinky_medial", "bone_18_pinky_distal"),
)



def matrix_from_row(row: np.ndarray) -> Matrix:
    from mathutils import Quaternion

    tx, ty, tz, qx, qy, qz, qw = (float(v) for v in row)
    matrix = Quaternion((qw, qx, qy, qz)).to_matrix().to_4x4()
    matrix.translation = Vector((tx, ty, tz))
    return matrix


@dataclass
class _BoneSpec:
    name: str
    actor_index: int
    head: Vector
    tail: Vector


class HumanBody:
    """Build once from an :class:`EpisodeScene`, then :meth:`pose` per frame."""

    def __init__(self, episode, bind_frame: int = 0) -> None:
        self.episode = episode
        self.bind_frame = bind_frame
        self.actor_index = {entry["name"]: entry["index"] for entry in episode.spec["actors"]}
        self.prefix = self._detect_prefix()
        self.visual_shoulder = None
        self.figure = None
        self.bones: list[_BoneSpec] = []
        self.rest_world: dict[str, Matrix] = {}
        self.armature: bpy.types.Object | None = None
        self.body: bpy.types.Object | None = None

    # -- discovery -----------------------------------------------------------

    def _detect_prefix(self) -> str | None:
        for name in self.actor_index:
            if name.endswith("/" + UPPER_ARM):
                return name[: -len(UPPER_ARM)]
        for name in self.actor_index:
            if name.endswith("/" + WRIST_ROOT):
                return name[: -len(WRIST_ROOT)]
        return None

    @property
    def available(self) -> bool:
        return self.prefix is not None and all(
            self._index(link) is not None
            for link in (WRIST_ROOT, *(l for chain in HAND_CHAINS for l in chain))
        )

    def _index(self, link: str) -> int | None:
        if self.visual_shoulder is not None and link in (UPPER_ARM, FOREARM):
            return -1
        return self.actor_index.get(f"{self.prefix}{link}")

    def _world(self, link: str, frame: int) -> Matrix:
        if self.visual_shoulder is not None and link in (UPPER_ARM, FOREARM):
            from superdex_scenarios.rendering.human_placement import arm_joints, segment_matrix
            wrist_world = self._world(WRIST_ROOT, frame)
            wrist = np.array(wrist_world.translation)
            elbow = arm_joints(self.visual_shoulder, wrist)
            head, tail = (self.visual_shoulder, elbow) if link == UPPER_ARM else (elbow, wrist)
            result = segment_matrix(head, tail, np.array(wrist_world.to_3x3())[:, 0])
            if link in self.rest_world:
                spec = next(b for b in self.bones if b.name == link)
                result[:3, 0] *= np.linalg.norm(tail-head) / (spec.tail-spec.head).length
            return Matrix(result)
        index = self._index(link)
        assert index is not None, link
        return matrix_from_row(self.episode.transforms[frame][index])

    # -- geometry --------------------------------------------------------------

    def _bone_specs(self) -> list[_BoneSpec]:
        frame = self.bind_frame
        wrist = self._world(WRIST_ROOT, frame).translation
        specs = []
        if self._index(UPPER_ARM) is not None and self._index(FOREARM) is not None:
            shoulder = self._world(UPPER_ARM, frame).translation
            elbow = self._world(FOREARM, frame).translation
            specs.extend([
                _BoneSpec(UPPER_ARM, self._index(UPPER_ARM), shoulder, elbow),
                _BoneSpec(FOREARM, self._index(FOREARM), elbow, wrist),
            ])
        # Wrist root: from the wrist towards the middle finger base.
        middle = self._world("bone_09_middle_proximal", frame).translation
        specs.append(_BoneSpec(WRIST_ROOT, self._index(WRIST_ROOT), wrist, middle))
        for chain in HAND_CHAINS:
            heads = [self._world(link, frame).translation for link in chain]
            for position, link in enumerate(chain):
                head = heads[position]
                if position + 1 < len(chain):
                    tail = heads[position + 1]
                else:
                    direction = heads[position] - heads[position - 1]
                    tail = head + direction * 0.85
                specs.append(_BoneSpec(link, self._index(link), head, tail))
        return specs

    def _arm_tube(self) -> bpy.types.Object:
        """Continuous anatomical surface with oval joints and muscle relief."""
        frame = self.bind_frame
        shoulder = np.array(self._world(UPPER_ARM, frame).translation)
        elbow = np.array(self._world(FOREARM, frame).translation)
        wrist = np.array(self._world(WRIST_ROOT, frame).translation)
        upper_length = float(np.linalg.norm(elbow - shoulder))
        fore_length = float(np.linalg.norm(wrist - elbow))
        if min(upper_length, fore_length) < 0.001:
            raise ValueError("Human arm links must have nonzero lengths")
        upper_dir = (elbow - shoulder) / upper_length
        fore_dir = (wrist - elbow) / fore_length
        total = upper_length + fore_length
        segments = 64
        angles = np.arange(segments) * (2 * math.pi / segments)
        vertices, faces = [], []
        # A consistent lateral axis follows the recorded limb's orientation.
        reference = np.array(self._world(UPPER_ARM, frame).to_3x3())[:, 1]
        if abs(float(np.dot(reference, upper_dir))) > .9:
            reference = np.eye(3)[np.argmin(np.abs(upper_dir))]
        cap = HUMAN_APPEARANCE["shoulder_cap_length_m"]
        samples = list(np.linspace(-cap + .001, 0, 16, endpoint=False))
        samples += list(np.linspace(0, total, max(2, int(total / .003))))
        for index, distance in enumerate(samples):
            if distance <= upper_length:
                centre = shoulder + upper_dir * distance
                section = cross_section("upper_arm", max(0, distance) / upper_length, angles)
                if distance < 0:
                    section *= math.sqrt(max(0, 1 - (distance / cap) ** 2))
            else:
                centre = elbow + fore_dir * (distance - upper_length)
                section = cross_section("forearm", (distance - upper_length) / fore_length, angles)
            # Round the centreline over the elbow instead of adding a joint sphere.
            width = min(.028, upper_length / 4, fore_length / 4)
            blend = float(np.clip((distance - upper_length + width) / (2 * width), 0, 1))
            if 0 < blend < 1:
                start, end = elbow - upper_dir * width, elbow + fore_dir * width
                centre = (1 - blend) ** 2 * start + 2 * blend * (1 - blend) * elbow + blend ** 2 * end
            tangent = (1 - blend) * upper_dir + blend * fore_dir
            if distance > total - .07:
                wrist_forward = np.array(self._world("bone_09_middle_proximal", frame).translation) - wrist
                wrist_forward /= np.linalg.norm(wrist_forward)
                t = float(np.clip((distance - total + .07) / .07, 0, 1))
                a, b = wrist - fore_dir * .07, wrist - wrist_forward * .035
                centre = (1-t)**2*a + 2*t*(1-t)*b + t*t*wrist
                tangent = 2*(1-t)*(b-a) + 2*t*(wrist-b)
            tangent /= np.linalg.norm(tangent)
            if distance > upper_length:
                palm_normal = np.array(self._world(WRIST_ROOT, frame).to_3x3())[:, 2]
                wrist_forward = np.array(self._world("bone_09_middle_proximal", frame).translation) - wrist
                lateral = np.cross(wrist_forward, palm_normal)
                lateral /= np.linalg.norm(lateral)
                twist = (distance - upper_length) / fore_length
                reference_here = reference*(1-twist) + lateral*twist
            else:
                reference_here = reference
            u = reference_here - tangent * np.dot(reference_here, tangent)
            if np.linalg.norm(u) < .01:
                u = np.cross(tangent, np.eye(3)[np.argmin(np.abs(tangent))])
            u /= np.linalg.norm(u)
            v = np.cross(tangent, u)
            base = len(vertices)
            vertices.extend(tuple(centre + y * u + z * v) for y, z in section)
            if index:
                previous = base - segments
                for k in range(segments):
                    k2 = (k + 1) % segments
                    faces.append((previous + k, previous + k2, base + k2, base + k))
        first_centre = len(vertices)
        vertices.append(tuple(shoulder - upper_dir * cap))
        last_base = (len(samples) - 1) * segments
        last_centre = len(vertices)
        vertices.append(tuple(wrist))
        for k in range(segments):
            k2 = (k + 1) % segments
            faces.append((first_centre, k2, k))
            faces.append((last_centre, last_base + k, last_base + k2))
        mesh = bpy.data.meshes.new("human_arm_tube")
        mesh.from_pydata(vertices, [], faces)
        mesh.validate()
        mesh.update()
        tube = bpy.data.objects.new("human_arm_tube", mesh)
        bpy.context.scene.collection.objects.link(tube)
        groups = {name: tube.vertex_groups.new(name=name) for name in (UPPER_ARM, FOREARM, WRIST_ROOT)}
        for index, distance in enumerate(samples):
            elbow_blend = float(np.clip((distance - upper_length + .025) / .05, 0, 1))
            wrist_blend = float(np.clip((distance - total + .025) / .025, 0, 1))
            weights = (1 - elbow_blend, elbow_blend * (1 - wrist_blend), wrist_blend)
            ring = list(range(index * segments, (index + 1) * segments))
            for name, weight in zip(groups, weights):
                if weight > 0:
                    groups[name].add(ring, weight, "REPLACE")
        groups[UPPER_ARM].add([first_centre], 1, "REPLACE")
        groups[WRIST_ROOT].add([last_centre], 1, "REPLACE")
        return tube

    @staticmethod
    def _join(objects: list[bpy.types.Object], name: str) -> bpy.types.Object:
        with bpy.context.temp_override(
            object=objects[0], active_object=objects[0], selected_objects=objects,
            selected_editable_objects=objects,
        ):
            bpy.ops.object.join()
        objects[0].name = name
        return objects[0]

    def _hand_source(self) -> list[bpy.types.Object]:
        from superdex_scenarios.rendering.blender.hand_scan import fitted_scan

        return [fitted_scan(self)]

    def _hide_original(self) -> None:
        for link in (UPPER_ARM, FOREARM, "human_shoulder_anchor", WRIST_ROOT, "bone_01_wrist_stub", *(l for chain in HAND_CHAINS for l in chain)):
            index = self._index(link)
            root = self.episode.objects.get(index)
            if root is None:
                continue
            for obj in (root, *root.children_recursive):
                obj.hide_render = True
                obj.hide_viewport = True

    # -- build -----------------------------------------------------------------

    def build(self, material: bpy.types.Material, include_figure: bool = True) -> None:
        if include_figure:
            from superdex_scenarios.rendering.blender.human_figure import build_figure
            build_figure(self)
        self.bones = self._bone_specs()
        parts = self._hand_source()
        if self._index(UPPER_ARM) is not None and self._index(FOREARM) is not None:
            parts.insert(0, self._arm_tube())
        body = self._join(parts, "human_body") if len(parts) > 1 else parts[0]
        body.name = "human_body"
        body.data.materials.clear()
        body.data.materials.append(material)
        # Keep the scan's original surface: voxel remeshing and relaxation
        # destroy the creases and nails that make a close-up look human.
        for polygon in body.data.polygons:
            polygon.use_smooth = True
        body["attribution"] = "Hand by Artec 3D — https://www.artec3d.com/3d-models/hand — CC BY 3.0; adapted"
        print(f"Scanned human skin: {len(body.data.vertices)} vertices, {len(body.data.polygons)} faces")

        armature_data = bpy.data.armatures.new("human_armature")
        armature = bpy.data.objects.new("human_armature", armature_data)
        bpy.context.scene.collection.objects.link(armature)
        view_layer = bpy.context.view_layer
        for obj in bpy.data.objects:
            obj.select_set(False)
        armature.select_set(True)
        view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode="EDIT")
        for spec in self.bones:
            bone = armature_data.edit_bones.new(spec.name)
            bone.head = spec.head
            bone.tail = spec.tail
            bone.use_deform = True
        bpy.ops.object.mode_set(mode="OBJECT")
        for spec in self.bones:
            self.rest_world[spec.name] = self._world(spec.name, self.bind_frame)

        import json
        armature["actor_bind_matrices"] = json.dumps({
            name: [list(row) for row in matrix] for name, matrix in self.rest_world.items()
        })
        body.parent = armature
        modifier = body.modifiers.new("Recorded hand motion", type="ARMATURE")
        modifier.object = armature
        modifier.use_deform_preserve_volume = True
        armature.select_set(False)
        self.armature = armature
        self.body = body
        self._hide_original()
        self.pose(self.bind_frame)

    # -- per frame ---------------------------------------------------------------

    def pose(self, frame: int) -> None:
        if self.armature is None:
            return
        pose = self.armature.pose
        for spec in self.bones:
            world = self._world(spec.name, frame)
            rest_local = self.armature.data.bones[spec.name].matrix_local
            pose.bones[spec.name].matrix = (
                world @ self.rest_world[spec.name].inverted() @ rest_local
            )
        bpy.context.view_layer.update()


def make_skin_material() -> bpy.types.Material:
    material = bpy.data.materials.new("Human Skin")
    material.use_nodes = True
    tree = material.node_tree
    bsdf = tree.nodes["Principled BSDF"]
    import os

    if os.environ.get("SUPERDEX_FLAT_SKIN"):
        bsdf.inputs["Base Color"].default_value = (0.6, 0.6, 0.6, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.5
        return material

    def set_input(name: str, value) -> None:
        if name in bsdf.inputs:
            bsdf.inputs[name].default_value = value

    skin = HUMAN_APPEARANCE["skin"]
    set_input("Base Color", (*skin["base_color_linear"], 1.0))
    set_input("Roughness", skin["roughness"])
    set_input("Subsurface Weight", skin["subsurface_weight"])
    set_input("Subsurface Radius", (1.0, 0.25, 0.12))
    set_input("Subsurface Scale", skin["subsurface_scale_m"])
    set_input("Specular IOR Level", 0.35)
    set_input("Coat Weight", 0.04)
    set_input("Coat Roughness", 0.4)
    # Blotchy colour variation and pore-scale roughness/bump.
    coords = tree.nodes.new("ShaderNodeTexCoord")
    blotch = tree.nodes.new("ShaderNodeTexNoise")
    blotch.inputs["Scale"].default_value = 90.0
    blotch.inputs["Detail"].default_value = 6.0
    ramp = tree.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.3
    ramp.color_ramp.elements[0].color = (*skin["pigment_dark_linear"], 1.0)
    ramp.color_ramp.elements[1].position = 0.75
    ramp.color_ramp.elements[1].color = (*skin["pigment_light_linear"], 1.0)
    pores = tree.nodes.new("ShaderNodeTexNoise")
    pores.inputs["Scale"].default_value = 900.0
    pores.inputs["Detail"].default_value = 6.0
    rough = tree.nodes.new("ShaderNodeMapRange")
    rough.inputs["From Min"].default_value = 0.3
    rough.inputs["From Max"].default_value = 0.7
    rough.inputs["To Min"].default_value = skin["roughness_range"][0]
    rough.inputs["To Max"].default_value = skin["roughness_range"][1]
    bump = tree.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.12
    bump.inputs["Distance"].default_value = 0.00012
    links = tree.links
    links.new(coords.outputs["Object"], blotch.inputs["Vector"])
    links.new(coords.outputs["Object"], pores.inputs["Vector"])
    links.new(blotch.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(pores.outputs["Fac"], rough.inputs["Value"])
    links.new(rough.outputs["Result"], bsdf.inputs["Roughness"])
    links.new(pores.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return material


__all__ = ["HumanBody", "make_skin_material"]
