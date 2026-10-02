"""Textured, stationary standing person, sharing placement with replay previews."""
import bpy
import numpy as np
from mathutils import Matrix
from superdex_scenarios.rendering.human_placement import ASSETS, ROTATION, placement


def build_figure(human):
    wrist = np.array(human._world('bone_00_wrist_root', human.bind_frame).translation)
    index = human._index('human_upper_arm')
    shoulder = np.array(human._world('human_upper_arm', human.bind_frame).translation) if index is not None else None
    origin, shoulder = placement(wrist, shoulder)
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(ASSETS / 'standing.glb'))
    imported = set(bpy.data.objects) - before
    root = bpy.data.objects.new('Standing human — stationary', None)
    bpy.context.scene.collection.objects.link(root)
    matrix = np.eye(4)
    matrix[:3, :3] = ROTATION
    matrix[:3, 3] = origin
    root.matrix_world = Matrix(matrix)
    for obj in imported:
        if obj.parent not in imported:
            obj.parent = root
        if obj.type == 'MESH':
            for face in obj.data.polygons:
                face.use_smooth = True
    root['attribution'] = 'Full body scan by Artec 3D; CC BY 3.0; adapted — https://www.artec3d.com/3d-models/full-body-scan'
    root['motion'] = 'Fixed placement from recording frame 0; visual only'
    human.figure = root
    if index is None:
        human.visual_shoulder = shoulder
