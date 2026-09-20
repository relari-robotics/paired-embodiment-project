"""Blender tool: export a lightweight preview rig from a rendered review scene.

blender -b scene.blend --python tools/build_human_preview.py
The source scene must contain human_body and human_armature from HumanBody.
"""
import json
from pathlib import Path
import bpy

out = Path(__file__).resolve().parents[1] / 'superdex_scenarios/embodiments/assets/artec_hand'
body = bpy.data.objects['human_body']
armature = bpy.data.objects['human_armature']
rest = json.loads(armature['actor_bind_matrices'])
# Mesh simplification preserves the silhouette and rig for interactive use.
for obj in bpy.data.objects:
    obj.select_set(False)
body.hide_set(False)
body.select_set(True)
bpy.context.view_layer.objects.active = body
modifier = body.modifiers.new('Interactive resolution', 'DECIMATE')
modifier.ratio = .085
bpy.ops.object.modifier_move_up(modifier=modifier.name)
bpy.ops.object.modifier_apply(modifier=modifier.name)
armature.select_set(True)
material = body.data.materials[0]
# glTF cannot carry Blender's procedural skin network. Use its shared base tone.
appearance = json.loads((Path(__file__).resolve().parents[1] / 'superdex_scenarios/rendering/human_appearance.json').read_text())['skin']
material.node_tree.nodes.clear()
bsdf=material.node_tree.nodes.new('ShaderNodeBsdfPrincipled')
bsdf.inputs['Base Color'].default_value=(*appearance['base_color_linear'],1)
bsdf.inputs['Roughness'].default_value=appearance['roughness']
output=material.node_tree.nodes.new('ShaderNodeOutputMaterial')
material.node_tree.links.new(bsdf.outputs['BSDF'],output.inputs['Surface'])
bpy.ops.export_scene.gltf(filepath=str(out/'preview.glb'), use_selection=True,
                         export_format='GLB', export_animations=False,
                         export_yup=True, export_skins=True, export_all_influences=True)
(out/'preview_bind.json').write_text(json.dumps(rest,indent=2)+'\n')
print('Exported interactive human preview', len(body.data.polygons), 'faces')
