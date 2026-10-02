"""Fit the Artec hand scan to a recorded Meta XR hand without remeshing it.

All geometry detail stays on the original surface. The source landmark rig is
authored in the scan's millimetre coordinates. Pose fitting uses blended bone
transforms; the resulting skin groups also drive the recorded animation.
"""

from pathlib import Path

import bpy
import bmesh
import numpy as np

SCAN = Path(__file__).resolve().parents[2] / "embodiments/assets/artec_hand/hand.ply"

# Joint centres, not points on the visible skin. Source +Z faces the palm.
LANDMARKS = {
    "bone_00_wrist_root": (2, -62, 17),
    "bone_02_thumb_metacarpal": (-15, -43, 19),
    "bone_03_thumb_proximal": (-33, -26, 24),
    "bone_04_thumb_medial": (-53, -9, 29),
    "bone_05_thumb_distal": (-76, 4, 33),
    "bone_06_index_proximal": (-26, 24, 15),
    "bone_07_index_medial": (-27, 59, 19),
    "bone_08_index_distal": (-26, 78, 21),
    "bone_09_middle_proximal": (1, 25, 17),
    "bone_10_middle_medial": (0, 65, 20),
    "bone_11_middle_distal": (1, 87, 22),
    "bone_12_ring_proximal": (24, 15, 19),
    "bone_13_ring_medial": (29, 49, 23),
    "bone_14_ring_distal": (32, 69, 25),
    "bone_15_pinky_metacarpal": (23, -35, 17),
    "bone_16_pinky_proximal": (40, 0, 23),
    "bone_17_pinky_medial": (52, 27, 27),
    "bone_18_pinky_distal": (56, 44, 29),
}
TIPS = {
    "bone_05_thumb_distal": (-94, 9, 34),
    "bone_08_index_distal": (-25, 96, 23),
    "bone_11_middle_distal": (1, 104, 23),
    "bone_14_ring_distal": (33, 83, 26),
    "bone_18_pinky_distal": (59, 56, 30),
}


def _frame(head, tail, normal):
    direction = tail - head
    length = np.linalg.norm(direction)
    if length < 1e-6:
        raise ValueError("Hand landmark segments must have nonzero length")
    y = direction / length
    x = np.cross(y, normal)
    if np.linalg.norm(x) < .01:
        x = np.cross(y, np.eye(3)[np.argmin(np.abs(y))])
    x /= np.linalg.norm(x)
    return np.column_stack((x, y, np.cross(x, y))), length


def fitted_scan(human):
    """Return a detailed mesh in bind-frame world coordinates with skin weights."""
    if not SCAN.exists():
        raise FileNotFoundError(f"Missing attributed Artec hand scan: {SCAN}")
    from .human_body import HAND_CHAINS, WRIST_ROOT

    bpy.ops.wm.ply_import(filepath=str(SCAN))
    obj = bpy.context.object
    obj.name = "scanned_hand"
    if human._index("human_forearm") is not None:
        # The scan includes a flat scanning cut below the anatomical wrist.
        # Remove it so it cannot protrude as a flange through the attached arm.
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bmesh.ops.bisect_plane(
            bm, geom=[*bm.verts, *bm.edges, *bm.faces], dist=.001,
            plane_co=(0, -62, 0), plane_no=(0, 1, 0), clear_inner=True,
        )
        bm.to_mesh(obj.data)
        bm.free()
    source = np.empty(len(obj.data.vertices) * 3, dtype=np.float64)
    obj.data.vertices.foreach_get("co", source)
    source = source.reshape(-1, 3) * .001
    specs = {spec.name: spec for spec in human.bones if spec.name in LANDMARKS}
    targets = {WRIST_ROOT: "bone_09_middle_proximal"}
    for chain in HAND_CHAINS:
        targets.update(zip(chain[:-1], chain[1:]))
    normal = np.array(human._world(WRIST_ROOT, human.bind_frame).to_3x3())[:, 2]
    wrist = np.array(LANDMARKS[WRIST_ROOT]) * .001
    middle = np.array(LANDMARKS["bone_09_middle_proximal"]) * .001
    scale = (specs[WRIST_ROOT].tail - specs[WRIST_ROOT].head).length / np.linalg.norm(middle - wrist)
    if human._index("human_forearm") is not None:
        # Blend the scan's open wrist ring into the attached arm cross-section.
        angle = np.arctan2(source[:, 2] - wrist[2], source[:, 0] - wrist[0])
        fade = np.clip((source[:, 1] + .062) / .025, 0, 1)
        fade = fade * fade * (3 - 2 * fade)
        for column, radius, trig in ((0, .025, np.cos), (2, .019, np.sin)):
            ring = wrist[column] + radius / scale * trig(angle)
            source[:, column] = ring * (1-fade) + source[:, column] * fade
    weights = []
    names = list(specs)
    for name in names:
        head = np.array(LANDMARKS[name]) * .001
        tail = np.array(TIPS[name] if name in TIPS else LANDMARKS[targets[name]]) * .001
        source_length = np.linalg.norm(tail-head)
        axis = tail - head
        fraction = np.clip((source - head) @ axis / (axis @ axis), 0, 1)
        nearest = head + fraction[:, None] * axis
        distance2 = np.sum((source - nearest) ** 2, axis=1)
        weight = 1 / (distance2 + .004 ** 2) ** 3
        if name != WRIST_ROOT:
            projection = (source - head) @ (axis / source_length)
            gate = np.clip((projection + .012) / .024, 0, 1)
            weight *= gate * gate * (3 - 2 * gate)
            wrist_gate = np.clip((source[:, 1] + .052) / .020, 0, 1)
            weight *= wrist_gate * wrist_gate * (3 - 2 * wrist_gate)
        weights.append(weight)
    weights = np.stack(weights, axis=1)
    weights /= weights.sum(axis=1, keepdims=True)
    weights[weights < 1e-5] = 0
    weights /= weights.sum(axis=1, keepdims=True)
    # Fit one smooth deformation field. Blending separate rigid fits here
    # creates folds at the thumb web even before any animation is applied.
    controls = [np.array(LANDMARKS[name])*.001 for name in names]
    destinations = [np.array(specs[name].head) for name in names]
    for name, tip in TIPS.items():
        controls.append(np.array(tip)*.001)
        destinations.append(np.array(specs[name].tail))
    controls = np.asarray(controls)
    source_frame, source_length = _frame(wrist,middle,np.array([0,0,1]))
    target_head, target_tail = np.array(specs[WRIST_ROOT].head), np.array(specs[WRIST_ROOT].tail)
    target_frame, target_length = _frame(target_head,target_tail,normal)
    rotation = target_frame @ np.diag([scale,target_length/source_length,scale]) @ source_frame.T
    offset = target_head - rotation @ wrist
    destinations = np.asarray(destinations) - (controls @ rotation.T + offset)
    # Fit displacement over the palm plane; preserve scan depth along its normal.
    controls = controls[:,:2]*10
    count = len(controls)
    kernel = np.linalg.norm(controls[:,None,:]-controls[None,:,:],axis=2)**3
    affine = np.column_stack((np.ones(count),controls))
    system = np.block([[kernel + np.eye(count)*.0001, affine],
                       [affine.T, np.zeros((3,3))]])
    coefficients = np.linalg.solve(system,np.vstack((destinations,np.zeros((3,3)))))
    result = source @ rotation.T + offset
    for start in range(0,len(source),16384):
        points = source[start:start+16384,:2]*10
        radial = np.linalg.norm(points[:,None,:]-controls[None,:,:],axis=2)**3
        basis = np.column_stack((radial,np.ones(len(points)),points))
        result[start:start+len(points)] += basis @ coefficients
    if human._index("human_forearm") is not None:
        ring = (target_head + np.cos(angle)[:,None]*.025*target_frame[:,0]
                + np.sin(angle)[:,None]*.019*target_frame[:,2]
                + (source[:,1]-wrist[1])[:,None]*(target_length/source_length)*target_frame[:,1])
        result = result*fade[:,None] + ring*(1-fade[:,None])
    obj.data.vertices.foreach_set("co", result.ravel())
    obj.data.update()
    for i, name in enumerate(names):
        group = obj.vertex_groups.new(name=name)
        for vertex in np.flatnonzero(weights[:, i] > 0):
            group.add([int(vertex)], float(weights[vertex, i]), "REPLACE")
    obj["attribution"] = "Hand by Artec 3D — https://www.artec3d.com/3d-models/hand — CC BY 3.0; adapted"
    return obj
