"""Prepare stationary human assets from Artec's attributed full-body OBJ.

Usage: .venv/bin/python tools/build_human_figure_assets.py --source PATH/full_body.obj
Requires numpy, scipy, trimesh, and Pillow. Source and license links live alongside the
derived GLBs. The pose is baked once; torso and legs have no animation tracks.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def smoothstep(lo, hi, value):
    t = np.clip((value - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    destination = Path(__file__).resolve().parents[1] / "superdex_scenarios/embodiments/assets/artec_figure"
    destination.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.load(args.source, force="mesh")
    source = mesh.vertices.copy()
    scale = 1.75 / np.ptp(source[:, 1])
    # Authoring coordinates: +X person's right, +Y forward, +Z up.
    vertices = np.column_stack((-source[:, 0], source[:, 2], source[:, 1] - source[:, 1].min())) * scale
    centres = vertices[mesh.faces].mean(axis=1)
    # The scanned platform is tilted; clip against its measured surface plane.
    floor_normal = np.array([-.02968321, .14176858, .98945469])
    floor_offset = .0878190115
    ground = .088
    remove = centres @ floor_normal < floor_offset + .009
    # Separate exposed skin by the source photograph, preserving the jeans
    # beneath the scanned hand. Flatten the hand-on-hip imprint into clothing.
    from scipy.spatial import cKDTree
    pixels = np.asarray(mesh.visual.material.image)[:, :, :3]
    uv = mesh.visual.uv.copy()
    rgb = pixels[np.clip(((1-uv[:,1])*pixels.shape[0]).astype(int),0,pixels.shape[0]-1),
                 np.clip((uv[:,0]*pixels.shape[1]).astype(int),0,pixels.shape[1]-1)].astype(float)
    skin = (rgb[:,0] > rgb[:,1]*1.07) & (rgb[:,0] > rgb[:,2]*1.2)
    hand = skin & (vertices[:,0] > .155) & (vertices[:,2] > .94) & (vertices[:,2] < 1.10)
    garment = (~skin) & (vertices[:,0] > .1) & (vertices[:,2] > .9) & (vertices[:,2] < 1.12)
    candidates = np.flatnonzero(garment)
    nearest = cKDTree(vertices[candidates]).query(vertices[hand])[1]
    uv[hand] = uv[candidates[nearest]]
    # Project the hand into a smooth, fitted hip surface.
    angle = np.arctan2((vertices[hand,1]-.01)/.115, vertices[hand,0]/.225)
    vertices[hand,0] = .225*np.cos(angle)
    vertices[hand,1] = .01+.115*np.sin(angle)
    mesh.visual.uv = uv
    exposed_arm = skin & (vertices[:,0] > .24) & (vertices[:,2] >= 1.095) & (vertices[:,2] < 1.30)
    remove |= exposed_arm[mesh.faces].any(axis=1)
    # Cover the occluded garment surface that the original scan could not see.
    angles = np.linspace(-1.75, 1.75, 48)
    heights = np.linspace(.935, 1.135, 30)
    patch_vertices = np.array([((.23 + .015*(z-.935)/.2)*np.cos(a),
                                .025+.14*np.sin(a),z) for z in heights for a in angles])
    patch_faces = []
    for j in range(len(heights)-1):
        for i in range(len(angles)-1):
            a=j*len(angles)+i; b=a+len(angles)
            patch_faces.extend(((a,a+1,b+1),(a,b+1,b)))
    denim = np.flatnonzero((vertices[:,2] > .84) & (vertices[:,2] < .95)
                            & (rgb[:,2] > rgb[:,0]*1.12) & (rgb.mean(axis=1)>40))
    # Give the reconstructed surface a contiguous denim swatch instead of
    # interpolating between unrelated UV islands in the photographed atlas.
    from PIL import Image
    sample = denim[cKDTree(vertices[denim]).query([.19,.04,.90])[1]]
    px = int(uv[sample,0]*pixels.shape[1]); py = int((1-uv[sample,1])*pixels.shape[0])
    swatch = mesh.visual.material.image.crop((px-24,py-24,px+24,py+24)).resize((256,256))
    original_image = mesh.visual.material.image
    atlas = Image.new("RGB", (original_image.width, original_image.height+256))
    atlas.paste(original_image,(0,0));atlas.paste(swatch,(0,original_image.height))
    mesh.visual.uv[:,1] = 1-(1-mesh.visual.uv[:,1])*original_image.height/atlas.height
    mesh.visual.material.image = atlas
    patch_uv = np.array([(i/(len(angles)-1)*255/atlas.width,
                          (1-j/(len(heights)-1))*255/atlas.height)
                         for j in range(len(heights)) for i in range(len(angles))])
    remove |= (centres[:,0] > .155) & (centres[:,2] > .94) & (centres[:,2] < 1.14)
    remove |= (centres[:,0] > .245) & (centres[:,2] > 1.09) & (centres[:,2] < 1.18)
    floor_x = np.array([1., 0., 0.]) - floor_normal * floor_normal[0]
    floor_x /= np.linalg.norm(floor_x)
    basis = np.column_stack((floor_x, np.cross(floor_normal, floor_x), floor_normal))
    vertices = vertices @ basis - [0, 0, floor_offset + .009]
    vertices *= 1.75 / (1.75 - ground)
    mesh.vertices = vertices
    mesh.update_faces(~remove)
    mesh.remove_unreferenced_vertices()
    patch_vertices = (patch_vertices @ basis - [0,0,floor_offset+.009]) * (1.75/(1.75-ground))
    patch = trimesh.Trimesh(vertices=patch_vertices, faces=patch_faces,
        visual=trimesh.visual.texture.TextureVisuals(uv=patch_uv,material=mesh.visual.material),process=False)
    mesh = trimesh.util.concatenate([mesh,patch])
    image = mesh.visual.material.image.convert("RGB")
    image.format = "JPEG"
    mesh.visual.material = trimesh.visual.material.PBRMaterial(
        name="Scanned clothing and skin", baseColorTexture=image,
        baseColorFactor=[255, 255, 255, 255], metallicFactor=0, roughnessFactor=.78,
    )
    metadata = {"attribution": "Full body scan by Artec 3D — https://www.artec3d.com/3d-models/full-body-scan — CC BY 3.0; adapted"}
    shoulder = (np.array([.35, .04, 1.29]) @ basis - [0, 0, floor_offset + .009]) * (1.75 / (1.75 - ground))
    output = {}
    for posture in ("standing",):
        figure = mesh.copy()
        v = figure.vertices.copy()
        anchor = shoulder.copy()
        if posture == "standing":
            # The low work surface calls for a fixed bend at the hips.
            angle = np.deg2rad(58)
            def lean(points):
                d = points - [0, 0, .91]
                rotated = np.column_stack((d[:, 0], np.cos(angle)*d[:, 1] + np.sin(angle)*d[:, 2],
                                            -np.sin(angle)*d[:, 1] + np.cos(angle)*d[:, 2])) + [0, 0, .91]
                w = smoothstep(.83, 1.01, points[:, 2])[:, None]
                return points * (1-w) + rotated*w
            v = lean(v)
            anchor = lean(anchor[None, :])[0]
        # GLB stores Y-up; Blender and Three.js then agree on coordinates.
        figure.vertices = np.column_stack((v[:, 0], v[:, 2], -v[:, 1]))
        figure.visual.material.baseColorTexture.format = "JPEG"
        figure.metadata.update(metadata)
        figure.export(destination / f"{posture}.glb")
        output[posture] = {"right_shoulder": anchor.tolist(), "height_m": 1.75,
                           "floor_z": float(v[:, 2].min())}
        print(f"{posture}: {len(figure.faces)} triangles; shoulder {anchor.round(3)}")
    (destination / "placement.json").write_text(json.dumps(output, indent=2) + "\n")
    (destination / "LICENSE.txt").write_bytes(args.source.with_name("license.txt").read_bytes())


if __name__ == "__main__":
    main()
