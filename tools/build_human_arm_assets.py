"""Rebuild the authored anatomical GLBs (numpy, trimesh, Pillow required).

Run from the repository root: .venv/bin/python tools/build_human_arm_assets.py
The original actor origins and +X longitudinal axis are preserved.
"""

from pathlib import Path
import sys

import numpy as np
from PIL import Image
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from superdex_scenarios.rendering.arm_surface import HUMAN_APPEARANCE, segment_mesh

DESTINATION = Path(__file__).resolve().parents[1] / "superdex_scenarios/embodiments/assets"


def skin_texture():
    """Deterministic multiscale pigmentation and fine pores, embedded in the GLB."""
    rng = np.random.default_rng(731)
    size = 1024
    variation = np.zeros((size, size))
    for resolution, strength in ((16, 2.0), (64, 1.1), (256, .7), (1024, .65)):
        noise = Image.fromarray(rng.integers(0, 256, (resolution, resolution), dtype=np.uint8))
        variation += (np.asarray(noise.resize((size, size), Image.Resampling.BICUBIC)) / 255 - .5) * strength
    skin = HUMAN_APPEARANCE["skin"]
    base = np.asarray(skin["base_color_linear"])
    srgb = np.where(base <= .0031308, base * 12.92, 1.055 * base ** (1 / 2.4) - .055)
    color = srgb * 255 + variation[..., None] * np.array([5, 4, 3])
    # Sparse freckles, restrained enough to avoid a speckled/plastic appearance.
    freckles = rng.random((size, size)) > .999
    color[freckles] *= .8
    albedo = Image.fromarray(np.clip(color, 0, 255).astype(np.uint8))
    pores = rng.normal(0, .035, (512, 512))
    dy, dx = np.gradient(pores)
    normals = np.stack((-dx, -dy, np.ones_like(dx)), axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    normal_map = Image.fromarray(np.uint8(np.clip(normals * .5 + .5, 0, 1) * 255))
    roughness = np.zeros((512, 512, 3), dtype=np.uint8)
    roughness[..., 0] = 255
    roughness[..., 1] = np.uint8(np.clip(skin["roughness"] + pores, 0, 1) * 255)
    return albedo, normal_map, Image.fromarray(roughness)


def main():
    texture, normal_map, roughness = skin_texture()
    material = trimesh.visual.material.PBRMaterial(
        name="Human skin", baseColorTexture=texture,
        baseColorFactor=[255, 255, 255, 255], metallicFactor=0,
        normalTexture=normal_map, metallicRoughnessTexture=roughness,
        roughnessFactor=1,
    )
    for part, length in (("upper_arm", .302), ("forearm", .246)):
        vertices, faces = segment_mesh(part, length)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        # Cylindrical coordinates; texture is intentionally low contrast at the seam.
        start_cap = HUMAN_APPEARANCE["shoulder_cap_length_m"] if part == "upper_arm" else .018
        end_cap = .025 if part == "upper_arm" else .018
        uv = np.column_stack(((vertices[:, 0] + start_cap) / (length + start_cap + end_cap),
                              np.arctan2(vertices[:, 2], vertices[:, 1]) / (2 * np.pi) + .5))
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
        assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0
        mesh.export(DESTINATION / f"{part}.glb")
        print(f"{part}: {len(vertices)} vertices, watertight")
    # A shaped deltoid cap, with an asymmetric insertion toward the upper arm.
    shoulder = trimesh.creation.icosphere(subdivisions=5)
    vertices = shoulder.vertices.copy()
    insertion = np.maximum(vertices[:, 0], 0)
    vertices *= np.column_stack((np.full(len(vertices), .065),
                                .058 - .010 * insertion, .061 - .010 * insertion))
    vertices[:, 2] += .006 * (1 - shoulder.vertices[:, 0] ** 2)
    shoulder.vertices = vertices
    uv = np.column_stack((np.arctan2(vertices[:, 1], vertices[:, 0]) / (2 * np.pi) + .5,
                          np.arccos(np.clip(shoulder.vertex_normals[:, 2], -1, 1)) / np.pi))
    shoulder.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    shoulder.export(DESTINATION / "shoulder_anchor.glb")
    print(f"shoulder_anchor: {len(vertices)} vertices, watertight")


if __name__ == "__main__":
    main()
