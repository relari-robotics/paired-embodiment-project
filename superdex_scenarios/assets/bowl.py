"""Procedural bowl matching the bowl in the recorded demonstrations.

The recorded bowl (measured from the registered depth of the demonstrations)
is 175 mm across at the rim, 70 mm tall, with a flat 110 mm floor and a
smoothly curving wall; it has no lip.  This module revolves that profile into
a watertight triangle mesh for physics and writes the same mesh as a Y-up GLB
for the render pipelines.

    python -m superdex_scenarios.assets.bowl --glb scenarios/ball_bowl/embodiments/openarm_v2/studio/render/gray_bowl.glb
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import numpy.typing as npt

OUTER_RADIUS = 0.0875
"""Outer radius at the rim (metres)."""
HEIGHT = 0.070
"""Rim height above the desk (metres)."""
FLOOR_RADIUS = 0.055
"""Radius of the flat inner floor."""
FLOOR_THICKNESS = 0.006
WALL_THICKNESS = 0.004
WALL_EXPONENT = 2.0
"""Inner wall height grows with this power of the normalised radius (measured ~2)."""


def profile(scale: npt.ArrayLike = (1.0, 1.0, 1.0), wall_samples: int = 14) -> npt.NDArray[np.float64]:
    """Closed (r, z) polyline of the bowl's cross-section, from the axis at the bottom
    around the outside, over the rim, down the inside, back to the axis."""
    sx, _, sz = (float(v) for v in np.asarray(scale, dtype=float))
    inner_rim = OUTER_RADIUS - WALL_THICKNESS
    s = np.linspace(0.0, 1.0, wall_samples)
    inner_r = FLOOR_RADIUS + (inner_rim - FLOOR_RADIUS) * s
    inner_z = FLOOR_THICKNESS + (HEIGHT - FLOOR_THICKNESS) * s**WALL_EXPONENT
    outer_r = inner_r + WALL_THICKNESS
    outer_z = inner_z - FLOOR_THICKNESS * (1.0 - s)  # base sits on the desk, wall keeps its thickness
    outer_z[0] = 0.0
    points = [(0.0, 0.0), (outer_r[0], 0.0)]
    points += list(zip(outer_r[1:], np.maximum(outer_z[1:], 0.0)))
    points += list(zip(inner_r[::-1], inner_z[::-1]))
    points += [(0.0, FLOOR_THICKNESS)]
    pts = np.asarray(points, dtype=float)
    pts[:, 0] *= sx
    pts[:, 1] *= sz
    return pts


def mesh(scale: npt.ArrayLike = (1.0, 1.0, 1.0), segments: int = 64) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]:
    """Revolve :func:`profile` about Z. Returns (vertices, faces) with outward normals,
    Z up, base on z = 0, centred on the axis; ``scale`` is (x, y, z)."""
    sx, sy, sz = (float(v) for v in np.asarray(scale, dtype=float))
    prof = profile((1.0, 1.0, 1.0))
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    cos, sin = np.cos(angles), np.sin(angles)
    vertices: list[np.ndarray] = []
    rings: list[np.ndarray] = []
    for r, z in prof:
        if r < 1e-9:
            vertices.append(np.array([0.0, 0.0, z]))
            rings.append(np.full(segments, len(vertices) - 1))
        else:
            start = len(vertices)
            vertices.extend(np.column_stack([r * cos, r * sin, np.full(segments, z)]))
            rings.append(np.arange(start, start + segments))
    faces: list[tuple[int, int, int]] = []
    for a, b in zip(rings[:-1], rings[1:]):
        for k in range(segments):
            k1 = (k + 1) % segments
            quad = (a[k], a[k1], b[k1], b[k])
            if a[k] == a[k1]:      # apex ring (axis point)
                faces.append((quad[0], quad[2], quad[3]))
            elif b[k] == b[k1]:
                faces.append((quad[0], quad[1], quad[2]))
            else:
                faces.append((quad[0], quad[1], quad[2]))
                faces.append((quad[0], quad[2], quad[3]))
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=np.int32)
    # Orient outward: the profile runs counter-clockwise in (r, z) so the revolved
    # surface may be inward; fix by the signed volume.
    volume = np.einsum("ij,ij->i", np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), v[f[:, 0]]).sum() / 6.0
    if volume < 0.0:
        f = f[:, [0, 2, 1]]
    v[:, 0] *= sx
    v[:, 1] *= sy
    v[:, 2] *= sz
    return v, f


def write_glb(path: str | Path, scale: npt.ArrayLike = (1.0, 1.0, 1.0), color=(0.43, 0.46, 0.50)) -> Path:
    """Write the bowl as a Y-up GLB (the convention of the Studio render assets)."""
    import trimesh

    v, f = mesh(scale)
    y_up = np.column_stack([v[:, 0], v[:, 2], -v[:, 1]])
    tm = trimesh.Trimesh(y_up, f, process=False)
    tm.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=[int(255 * c) for c in color] + [255], metallicFactor=0.0, roughnessFactor=0.45
        )
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tm.export(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--glb", type=Path, nargs="+", required=True, help="GLB file(s) to write")
    args = parser.parse_args()
    for target in args.glb:
        write_glb(target)
        print(f"Wrote {target}")


if __name__ == "__main__":
    main()
