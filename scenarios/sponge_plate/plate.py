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

"""Turn a thin scanned plate shell into a solid collision body.

A scanned plate is a closed shell only a few millimetres thick.  A soft body
pressed onto it penetrates past the shell's mid-plane, is pushed out of the far
side, and falls through.  :func:`solidify_plate` samples the scan's top surface
on a regular grid (vertical ray casts, done with 2-D barycentric tests so no
spatial index is needed) and extrudes it down to a flat base.  The result keeps
the dish profile and rim exactly and is thick everywhere, so contact against it
is robust under pressing and impacts.

The heightfield is kept alongside the mesh: the wiping planner samples it to
follow the concave dish floor, and the cleanliness map is defined on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt


def read_stl(path: Path) -> npt.NDArray[np.float64]:
    """Read a binary or ASCII STL into an (n, 3, 3) triangle array."""
    data = path.read_bytes()
    if len(data) >= 84 and not data[:5].lower().startswith(b"solid"):
        count = int(np.frombuffer(data[80:84], dtype="<u4")[0])
        record = np.dtype(
            [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")]
        )
        return np.frombuffer(data[84 : 84 + count * record.itemsize], dtype=record)[
            "vertices"
        ].astype(np.float64)
    vertices = [
        [float(v) for v in line.split()[1:4]]
        for line in data.decode("ascii", "replace").splitlines()
        if line.strip().startswith("vertex")
    ]
    return np.asarray(vertices, dtype=np.float64).reshape(-1, 3, 3)


def top_heightfield(
    triangles: npt.NDArray[np.float64], step: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Height of the top-most surface on a grid; NaN where no triangle is above."""
    lo = triangles.reshape(-1, 3).min(axis=0)
    hi = triangles.reshape(-1, 3).max(axis=0)
    xs = np.arange(lo[0], hi[0] + step, step)
    ys = np.arange(lo[1], hi[1] + step, step)
    heights = np.full((len(xs), len(ys)), -np.inf)
    for a, b, c in triangles:
        i0 = int(np.searchsorted(xs, min(a[0], b[0], c[0])))
        i1 = int(np.searchsorted(xs, max(a[0], b[0], c[0]), "right"))
        j0 = int(np.searchsorted(ys, min(a[1], b[1], c[1])))
        j1 = int(np.searchsorted(ys, max(a[1], b[1], c[1]), "right"))
        if i1 <= i0 or j1 <= j0:
            continue
        denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(denominator) < 1e-14:
            continue
        gx, gy = np.meshgrid(xs[i0:i1], ys[j0:j1], indexing="ij")
        l1 = ((b[1] - c[1]) * (gx - c[0]) + (c[0] - b[0]) * (gy - c[1])) / denominator
        l2 = ((c[1] - a[1]) * (gx - c[0]) + (a[0] - c[0]) * (gy - c[1])) / denominator
        l3 = 1.0 - l1 - l2
        inside = (l1 >= -1e-9) & (l2 >= -1e-9) & (l3 >= -1e-9)
        z = l1 * a[2] + l2 * b[2] + l3 * c[2]
        block = heights[i0:i1, j0:j1]
        np.maximum(block, np.where(inside, z, -np.inf), out=block)
    heights[~np.isfinite(heights)] = np.nan
    return xs, ys, heights


def _remove_pinch_cells(cell: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    """Drop cells that touch the region only diagonally so the boundary is a set of simple loops."""
    cell = cell.copy()
    changed = True
    while changed:
        changed = False
        for i in range(1, cell.shape[0]):
            for j in range(1, cell.shape[1]):
                a, b = cell[i - 1, j - 1], cell[i, j]
                c, d = cell[i - 1, j], cell[i, j - 1]
                if a and b and not c and not d:
                    cell[i, j] = False
                    changed = True
                elif c and d and not a and not b:
                    cell[i, j - 1] = False
                    changed = True
    return cell


@dataclass(frozen=True)
class SolidPlate:
    """Solid plate mesh plus the heightfield it was built from (plate-local metres)."""

    vertices: npt.NDArray[np.float64]
    faces: npt.NDArray[np.int32]
    xs: npt.NDArray[np.float64]
    ys: npt.NDArray[np.float64]
    heights: npt.NDArray[np.float64]
    base_z: float
    scan_min_z: float = 0.0
    """Lowest Z of the source scan; the solid is the scan shifted by ``-scan_min_z``."""

    @property
    def top_z(self) -> float:
        return float(np.nanmax(self.heights))

    @property
    def center_xy(self) -> npt.NDArray[np.float64]:
        valid = np.isfinite(self.heights)
        return np.array(
            [
                0.5 * (self.xs[valid.any(axis=1)][0] + self.xs[valid.any(axis=1)][-1]),
                0.5 * (self.ys[valid.any(axis=0)][0] + self.ys[valid.any(axis=0)][-1]),
            ]
        )

    @property
    def outer_radius(self) -> float:
        valid = np.isfinite(self.heights)
        gx, gy = np.meshgrid(self.xs, self.ys, indexing="ij")
        offsets = np.column_stack([gx[valid], gy[valid]]) - self.center_xy
        return float(np.max(np.linalg.norm(offsets, axis=1)))

    def surface_height(self, x: float, y: float) -> float:
        """Bilinear top-surface height at a plate-local XY (NaN outside the plate)."""
        i = float(np.clip((x - self.xs[0]) / (self.xs[1] - self.xs[0]), 0, len(self.xs) - 1.001))
        j = float(np.clip((y - self.ys[0]) / (self.ys[1] - self.ys[0]), 0, len(self.ys) - 1.001))
        i0, j0 = int(i), int(j)
        fi, fj = i - i0, j - j0
        corners = self.heights[i0 : i0 + 2, j0 : j0 + 2]
        if not np.all(np.isfinite(corners)):
            return float("nan")
        return float(
            (1 - fi) * ((1 - fj) * corners[0, 0] + fj * corners[0, 1])
            + fi * ((1 - fj) * corners[1, 0] + fj * corners[1, 1])
        )

    def floor_height(self, radius: float) -> float:
        """Highest top-surface point within ``radius`` of the centre (the dish floor level)."""
        gx, gy = np.meshgrid(self.xs, self.ys, indexing="ij")
        inside = np.hypot(gx - self.center_xy[0], gy - self.center_xy[1]) <= radius
        return float(np.nanmax(np.where(inside, self.heights, np.nan)))


def solidify_plate(
    stl_path: Path, *, step: float = 0.003, base_depth: float = 0.006
) -> SolidPlate:
    """Build a solid plate body from a scanned shell (see module docstring).

    The scan is translated so its lowest point sits at ``z = 0``; the solid's
    flat base lies ``base_depth`` below that.
    """
    triangles = read_stl(stl_path)
    scan_min_z = float(triangles[..., 2].min())
    triangles = triangles - np.array([0.0, 0.0, scan_min_z])
    xs, ys, heights = top_heightfield(triangles, step)
    base_z = -float(base_depth)
    valid = np.isfinite(heights)
    nx, ny = heights.shape
    cell = valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:] & valid[1:, 1:]
    cell = _remove_pinch_cells(cell)
    used = np.zeros_like(valid)
    used[:-1, :-1] |= cell
    used[1:, :-1] |= cell
    used[:-1, 1:] |= cell
    used[1:, 1:] |= cell

    vertices: list[list[float]] = []
    top = -np.ones((nx, ny), dtype=int)
    bottom = -np.ones((nx, ny), dtype=int)
    for i, j in zip(*np.nonzero(used)):
        top[i, j] = len(vertices)
        vertices.append([xs[i], ys[j], heights[i, j]])
        bottom[i, j] = len(vertices)
        vertices.append([xs[i], ys[j], base_z])

    faces: list[tuple[int, int, int]] = []

    def wall(t0: int, t1: int, b0: int, b1: int) -> None:
        faces.append((t0, b0, b1))
        faces.append((t0, b1, t1))

    for i, j in zip(*np.nonzero(cell)):
        a, b, c, d = top[i, j], top[i + 1, j], top[i + 1, j + 1], top[i, j + 1]
        faces.extend([(a, b, c), (a, c, d)])
        a, b, c, d = bottom[i, j], bottom[i + 1, j], bottom[i + 1, j + 1], bottom[i, j + 1]
        faces.extend([(a, c, b), (a, d, c)])
        if j == 0 or not cell[i, j - 1]:
            wall(top[i, j], top[i + 1, j], bottom[i, j], bottom[i + 1, j])
        if j == ny - 2 or not cell[i, j + 1]:
            wall(top[i + 1, j + 1], top[i, j + 1], bottom[i + 1, j + 1], bottom[i, j + 1])
        if i == 0 or not cell[i - 1, j]:
            wall(top[i, j + 1], top[i, j], bottom[i, j + 1], bottom[i, j])
        if i == nx - 2 or not cell[i + 1, j]:
            wall(top[i + 1, j], top[i + 1, j + 1], bottom[i + 1, j], bottom[i + 1, j + 1])

    heights_used = np.where(used, heights, np.nan)
    return SolidPlate(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int32),
        xs=xs,
        ys=ys,
        heights=heights_used,
        base_z=base_z,
        scan_min_z=scan_min_z,
    )


def is_closed_surface(faces: npt.NDArray[np.int32]) -> bool:
    """True when every edge is shared by exactly two faces with opposite orientation."""
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    keys = edges[:, 0].astype(np.int64) * (edges.max() + 1) + edges[:, 1]
    reverse = edges[:, 1].astype(np.int64) * (edges.max() + 1) + edges[:, 0]
    _, counts = np.unique(keys, return_counts=True)
    return bool(np.all(counts == 1) and np.array_equal(np.sort(keys), np.sort(reverse)))


__all__ = ["SolidPlate", "is_closed_surface", "read_stl", "solidify_plate", "top_heightfield"]
