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

"""Plate cleanliness map for the sponge-and-plate task.

A grid of cells over the dish floor records which cells the sponge has wiped.
The module is pure NumPy so the metric can be unit tested without SuperDex.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

DIRT_RADIUS = 0.070
DIRT_CELL = 0.010
CLEAN_MIN_NORMAL_FORCE = 1.0e-4  # N per contact sample
CLEAN_MIN_SLIDING_SPEED = 0.010  # m/s
CLEAN_SUCCESS_COVERAGE = 0.85
HISTORY_STRIDE = 6  # physics steps between stored coverage history entries


class CleanlinessMap:
    """Grid of dish-floor cells that the sponge must wipe under pressure.

    A cell is cleaned by a sponge-to-plate contact sample whose normal force
    exceeds ``CLEAN_MIN_NORMAL_FORCE`` while the sponge material slides over
    the plate faster than ``CLEAN_MIN_SLIDING_SPEED``.  Cells are defined in
    plate-local XY within ``DIRT_RADIUS`` of the plate centre.
    """

    def __init__(self, specification: Any) -> None:
        """``specification`` needs a ``plate_center_xy`` (world XY of the dish centre)."""
        self.specification = specification
        self.center = np.asarray(specification.plate_center_xy, dtype=float)
        half = int(np.ceil(DIRT_RADIUS / DIRT_CELL))
        coordinates = (np.arange(-half, half + 1) + 0.5) * DIRT_CELL
        gx, gy = np.meshgrid(coordinates, coordinates, indexing="ij")
        self.cells = np.column_stack([gx.ravel(), gy.ravel()])
        inside = np.hypot(self.cells[:, 0], self.cells[:, 1]) <= DIRT_RADIUS
        self.cells = self.cells[inside]
        self.cleaned = np.zeros(len(self.cells), dtype=bool)
        self.first_cleaned_step = np.full(len(self.cells), -1, dtype=int)
        self.history: list[tuple[int, float]] = []
        self._half = half

    @property
    def coverage(self) -> float:
        return float(np.mean(self.cleaned)) if len(self.cells) else 0.0

    def _cell_index(self, local_xy: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
        """Map plate-local XY samples to cell rows (or -1 when outside the map)."""
        ij = np.floor(local_xy / DIRT_CELL).astype(int) + self._half
        side = 2 * self._half + 1
        valid = (ij >= 0).all(axis=1) & (ij < side).all(axis=1)
        flat = np.where(valid, ij[:, 0] * side + ij[:, 1], -1)
        lookup = -np.ones(side * side, dtype=int)
        cell_ij = np.floor(self.cells / DIRT_CELL).astype(int) + self._half
        lookup[cell_ij[:, 0] * side + cell_ij[:, 1]] = np.arange(len(self.cells))
        return np.where(flat >= 0, lookup[np.maximum(flat, 0)], -1)

    def update(
        self,
        step: int,
        positions: npt.NDArray[np.float64],
        normal_forces: npt.NDArray[np.float64],
        sliding_speeds: npt.NDArray[np.float64],
    ) -> None:
        """Mark cells touched by qualifying contact samples (world positions)."""
        if len(positions):
            qualifying = (normal_forces >= CLEAN_MIN_NORMAL_FORCE) & (
                sliding_speeds >= CLEAN_MIN_SLIDING_SPEED
            )
            if np.any(qualifying):
                local = positions[qualifying, :2] - self.center
                rows = self._cell_index(local)
                rows = rows[rows >= 0]
                newly = rows[~self.cleaned[rows]]
                self.cleaned[newly] = True
                self.first_cleaned_step[newly] = step
        self.history.append((int(step), self.coverage))

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "superdex-cleanliness-map-v1",
            "plate_center_xy_m": self.center.tolist(),
            "cell_size_m": DIRT_CELL,
            "dirt_radius_m": DIRT_RADIUS,
            "min_normal_force_n": CLEAN_MIN_NORMAL_FORCE,
            "min_sliding_speed_m_s": CLEAN_MIN_SLIDING_SPEED,
            "success_coverage": CLEAN_SUCCESS_COVERAGE,
            "coverage": self.coverage,
            "cells_local_xy_m": self.cells.round(4).tolist(),
            "cleaned": self.cleaned.tolist(),
            "first_cleaned_step": self.first_cleaned_step.tolist(),
            "coverage_history": [
                {"step": step, "coverage": round(value, 4)}
                for step, value in self.history[:: HISTORY_STRIDE]
            ],
        }


__all__ = [
    "CLEAN_MIN_NORMAL_FORCE",
    "CLEAN_MIN_SLIDING_SPEED",
    "CLEAN_SUCCESS_COVERAGE",
    "DIRT_CELL",
    "DIRT_RADIUS",
    "CleanlinessMap",
]
