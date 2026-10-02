"""Simulator-independent geometry, layouts, and geometric success checks (metres)."""

from dataclasses import asdict, dataclass
import numpy as np

TRAY_SIZE = np.array([0.160, 0.285, 0.008])
WALL = 0.008
WALL_HEIGHT = 0.025
DIVIDER_LENGTH = 0.140
DIVIDER_THICKNESS = 0.020
SLOT_CLEARANCE = 0.008
SLOT_Y = (-0.0475, 0.0475)
BIN_Y = (-0.095, 0.0, 0.095)
PART_SIZE = np.array([0.040, 0.045, 0.060])
COLORS = ((0.88, 0.29, 0.18), (0.12, 0.56, 0.64), (0.92, 0.67, 0.12))
# Each tuple is (centre, size), measured from the divider's bottom centre.
# A broad exposed tab provides a parallel-jaw grasp above the low divider wall.
DIVIDER_BOXES = (
    ((0, 0, 0.0175), (DIVIDER_LENGTH, DIVIDER_THICKNESS, 0.035)),
    ((0.040, 0, 0.050), (0.050, 0.042, 0.050)),
)
DIVIDER_GRASP = np.array([0.040, 0, 0.056])
PART_GRASP = np.array([0, 0, 0.045])


@dataclass(frozen=True)
class ScenarioSpecification:
    organizer_xy: tuple = (-0.015, 0.0)
    divider_xy: tuple = ((0.000, -0.230), (0.000, 0.230))
    part_xy: tuple = ((-0.040, -0.380), (-0.150, -0.150), (-0.040, 0.380))
    seed: int | None = None

    @classmethod
    def fixed(cls):
        return cls()

    @classmethod
    def from_layout(cls, data):
        unknown = set(data) - {
            "organizer_xy",
            "divider_xy",
            "part_xy",
            "comment",
            "source",
        }
        if unknown:
            raise ValueError(f"Unknown organizer layout fields: {sorted(unknown)}")
        base = cls()
        values = {}
        for key, shape in (
            ("organizer_xy", (2,)),
            ("divider_xy", (2, 2)),
            ("part_xy", (3, 2)),
        ):
            value = np.asarray(data.get(key, getattr(base, key)), dtype=float)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"{key} must be finite with shape {shape}, in metres.")
            values[key] = tuple(value) if len(shape) == 1 else tuple(map(tuple, value))
        return cls(**values)

    @classmethod
    def randomized(cls, seed):
        rng = np.random.default_rng(seed)
        base = cls()
        # Independent object translations; appearance never encodes mechanics.
        return cls(
            tuple(np.array(base.organizer_xy) + rng.uniform(-0.012, 0.012, 2)),
            tuple(
                map(
                    tuple,
                    np.array(base.divider_xy) + rng.uniform(-0.008, 0.008, (2, 2)),
                )
            ),
            tuple(
                map(tuple, np.array(base.part_xy) + rng.uniform(-0.010, 0.010, (3, 2)))
            ),
            seed,
        )

    def validate(self, desk_min, desk_size):
        """Reject off-table or overlapping initial objects instead of silently moving them."""
        boxes = [("organizer", np.array(self.organizer_xy), TRAY_SIZE[:2] / 2)]
        boxes += [
            (f"divider_{i}", np.array(xy), np.array([0.070, 0.021]))
            for i, xy in enumerate(self.divider_xy)
        ]
        boxes += [
            (f"part_{i}", np.array(xy), PART_SIZE[:2] / 2)
            for i, xy in enumerate(self.part_xy)
        ]
        lower, upper = np.array(desk_min[:2]), np.array(desk_min[:2]) + desk_size[:2]
        for index, (name, centre, half) in enumerate(boxes):
            if np.any(centre - half < lower + 0.010) or np.any(
                centre + half > upper - 0.010
            ):
                raise ValueError(f"{name} is outside the usable desk footprint.")
            for other, other_centre, other_half in boxes[:index]:
                if np.all(np.abs(centre - other_centre) < half + other_half + 0.008):
                    raise ValueError(f"Initial footprints overlap: {other} and {name}.")

    def to_dict(self):
        return {
            "task": "organizer",
            **asdict(self),
            "units": "metres",
            "tray_fixed": True,
            "physics": "rigid contact; clearance-fit dividers; uncalibrated material parameters",
        }


def union_box_mesh(boxes):
    """Closed exterior of an axis-aligned box union; no overlapping internal faces."""
    boxes = [
        (np.array(c) - np.array(s) / 2, np.array(c) + np.array(s) / 2) for c, s in boxes
    ]
    axes = [
        np.unique(np.round([v[a] for box in boxes for v in box], 9)) for a in range(3)
    ]
    shape = tuple(len(a) - 1 for a in axes)
    occupied = np.zeros(shape, dtype=bool)
    for idx in np.ndindex(shape):
        centre = np.array(
            [(axes[a][idx[a]] + axes[a][idx[a] + 1]) / 2 for a in range(3)]
        )
        occupied[idx] = any(
            np.all(centre > lo) and np.all(centre < hi) for lo, hi in boxes
        )
    vertices, faces, lookup = [], [], {}
    for idx in np.ndindex(shape):
        if not occupied[idx]:
            continue
        for axis in range(3):
            for direction in (-1, 1):
                adjacent = list(idx)
                adjacent[axis] += direction
                if 0 <= adjacent[axis] < shape[axis] and occupied[tuple(adjacent)]:
                    continue
                b, c = (axis + 1) % 3, (axis + 2) % 3
                corners = []
                for u, v in ((0, 0), (1, 0), (1, 1), (0, 1)):
                    grid = list(idx)
                    grid[axis] += int(direction == 1)
                    grid[b] += u
                    grid[c] += v
                    key = tuple(grid)
                    if key not in lookup:
                        lookup[key] = len(vertices)
                        vertices.append([axes[a][grid[a]] for a in range(3)])
                    corners.append(lookup[key])
                if direction == -1:
                    corners.reverse()
                faces.extend(
                    (
                        (corners[0], corners[1], corners[2]),
                        (corners[0], corners[2], corners[3]),
                    )
                )
    return np.asarray(vertices), np.asarray(faces, dtype=np.int32)
