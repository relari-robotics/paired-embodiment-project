"""Tea packet dimensions and validated, reproducible tabletop layouts (metres)."""

from dataclasses import asdict, dataclass
import numpy as np

BOX_SIZE = np.array([0.160, 0.285, 0.008])
WALL = 0.008
HEIGHT = 0.060
DIVIDER = 0.004
SLOT_Y = (-0.0475, 0.0475)
BIN_Y = (-0.095, 0.0, 0.095)
PACKET_SIZE = np.array([0.062, 0.010, 0.075])
GRASP = np.array([0.0, 0.0, 0.052 - PACKET_SIZE[2] / 2])
TEAS = ("rooibos", "chamomile", "earl_grey")
COLORS = ((0.65, 0.075, 0.035), (0.85, 0.56, 0.025), (0.025, 0.13, 0.32))


def box_components():
    boxes = [((0, 0, BOX_SIZE[2] / 2), BOX_SIZE)]
    for axis in (0, 1):
        for sign in (-1, 1):
            centre = np.array([0.0, 0.0, BOX_SIZE[2] + HEIGHT / 2])
            centre[axis] = sign * (BOX_SIZE[axis] - WALL) / 2
            size = np.array([*BOX_SIZE[:2], HEIGHT])
            size[axis] = WALL
            boxes.append((centre, size))
    boxes += [
        ((0, y, BOX_SIZE[2] + HEIGHT / 2), (BOX_SIZE[0] - 2 * WALL, DIVIDER, HEIGHT))
        for y in SLOT_Y
    ]
    return boxes


def holder_components():
    # Open-bottom, low presentation cradle. The packet rests on the desk.
    return [((0, sign * 0.007, 0.009), (0.074, 0.003, 0.018)) for sign in (-1, 1)]


def packet_surface(nx=16, nz=20, tea=0, detail=False):
    """Closed, centred pillow envelope shared by physics and rendering.

    Dense surface samples are important for contact with thin compartment walls.
    Optional render microgeometry is bounded by 0.23 mm; physical silhouette and
    seal thickness otherwise match. Returned UVs are local to one printed face.
    """
    w, thickness, h = PACKET_SIZE
    vertices, faces, uv = [], [], []
    for side in (-1, 1):
        for j in range(nz + 1):
            v = j / nz
            for i in range(nx + 1):
                u = i / nx
                x, z = (u - 0.5) * w, (v - 0.5) * h
                edge = min(u * w, (1 - u) * w, v * h, (1 - v) * h)
                t = np.clip((edge - 0.003) / 0.007, 0, 1)
                pillow = t * t * (3 - 2 * t)
                half = 0.00035 + 0.00435 * pillow
                if detail:
                    half += (
                        0.00018
                        * np.sin(u * 37 + v * 18 + tea)
                        * np.sin(v * 41 - u * 12)
                        * pillow
                    )
                    if edge < 0.003:
                        half += 0.000045 * (0.5 + 0.5 * np.cos(2 * np.pi * x / 0.00065))
                vertices.append((x, side * half, z))
                uv.append((u if side == -1 else 1 - u, v))
    stride, sheet = nx + 1, (nx + 1) * (nz + 1)
    for side in range(2):
        for j in range(nz):
            for i in range(nx):
                a = side * sheet + j * stride + i
                quad = (a, a + 1, a + stride + 1, a + stride)
                faces.append(quad if side == 0 else quad[::-1])
    perimeter = list(range(stride)) + [j * stride + nx for j in range(1, nz + 1)]
    perimeter += [nz * stride + i for i in range(nx - 1, -1, -1)]
    perimeter += [j * stride for j in range(nz - 1, 0, -1)]
    for a, b in zip(perimeter, perimeter[1:] + perimeter[:1]):
        faces.append((b, a, a + sheet, b + sheet))
    return np.asarray(vertices), np.asarray(faces), np.asarray(uv)


@dataclass(frozen=True)
class ScenarioSpecification:
    organizer_xy: tuple = (-0.015, 0.0)
    part_xy: tuple = ((-0.040, -0.380), (-0.160, -0.150), (-0.040, 0.380))
    tea_order: tuple = (0, 1, 2)
    seed: int | None = None

    @classmethod
    def fixed(cls):
        return cls()

    @classmethod
    def from_layout(cls, data):
        unknown = set(data) - {"organizer_xy", "part_xy", "tea_order", "comment"}
        if unknown:
            raise ValueError(f"Unknown tea layout fields: {sorted(unknown)}")
        base = cls()
        values = {}
        for key, shape in (("organizer_xy", (2,)), ("part_xy", (3, 2))):
            value = np.asarray(data.get(key, getattr(base, key)), dtype=float)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"{key} must be finite with shape {shape}")
            values[key] = tuple(value) if len(shape) == 1 else tuple(map(tuple, value))
        order = tuple(data.get("tea_order", base.tea_order))
        if len(order) != 3 or set(order) != {0, 1, 2}:
            raise ValueError("tea_order must be a permutation of [0, 1, 2]")
        return cls(**values, tea_order=order)

    @classmethod
    def randomized(cls, seed):
        rng = np.random.default_rng(seed)
        base = cls()
        return cls(
            tuple(np.array(base.organizer_xy) + rng.uniform(-0.010, 0.010, 2)),
            tuple(
                map(tuple, np.array(base.part_xy) + rng.uniform(-0.008, 0.008, (3, 2)))
            ),
            base.tea_order,
            seed,
        )

    def validate(self, desk_min, desk_size):
        # Validate direct construction as well as JSON input.
        self.from_layout({k: v for k, v in asdict(self).items() if k != "seed"})
        footprints = [("tea box", np.array(self.organizer_xy), BOX_SIZE[:2] / 2)]
        footprints += [
            (f"packet {i}", np.array(xy), np.array([0.037, 0.0095]))
            for i, xy in enumerate(self.part_xy)
        ]
        lower, upper = (
            np.asarray(desk_min[:2]),
            np.asarray(desk_min[:2]) + desk_size[:2],
        )
        for i, (name, centre, half) in enumerate(footprints):
            if np.any(centre - half < lower + 0.010) or np.any(
                centre + half > upper - 0.010
            ):
                raise ValueError(f"{name} is outside the usable desk footprint")
            for other, p, h in footprints[:i]:
                if np.all(np.abs(centre - p) < half + h + 0.008):
                    raise ValueError(f"Initial footprints overlap: {other} and {name}")

    def to_dict(self):
        return {
            "task": "tea_sorting",
            **asdict(self),
            "units": "metres",
            "tea_classes": TEAS,
            "packet_size_m": PACKET_SIZE.tolist(),
            "packet_mass_kg": 0.004,
            "box_fixed": True,
            "physics": "rigid sealed packets; physical jaw contact; no bending or tearing; uncalibrated friction",
            "observation": "privileged actor poses and semantic tea class, not image recognition",
        }
