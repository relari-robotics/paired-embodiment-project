"""Anatomical arm cross sections in metres, shared by mesh exporters/renderers.

The longitudinal axis points from the shoulder to the wrist. The first radial
axis is lateral; the second is anterior. These are authored surfaces, not scans.
"""

import json
from pathlib import Path

import numpy as np


HUMAN_APPEARANCE = json.loads(
    Path(__file__).with_name("human_appearance.json").read_text(encoding="utf-8")
)


# Fraction, lateral radius, anterior radius. Smooth interpolation preserves
# the narrow epicondyles and wrist instead of inflating each joint into a ball.
PROFILES = {
    "upper_arm": np.array([
        [0, .052, .053], [.12, .057, .058], [.28, .050, .055],
        [.48, .043, .052], [.65, .036, .045], [.82, .031, .035],
        [1, .033, .030],
    ]),
    "forearm": np.array([
        [0, .033, .030], [.14, .039, .035], [.30, .040, .034],
        [.48, .035, .030], [.68, .029, .025], [.85, .026, .021],
        [1, .025, .019],
    ]),
}


def cross_section(part: str, fraction: float, angles: np.ndarray) -> np.ndarray:
    """Return lateral/anterior offsets with muscle, ulna, and tendon relief."""
    profile = PROFILES[part]
    t = float(np.clip(fraction, 0, 1))
    i = int(np.clip(np.searchsorted(profile[:, 0], t) - 1, 0, len(profile) - 2))
    blend = (t - profile[i, 0]) / (profile[i + 1, 0] - profile[i, 0])
    blend = blend * blend * (3 - 2 * blend)
    lateral, anterior = profile[i, 1:] * (1 - blend) + profile[i + 1, 1:] * blend

    def lobe(centre, width):
        delta = np.arctan2(np.sin(angles - centre), np.cos(angles - centre))
        return np.exp(-.5 * (delta / width) ** 2)

    if part == "upper_arm":
        # Deltoid insertion, anterior biceps belly, posterior triceps.
        relief = (.004 * np.exp(-((t - .17) / .18) ** 2) * lobe(0, .75)
                  + .005 * np.exp(-((t - .47) / .22) ** 2) * lobe(np.pi / 2, .65)
                  + .003 * np.exp(-((t - .43) / .29) ** 2) * lobe(-np.pi / 2, .8))
    else:
        # Brachioradialis sweeps toward the thumb; the ulna forms a subtle ridge.
        relief = (.0035 * np.exp(-((t - .27) / .24) ** 2) * lobe(.3 + .8 * t, .5)
                  + .0015 * np.sin(np.pi * t) ** 2 * lobe(np.pi, .20)
                  + .0012 * np.exp(-((t - .86) / .16) ** 2)
                  * (lobe(1.25, .11) + lobe(1.75, .11)))
    return np.column_stack(((lateral + relief) * np.cos(angles),
                            (anterior + relief) * np.sin(angles)))


def segment_mesh(part: str, length: float, rings: int = 96, sides: int = 64):
    """Closed mesh along +X with rounded, elliptical joint ends and outward faces."""
    angles = np.arange(sides) * (2 * np.pi / sides)
    vertices = []
    # Separate polar vertices avoid zero-area faces at the caps.
    cap_length = HUMAN_APPEARANCE["shoulder_cap_length_m"] if part == "upper_arm" else .018
    end_cap_length = .025 if part == "upper_arm" else .018
    samples = []
    for phi in np.linspace(-np.pi / 2, 0, 10, endpoint=False)[1:]:
        samples.append((cap_length * np.sin(phi), 0, np.cos(phi)))
    samples.extend((t * length, t, 1) for t in np.linspace(0, 1, rings))
    for phi in np.linspace(0, np.pi / 2, 10)[1:-1]:
        samples.append((length + end_cap_length * np.sin(phi), 1, np.cos(phi)))
    for x, t, scale in samples:
        section = cross_section(part, t, angles) * scale
        vertices.extend((x, y, z) for y, z in section)
    faces = []
    for ring in range(len(samples) - 1):
        for side in range(sides):
            a = ring * sides + side
            b = ring * sides + (side + 1) % sides
            faces.extend(((a, b, b + sides), (a, b + sides, a + sides)))
    first, last = len(vertices), len(vertices) + 1
    vertices.extend(((-cap_length, 0, 0), (length + end_cap_length, 0, 0)))
    base = (len(samples) - 1) * sides
    for side in range(sides):
        nxt = (side + 1) % sides
        faces.extend(((first, nxt, side), (last, base + side, base + nxt)))
    return np.asarray(vertices), np.asarray(faces)
