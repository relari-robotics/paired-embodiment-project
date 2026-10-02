"""Camera metadata for ball-and-bowl recordings."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scenarios.ball_bowl.scenario import DESK_MIN, DESK_SIZE, DESK_TOP_Z
from superdex_scenarios.camera_profiles import camera_spec_payload, resolve_profile
from superdex_scenarios.embodiments import CameraSpec, EmbodimentModel

SCENARIO_ROOT = Path(__file__).resolve().parent


def _desk_camera() -> CameraSpec:
    profile, profile_hash = resolve_profile(SCENARIO_ROOT / "camera_calibration.json")
    desk_origin = np.asarray(
        [DESK_MIN[0] + 0.5 * DESK_SIZE[0], 0.0, DESK_TOP_Z], dtype=float
    )
    camera = camera_spec_payload(profile, profile_hash, desk_origin)
    return CameraSpec(
        name=camera["name"],
        label=camera["label"],
        kind=camera["kind"],
        intrinsics=camera["intrinsics"],
        world_from_camera_cv=camera["world_from_camera_cv"],
        calibration_status=camera["calibration_status"],
        provenance=camera["provenance"],
    )


def desk_camera_view_limits(
    low_z_world_m: float, high_z_world_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Wireframe of what the real desk camera sees between two world heights.

    The four image-corner rays of the calibrated desk camera are cut at both
    heights: a teleoperating hand is only tracked while it is inside this
    truncated pyramid. Returns ``(nodes, edges)`` for a curve network, with the
    low rectangle first.
    """

    camera = _desk_camera()
    intrinsics, pose = camera.intrinsics, camera.world_from_camera_cv
    position = np.asarray(pose["position_m"], dtype=float)
    right, up, forward = (
        np.asarray(pose[name], dtype=float)
        for name in ("right_world", "up_world", "forward_world")
    )
    width, height = intrinsics["width_px"], intrinsics["height_px"]
    nodes = []
    for plane_z in (low_z_world_m, high_z_world_m):
        for u, v in ((0, 0), (width, 0), (width, height), (0, height)):
            x = (u - intrinsics["cx_px"]) / intrinsics["fx_px"]
            y = (v - intrinsics["cy_px"]) / intrinsics["fy_px"]
            ray = x * right - y * up + forward  # OpenCV image y points down
            distance = (plane_z - position[2]) / ray[2]
            if not np.isfinite(distance) or distance <= 0.0:
                raise ValueError("the desk camera does not look at the requested heights")
            nodes.append(position + distance * ray)
    ring = [(0, 1), (1, 2), (2, 3), (3, 0)]
    edges = ring + [(a + 4, b + 4) for a, b in ring] + [(i, i + 4) for i in range(4)]
    return np.asarray(nodes, dtype=float), np.asarray(edges, dtype=int)


def camera_specs(embodiment: EmbodimentModel) -> tuple[CameraSpec, ...]:
    """Return only the cameras physically present for this embodiment."""
    return (_desk_camera(), *embodiment.cameras)


def camera_payloads(embodiment: EmbodimentModel) -> list[dict[str, object]]:
    return [camera.to_dict() for camera in camera_specs(embodiment)]
