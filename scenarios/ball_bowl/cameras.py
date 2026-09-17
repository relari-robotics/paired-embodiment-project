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


def camera_specs(embodiment: EmbodimentModel) -> tuple[CameraSpec, ...]:
    """Return only the cameras physically present for this embodiment."""
    return (_desk_camera(), *embodiment.cameras)


def camera_payloads(embodiment: EmbodimentModel) -> list[dict[str, object]]:
    return [camera.to_dict() for camera in camera_specs(embodiment)]
