"""Camera metadata for ball-and-bowl recordings."""

from __future__ import annotations

import json
from pathlib import Path

from superdex_scenarios.embodiments import CameraSpec, EmbodimentModel

SCENARIO_ROOT = Path(__file__).resolve().parent


def _desk_camera() -> CameraSpec:
    calibration = json.loads(
        (SCENARIO_ROOT / "camera_calibration.json").read_text(encoding="utf-8")
    )
    camera = calibration["camera"]
    return CameraSpec(
        name=camera["name"],
        label="Desk ZED",
        kind="fixed",
        intrinsics=calibration["intrinsics"],
        world_from_camera_cv=calibration["simulation_world_from_camera_cv"],
        calibration_status="measured",
        provenance=calibration["provenance"],
    )


def camera_specs(embodiment: EmbodimentModel) -> tuple[CameraSpec, ...]:
    """Return only the cameras physically present for this embodiment."""
    return (_desk_camera(), *embodiment.cameras)


def camera_payloads(embodiment: EmbodimentModel) -> list[dict[str, object]]:
    return [camera.to_dict() for camera in camera_specs(embodiment)]
