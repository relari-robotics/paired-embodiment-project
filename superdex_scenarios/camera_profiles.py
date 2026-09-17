"""Resolve the global Relari Gemini desk profile for simulation cameras."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from platformdirs import user_config_path

FORMAT = "relari-gemini335-desk-calibration-v1"


def _config_root() -> Path:
    return user_config_path("relari", appauthor=False) / "calibration" / "cameras"


def _validate(profile: dict) -> dict:
    if profile.get("format") != FORMAT or profile.get("schema_version") != 1:
        raise ValueError("unsupported Relari camera profile")
    if "gemini 335" not in str(profile.get("device", {}).get("model", "")).lower():
        raise ValueError("Superdex desk calibration supports only Orbbec Gemini 335")
    extrinsic = profile.get("desk_from_color_camera_cv", {})
    if (extrinsic.get("parent_frame"), extrinsic.get("child_frame")) != (
        "desk",
        "camera.color.opencv",
    ):
        raise ValueError("camera profile must contain desk<-camera.color.opencv")
    rotation = np.asarray(extrinsic.get("rotation_row_major"), dtype=float).reshape(
        3, 3
    )
    translation = np.asarray(extrinsic.get("translation_m"), dtype=float)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError("camera profile translation must be a finite 3-vector")
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-6
    ):
        raise ValueError("camera profile rotation must be proper and orthonormal")
    color = profile.get("rgbd_calibration", {}).get("color_intrinsic", {})
    if not all(key in color for key in ("width", "height", "fx", "fy", "cx", "cy")):
        raise ValueError("camera profile is missing Gemini color intrinsics")
    return profile


def resolve_profile(default_path: Path) -> tuple[dict, str]:
    active = _config_root() / "active.json"
    if active.exists():
        selection = json.loads(active.read_text(encoding="utf-8"))
        name = selection.get("profile")
        if not isinstance(name, str) or not name:
            raise ValueError("invalid Relari active camera selection")
        path = _config_root() / "profiles" / f"{name}.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"active Relari camera profile does not exist: {path}"
            )
    else:
        path = default_path
    raw = path.read_bytes()
    return _validate(json.loads(raw)), hashlib.sha256(raw).hexdigest()


def camera_spec_payload(profile: dict, profile_hash: str, desk_origin_world) -> dict:
    color = profile["rgbd_calibration"]["color_intrinsic"]
    extrinsic = profile["desk_from_color_camera_cv"]
    rotation = np.asarray(extrinsic["rotation_row_major"], dtype=float).reshape(3, 3)
    position = np.asarray(desk_origin_world, dtype=float) + np.asarray(
        extrinsic["translation_m"], dtype=float
    )
    return {
        "name": "desk_gemini_335",
        "label": "Desk Gemini 335",
        "kind": "fixed",
        "intrinsics": {
            "width_px": int(color["width"]),
            "height_px": int(color["height"]),
            "fx_px": float(color["fx"]),
            "fy_px": float(color["fy"]),
            "cx_px": float(color["cx"]),
            "cy_px": float(color["cy"]),
        },
        "world_from_camera_cv": {
            "position_m": position.tolist(),
            "right_world": rotation[:, 0].tolist(),
            "up_world": (-rotation[:, 1]).tolist(),
            "forward_world": rotation[:, 2].tolist(),
        },
        "calibration_status": profile["calibration_status"],
        "provenance": {
            **profile.get("provenance", {}),
            "profile": profile["name"],
            "profile_sha256": profile_hash,
            "device": profile["device"],
        },
    }
