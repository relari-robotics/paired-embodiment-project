import json

import numpy as np
import pytest

from superdex_scenarios import camera_profiles


def _profile(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_modeled_profile_places_camera_above_desk(monkeypatch, tmp_path):
    monkeypatch.setattr(camera_profiles, "_config_root", lambda: tmp_path)
    path = (
        camera_profiles.Path(__file__).parents[1]
        / "scenarios/ball_bowl/camera_calibration.json"
    )
    profile, digest = camera_profiles.resolve_profile(path)
    payload = camera_profiles.camera_spec_payload(
        profile, digest, [-0.0552, 0.0, 0.388]
    )
    assert payload["name"] == "desk_gemini_335"
    assert payload["calibration_status"] == "modeled"
    np.testing.assert_allclose(
        payload["world_from_camera_cv"]["position_m"], [-0.0552, 0, 0.888]
    )


def test_active_profile_overrides_default(monkeypatch, tmp_path):
    default = (
        camera_profiles.Path(__file__).parents[1]
        / "scenarios/ball_bowl/camera_calibration.json"
    )
    profile = _profile(default)
    profile["name"] = "measured"
    profile["calibration_status"] = "measured"
    profile["device"]["serial_number"] = "CP0H95300160"
    profile["desk_from_color_camera_cv"]["translation_m"] = [0.1, 0.2, 0.7]
    root = tmp_path / "calibration/cameras"
    (root / "profiles").mkdir(parents=True)
    (root / "profiles/measured.json").write_text(json.dumps(profile))
    (root / "active.json").write_text('{"profile":"measured"}')
    monkeypatch.setattr(camera_profiles, "_config_root", lambda: root)
    resolved, _ = camera_profiles.resolve_profile(default)
    assert resolved["name"] == "measured"


def test_invalid_active_profile_fails(monkeypatch, tmp_path):
    default = (
        camera_profiles.Path(__file__).parents[1]
        / "scenarios/ball_bowl/camera_calibration.json"
    )
    root = tmp_path / "calibration/cameras"
    (root / "profiles").mkdir(parents=True)
    (root / "profiles/bad.json").write_text('{"format":"unknown"}')
    (root / "active.json").write_text('{"profile":"bad"}')
    monkeypatch.setattr(camera_profiles, "_config_root", lambda: root)
    with pytest.raises(ValueError, match="unsupported Relari"):
        camera_profiles.resolve_profile(default)
