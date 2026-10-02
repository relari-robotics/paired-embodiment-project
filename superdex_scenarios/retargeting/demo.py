"""Loader for the recorded human demonstrations in ``reference/``.

Each demonstration folder holds an RGB-D recording of a person doing one of
the tasks and a fitted MANO hand-pose track for both hands (see
``reference/README.md``).  This module reads the fitted track into NumPy
arrays so a retargeting pipeline can start from clean data:

    from superdex_scenarios.retargeting.demo import load_demo

    demo = load_demo("reference/ball-in-the-bowl")
    right = demo.hands["right"]
    right.joints_camera.shape      # (frames, 21, 3) metres, RGB camera frame
    right.wrist_camera.shape       # (frames, 3) MANO root translation
    demo.time_s.shape              # (frames,) seconds from the first frame

Only NumPy and ``pyarrow`` are required (``uv pip install pyarrow``).  Nothing
here imports the simulator, and nothing here retargets anything: the mapping
from these camera-frame hand tracks to the simulated workcell is the project.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

MANO_JOINT_NAMES: tuple[str, ...] = (
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
)
"""The 21-joint order used by ``joints_3d_camera`` and the 2D keypoints."""

HAND_BONES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)
"""Parent/child joint pairs, useful for drawing the skeleton."""

FINGERTIPS: tuple[int, ...] = (4, 8, 12, 16, 20)
THUMB_TIP, INDEX_TIP = 4, 8


@dataclass
class HandTrack:
    """The fitted MANO track of one hand over every frame of a demonstration.

    Frames without a fit (WiLoR detection gaps) hold ``nan`` and are flagged
    by ``present``.  All 3D quantities are metres in the RGB camera's OpenCV
    frame (x right, y down, z forward).
    """

    side: str
    frame_idx: npt.NDArray[np.int64]
    present: npt.NDArray[np.bool_]
    joints_camera: npt.NDArray[np.float64]
    """(frames, 21, 3) anatomical joints, see :data:`MANO_JOINT_NAMES`."""
    wrist_camera: npt.NDArray[np.float64]
    """(frames, 3) MANO root translation (close to the wrist joint)."""
    global_orient: npt.NDArray[np.float64]
    """(frames, 3) MANO root orientation as an axis-angle vector."""
    hand_pose: npt.NDArray[np.float64]
    """(frames, 45) MANO finger joint rotations (15 joints x axis-angle)."""
    betas: npt.NDArray[np.float64]
    """(10,) MANO shape parameters shared over the recording."""
    keypoints_2d_observed: npt.NDArray[np.float64]
    """(frames, 21, 2) detector landmarks in RGB pixels."""
    keypoints_2d_fitted: npt.NDArray[np.float64]
    """(frames, 21, 2) reprojection of the fitted joints in RGB pixels."""
    depth_supported: npt.NDArray[np.bool_]
    """(frames, 21) whether registered depth backed each joint."""
    depth_residual_m: npt.NDArray[np.float64]
    """(frames, 21) fitted-skin minus measured depth where supported."""
    reprojection_error_px: npt.NDArray[np.float64]
    """(frames, 21) per-joint 2D error of the fit."""

    @property
    def fingertips_camera(self) -> npt.NDArray[np.float64]:
        return self.joints_camera[:, list(FINGERTIPS), :]

    def aperture_m(self) -> npt.NDArray[np.float64]:
        """Thumb-tip to index-tip distance per frame, a common gripper proxy."""
        return np.linalg.norm(
            self.joints_camera[:, THUMB_TIP] - self.joints_camera[:, INDEX_TIP], axis=1
        )

    def palm_frame(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Palm origin and 3x3 basis per frame built from wrist, index and pinky MCP.

        Columns: x from the wrist toward the middle MCP, z the palm normal
        (right hand: pointing out of the palm), y completing a right-handed
        frame.  A convenient, model-free wrist orientation; MANO's
        ``global_orient`` is the alternative.
        """
        wrist = self.joints_camera[:, 0]
        index = self.joints_camera[:, 5]
        middle = self.joints_camera[:, 9]
        pinky = self.joints_camera[:, 17]
        x = middle - wrist
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
        across = index - pinky
        z = np.cross(x, across)
        if self.side == "left":
            z = -z
        z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-9)
        y = np.cross(z, x)
        basis = np.stack([x, y, z], axis=2)
        origin = (wrist + index + pinky) / 3.0
        return origin, basis


@dataclass
class DemoEpisode:
    root: Path
    name: str
    frames: int
    fps: float
    time_s: npt.NDArray[np.float64]
    """Seconds from the first frame (from the colour camera device clock)."""
    hands: dict[str, HandTrack]
    calibration: dict[str, Any]
    """RGB/depth intrinsics, distortion, and the depth-to-colour extrinsics."""
    episode: dict[str, Any] = field(default_factory=dict)
    """Fit configuration and aggregate quality metrics (``episode.json``)."""
    rgb_video: Path | None = None
    depth_video: Path | None = None

    @property
    def color_intrinsics(self) -> dict[str, float]:
        return dict(self.calibration["color_intrinsic"])

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "frames": self.frames,
            "fps": self.fps,
            "duration_s": round(float(self.time_s[-1] - self.time_s[0]), 3),
            "hands": {
                side: int(track.present.sum()) for side, track in self.hands.items()
            },
        }


def _segment_stem(root: Path) -> str:
    segments = json.loads((root / "segments.json").read_text(encoding="utf-8"))["segments"]
    return f"segment-{int(segments[0]['segment']):06d}"


def _read_parquet(path: Path) -> dict[str, list[Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            "Reading the demonstration hand tracks needs pyarrow: uv pip install pyarrow"
        ) from error
    return pq.read_table(path).to_pydict()


def _stack(rows: list[Any], shape: tuple[int, ...], count: int, index: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
    out = np.full((count, *shape), np.nan, dtype=float)
    for frame, value in zip(index, rows):
        out[frame] = np.asarray(value, dtype=float).reshape(shape)
    return out


def load_demo(root: str | Path) -> DemoEpisode:
    """Load one ``reference/<task>`` demonstration folder."""
    root = Path(root).expanduser().resolve()
    if not (root / "segments.json").is_file():
        raise FileNotFoundError(f"{root} is not a demonstration folder (no segments.json).")
    stem = _segment_stem(root)
    frames_csv = root / "streams" / "camera.overhead" / f"{stem}.frames.csv"
    with frames_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    frame_count = len(rows)
    device_us = np.asarray([float(r["color_device_timestamp_us"]) for r in rows])
    time_s = (device_us - device_us[0]) * 1e-6
    manifest = json.loads((root / "metadata" / "manifest.json").read_text(encoding="utf-8"))
    fps = float(manifest.get("fps", 30.0))

    table = _read_parquet(root / "derived" / "mano" / f"{stem}.handpose.parquet")
    sides = np.asarray(table["side"])
    frame_idx_all = np.asarray(table["frame_idx"], dtype=np.int64)
    hands: dict[str, HandTrack] = {}
    for side in ("left", "right"):
        mask = sides == side
        if not np.any(mask):
            continue
        idx = frame_idx_all[mask]
        pick = lambda key: [v for v, m in zip(table[key], mask) if m]  # noqa: E731
        present = np.zeros(frame_count, dtype=bool)
        present[idx] = True
        betas_rows = pick("betas")
        hands[side] = HandTrack(
            side=side,
            frame_idx=np.arange(frame_count, dtype=np.int64),
            present=present,
            joints_camera=_stack(pick("joints_3d_camera"), (21, 3), frame_count, idx),
            wrist_camera=_stack(pick("translation_m"), (3,), frame_count, idx),
            global_orient=_stack(pick("global_orient"), (3,), frame_count, idx),
            hand_pose=_stack(pick("hand_pose"), (45,), frame_count, idx),
            betas=np.asarray(betas_rows[0], dtype=float) if betas_rows else np.zeros(10),
            keypoints_2d_observed=_stack(pick("keypoints_2d_observed"), (21, 2), frame_count, idx),
            keypoints_2d_fitted=_stack(pick("keypoints_2d_fitted"), (21, 2), frame_count, idx),
            depth_supported=_stack(pick("depth_supported"), (21,), frame_count, idx) > 0.5,
            depth_residual_m=_stack(pick("depth_residual_m"), (21,), frame_count, idx),
            reprojection_error_px=_stack(pick("reprojection_error_px"), (21,), frame_count, idx),
        )
    calibration = json.loads(
        (root / "streams" / "camera.overhead" / "calibration.json").read_text(encoding="utf-8")
    )
    episode_path = root / "episode.json"
    episode = json.loads(episode_path.read_text(encoding="utf-8")) if episode_path.is_file() else {}
    rgb = root / "streams" / "camera.overhead" / f"{stem}.rgb.mp4"
    depth = root / "streams" / "camera.overhead" / f"{stem}.depth.mkv"
    return DemoEpisode(
        root=root,
        name=root.name,
        frames=frame_count,
        fps=fps,
        time_s=time_s,
        hands=hands,
        calibration=calibration,
        episode=episode,
        rgb_video=rgb if rgb.is_file() else None,
        depth_video=depth if depth.is_file() else None,
    )


def project_to_color_pixels(points_camera: npt.ArrayLike, calibration: dict[str, Any]) -> npt.NDArray[np.float64]:
    """Pinhole projection (no distortion) of camera-frame points to RGB pixels."""
    points = np.asarray(points_camera, dtype=float).reshape(-1, 3)
    k = calibration["color_intrinsic"]
    z = np.maximum(points[:, 2], 1e-6)
    return np.column_stack([k["fx"] * points[:, 0] / z + k["cx"], k["fy"] * points[:, 1] / z + k["cy"]])


__all__ = [
    "FINGERTIPS",
    "HAND_BONES",
    "MANO_JOINT_NAMES",
    "DemoEpisode",
    "HandTrack",
    "load_demo",
    "project_to_color_pixels",
]
