#!/usr/bin/env python
"""Rebuild a Rerun recording of a recorded demonstration from its raw files.

The demonstrations under ``reference/`` ship without their QA recordings to
keep the package small. This script recreates one from the data that is
shipped: the RGB video, the 16-bit depth video, the camera calibration, and
the fitted MANO hand tracks.

    python tools/make_demo_rrd.py reference/ball-in-the-bowl
    rerun reference/ball-in-the-bowl/episode.rrd

The recording has the RGB video with the observed and fitted 2D skeletons, and
a 3D view in the colour camera's frame with the fitted joints and bones, the
MANO mesh vertices, and the registered depth point cloud coloured from the
video. Left-hand overlays are magenta, right-hand overlays yellow, detector
landmarks cyan. Everything is an estimate; see ``reference/README.md``.

Needs ``pyarrow``, ``rerun-sdk``, and ``ffmpeg`` (the bundled ``imageio-ffmpeg``
build is used when none is on ``PATH``). Nothing here imports the simulator.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from superdex_scenarios.retargeting.demo import HAND_BONES, load_demo  # noqa: E402

TIMELINE = "time"
COLORS = {"left": [230, 60, 200], "right": [240, 200, 40], "observed": [40, 220, 230]}
BONE_STRIPS = np.asarray(HAND_BONES, dtype=int)


def _ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as error:  # pragma: no cover - depends on the environment
        raise SystemExit("ffmpeg is required to decode the videos; install it or imageio-ffmpeg.") from error


def _decode_frames(video: Path, every: int, count: int, pix_fmt: str, width: int, height: int, channels: int):
    """Yield (frame_index, array) for every ``every``-th frame using one ffmpeg pass."""
    if count <= 0:
        return
    wanted = [i * every for i in range(count)]
    command = [
        _ffmpeg(), "-loglevel", "error", "-i", str(video),
        "-vf", f"select='not(mod(n\\,{every}))'", "-vsync", "0",
        "-f", "rawvideo", "-pix_fmt", pix_fmt, "-",
    ]
    dtype = np.uint16 if pix_fmt.endswith("16le") else np.uint8
    frame_bytes = width * height * channels * np.dtype(dtype).itemsize
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    assert process.stdout is not None
    try:
        for index in wanted:
            data = process.stdout.read(frame_bytes)
            if len(data) < frame_bytes:
                break
            frame = np.frombuffer(data, dtype=dtype)
            yield index, frame.reshape(height, width, channels) if channels > 1 else frame.reshape(height, width)
    finally:
        process.stdout.close()
        process.wait()


def _rerun():
    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise SystemExit("rerun-sdk is required: uv pip install rerun-sdk==0.34.0") from error
    return rr, rrb


def _load_mesh_vertices(root: Path, stem: str) -> dict[str, dict[int, np.ndarray]]:
    import pyarrow.parquet as pq

    path = root / "derived" / "mano" / f"{stem}.handpose_mesh.parquet"
    if not path.is_file():
        return {}
    table = pq.read_table(path).to_pydict()
    out: dict[str, dict[int, np.ndarray]] = {"left": {}, "right": {}}
    for side, index, vertices in zip(table["side"], table["frame_idx"], table["vertices_3d_camera"]):
        out[str(side)][int(index)] = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("demo", type=Path, help="a reference/<task> folder")
    parser.add_argument("--output", type=Path, default=None, help="output .rrd (default: <demo>/episode.rrd)")
    parser.add_argument("--cloud-fps", type=float, default=10.0, help="point-cloud frames per second (default: 10)")
    parser.add_argument("--pixel-stride", type=int, default=4, help="depth pixel stride for the cloud (default: 4)")
    parser.add_argument("--max-depth", type=float, default=1.6, help="drop cloud points beyond this depth in metres")
    parser.add_argument("--no-cloud", action="store_true", help="skip the depth point cloud")
    parser.add_argument("--no-mesh", action="store_true", help="skip the MANO mesh vertices")
    parser.add_argument("--no-video", action="store_true", help="skip embedding the RGB video")
    args = parser.parse_args()

    rr, rrb = _rerun()
    demo = load_demo(args.demo)
    output = (args.output or demo.root / "episode.rrd").expanduser().resolve()
    stem = demo.rgb_video.stem.replace(".rgb", "") if demo.rgb_video else ""
    calibration = demo.calibration
    color = calibration["color_intrinsic"]
    depth_k = calibration["depth_intrinsic"]
    d2c = calibration["depth_to_color"]
    rotation = np.asarray(d2c["rotation_row_major"], dtype=float).reshape(3, 3)
    translation = np.asarray(d2c["translation_mm"], dtype=float) / 1000.0

    rr.init(f"demo/{demo.name}", strict=True)
    rr.save(str(output))
    try:
        rr.log("world", rr.ViewCoordinates.RDF, static=True)
        rr.log(
            "world/camera",
            rr.Pinhole(
                width=color["width"],
                height=color["height"],
                focal_length=[color["fx"], color["fy"]],
                principal_point=[color["cx"], color["cy"]],
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=0.3,
            ),
            static=True,
        )
        time_column = rr.TimeColumn(TIMELINE, duration=demo.time_s)

        if demo.rgb_video is not None and not args.no_video:
            print("Embedding the RGB video...")
            video = rr.AssetVideo(path=demo.rgb_video)
            rr.log("world/camera/image", video, static=True)
            stamps = np.asarray(video.read_frame_timestamps_nanos(), dtype=np.int64)
            count = min(len(stamps), demo.frames)
            rr.send_columns(
                "world/camera/image",
                indexes=[rr.TimeColumn(TIMELINE, duration=demo.time_s[:count])],
                columns=rr.VideoFrameReference.columns_nanos(stamps[:count]),
            )

        mesh = {} if args.no_mesh else _load_mesh_vertices(demo.root, stem)
        print("Logging the fitted hand tracks...")
        for side, track in demo.hands.items():
            color_rgb = COLORS[side]
            for frame in np.flatnonzero(track.present):
                rr.set_time(TIMELINE, duration=float(demo.time_s[frame]))
                joints = track.joints_camera[frame]
                rr.log(f"world/hands/{side}/joints", rr.Points3D(joints, colors=color_rgb, radii=0.004))
                rr.log(
                    f"world/hands/{side}/bones",
                    rr.LineStrips3D([joints[b] for b in BONE_STRIPS], colors=color_rgb, radii=0.0015),
                )
                rr.log(
                    f"world/camera/image/{side}/fitted",
                    rr.LineStrips2D([track.keypoints_2d_fitted[frame][b] for b in BONE_STRIPS], colors=color_rgb, radii=2.0),
                )
                rr.log(
                    f"world/camera/image/{side}/observed",
                    rr.Points2D(track.keypoints_2d_observed[frame], colors=COLORS["observed"], radii=3.0),
                )
                vertices = mesh.get(side, {}).get(int(frame))
                if vertices is not None:
                    rr.log(f"world/hands/{side}/mesh", rr.Points3D(vertices, colors=color_rgb, radii=0.0012))
            rr.reset_time()

        if demo.depth_video is not None and not args.no_cloud:
            stride = max(1, args.pixel_stride)
            every = max(1, int(round(demo.fps / args.cloud_fps)))
            frames = list(range(0, demo.frames, every))
            print(f"Decoding {len(frames)} depth and RGB frames for the point cloud...")
            rgb_frames = (
                dict(_decode_frames(demo.rgb_video, every, len(frames), "rgb24", color["width"], color["height"], 3))
                if demo.rgb_video
                else {}
            )
            v, u = np.mgrid[0:depth_k["height"]:stride, 0:depth_k["width"]:stride]
            u = u.astype(float); v = v.astype(float)
            total = 0
            for index, depth in _decode_frames(demo.depth_video, every, len(frames), "gray16le", depth_k["width"], depth_k["height"], 1):
                z = depth[::stride, ::stride].astype(float) / 1000.0
                valid = (z > 0.0) & (z < args.max_depth)
                x = (u - depth_k["cx"]) / depth_k["fx"] * z
                y = (v - depth_k["cy"]) / depth_k["fy"] * z
                points = np.stack([x[valid], y[valid], z[valid]], axis=1) @ rotation.T + translation
                colors = None
                image = rgb_frames.get(index)
                if image is not None:
                    pu = np.round(color["fx"] * points[:, 0] / points[:, 2] + color["cx"]).astype(int)
                    pv = np.round(color["fy"] * points[:, 1] / points[:, 2] + color["cy"]).astype(int)
                    inside = (pu >= 0) & (pu < color["width"]) & (pv >= 0) & (pv < color["height"])
                    points = points[inside]
                    colors = image[pv[inside], pu[inside]]
                rr.set_time(TIMELINE, duration=float(demo.time_s[index]))
                rr.log("world/points", rr.Points3D(points.astype(np.float32), colors=colors, radii=0.002))
                total += len(points)
            rr.reset_time()
            print(f"  {total:,} cloud points over {len(frames)} frames")

        rr.log(
            "info/readme",
            rr.TextDocument(
                f"# {demo.name}\n\nRebuilt by `tools/make_demo_rrd.py` from the RGB and depth videos, the "
                "calibration, and the fitted MANO tracks. Frame: the colour camera (x right, y down, "
                "z forward), metres. Cyan: detector landmarks; magenta: fitted left hand; yellow: fitted "
                "right hand. All layers are estimates.",
                media_type=rr.MediaType.MARKDOWN,
            ),
            static=True,
        )
        rr.send_blueprint(
            rrb.Blueprint(
                rrb.Horizontal(
                    rrb.Spatial2DView(origin="world/camera/image", name="RGB with 2D skeletons"),
                    rrb.Spatial3DView(origin="world", name="Camera-frame 3D"),
                ),
                rrb.TimePanel(timeline=TIMELINE),
                collapse_panels=True,
            )
        )
    finally:
        rr.disconnect()
    print(f"Wrote {output} ({output.stat().st_size / (1024 * 1024):.1f} MiB)")


if __name__ == "__main__":
    main()
