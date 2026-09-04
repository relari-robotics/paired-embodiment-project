#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "h5py>=3.11",
#   "numpy>=2.0",
#   "rerun-sdk==0.34.0",
# ]
# ///
"""Build a self-contained Rerun recording from an exported task episode.

Run this file directly with ``uv run``.  It deliberately has no dependency on
SuperDex, so converting an already simulated episode never imports or modifies
the simulator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt
import rerun as rr
import rerun.blueprint as rrb

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPORT_DIR = REPOSITORY_ROOT / "scenarios" / "ball_bowl" / "exports" / "latest"
TIMELINE = "sim_time"
EXPECTED_RERUN_VERSION = "0.34.0"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=DEFAULT_EXPORT_DIR,
        help="directory made by runner.py --export-dir",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output .rrd (default: <export-dir>/ball_bowl.rrd)",
    )
    parser.add_argument(
        "--scene",
        type=Path,
        default=None,
        help="override the render manifest embedded in episode.json",
    )
    parser.add_argument(
        "--force-arrow-scale",
        type=float,
        default=0.01,
        metavar="M_PER_N",
        help="3D contact-arrow length scale in metres per Newton (default: 0.01)",
    )
    parser.add_argument(
        "--contact-chunk-steps",
        type=int,
        default=200,
        help="physics steps sent per Rerun batch (default: 200)",
    )
    parser.add_argument(
        "--max-contact-arrows-per-step",
        type=int,
        default=0,
        metavar="N",
        help="keep the N strongest surface samples per step; 0 keeps all (default: 0)",
    )
    parser.add_argument(
        "--skip-contact-arrows",
        action="store_true",
        help="omit individual 3D force arrows; dense aggregate force signals remain",
    )
    return parser.parse_args()


def _required(export_dir: Path, name: str) -> Path:
    path = export_dir / name
    if not path.is_file():
        raise SystemExit(f"Missing {path}; first run runner.py --export-dir.")
    return path


def _optional(export_dir: Path, name: str) -> Path | None:
    path = export_dir / name
    return path if path.is_file() else None


def _load_telemetry(path: Path) -> tuple[list[str], npt.NDArray[np.float64]]:
    with path.open(newline="", encoding="utf-8") as stream:
        columns = next(csv.reader(stream))
    values = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.shape[1] != len(columns):
        raise RuntimeError(
            f"Telemetry has {values.shape[1]} values but {len(columns)} headers."
        )
    if columns[:2] != ["step", "time_s"]:
        raise RuntimeError("Expected telemetry to begin with step,time_s.")
    if not np.all(np.diff(values[:, 1]) > 0.0):
        raise RuntimeError("Telemetry time_s must be strictly increasing.")
    if not np.all(np.isfinite(values)):
        raise RuntimeError("Telemetry contains NaN or infinite values.")
    return columns, values


def _flat_actor_name(name: str) -> str:
    """Keep every actor directly under /world/actors.

    Episode transforms are absolute world poses.  Flattening avoids accidentally
    composing a link's absolute pose with the robot root's absolute pose.
    """

    return re.sub(r"[^A-Za-z0-9_.-]+", "__", name).strip("_")


def _actor_entity(name: str) -> str:
    return f"world/actors/{_flat_actor_name(name)}"


def _render_assets(scene_path: Path) -> dict[str, tuple[Path, list[float]]]:
    manifest = json.loads(scene_path.read_text(encoding="utf-8"))
    assets: dict[str, tuple[Path, list[float]]] = {}
    for robot in manifest.get("actors", {}).get("articulated", []):
        for link in robot.get("links", []):
            model = link.get("renderModel")
            if not model:
                continue
            name = f"{robot['name']}/{link['name']}"
            path = (scene_path.parent / model).resolve()
            scale = [float(v) for v in link.get("renderModelScale", [1, 1, 1])]
            assets[name] = (path, scale)
    for actor in manifest.get("actors", {}).get("rigid", []):
        model = actor.get("renderModel")
        if not model:
            continue
        path = (scene_path.parent / model).resolve()
        scale = [float(v) for v in actor.get("renderModelScale", [1, 1, 1])]
        assets[actor["name"]] = (path, scale)
    missing = [str(path) for path, _ in assets.values() if not path.is_file()]
    if missing:
        raise RuntimeError("Missing render assets:\n  " + "\n  ".join(missing))
    return assets


def _log_scene(
    episode_path: Path,
    scene_path: Path,
) -> tuple[int, float, set[str]]:
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    if episode.get("format") != "superdex-transform-replay-v2":
        raise RuntimeError(f"Unsupported episode format: {episode.get('format')!r}")
    if episode.get("coordinateSystem") != "FLU":
        raise RuntimeError("Only the scenario's FLU episode coordinates are supported.")

    actors = [str(name) for name in episode["actors"]]
    frames = np.asarray(episode["frames"], dtype=np.float64)
    if frames.shape != (len(episode["frames"]), len(actors), 7):
        raise RuntimeError(f"Unexpected episode frame shape {frames.shape}.")
    fps = float(episode["fps"])
    times = np.arange(len(frames), dtype=np.float64) / fps
    time_column = rr.TimeColumn(TIMELINE, duration=times)

    rr.log("world", rr.ViewCoordinates.FLU, static=True)
    render_assets = _render_assets(scene_path)
    render_overrides = episode.get("renderOverrides", {})
    embedded: set[str] = set()
    for actor_index, actor_name in enumerate(actors):
        entity = _actor_entity(actor_name)
        rr.send_columns(
            entity,
            indexes=[time_column],
            columns=rr.Transform3D.columns(
                translation=frames[:, actor_index, :3],
                quaternion=frames[:, actor_index, 3:7],
            ),
        )
        render = render_assets.get(actor_name)
        if render is not None:
            asset_path, scale = render
            override = render_overrides.get(actor_name, {})
            scale = [float(v) for v in override.get("scale", scale)]
            color = override.get("color")
            albedo = None
            if color is not None:
                rgba = np.asarray([*color, 1.0], dtype=float)
                albedo = np.asarray(np.round(255.0 * rgba), dtype=np.uint8)
            # GLB uses the conventional right-handed Y-up frame.  Convert it to
            # the scenario's Z-up frame with +90 degrees around X. Anisotropic
            # Mochi scales must likewise be reordered from XYZ to XZY.
            glb_scale = [scale[0], scale[2], scale[1]]
            rr.log(
                f"{entity}/model",
                rr.Asset3D(path=asset_path, albedo_factor=albedo),
                rr.Transform3D(
                    scale=glb_scale,
                    quaternion=rr.Quaternion(
                        xyzw=[np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]
                    ),
                ),
                static=True,
            )
            embedded.add(actor_name)

    # The ground plane has collision geometry but no render asset in the Studio
    # manifest.  Give it a thin neutral slab so the 3D replay has visual context.
    ground = _actor_entity("ground")
    rr.log(
        f"{ground}/model",
        rr.Boxes3D(
            centers=[[0.0, 0.0, -0.006]],
            half_sizes=[[2.0, 2.0, 0.006]],
            colors=[[62, 68, 76]],
        ),
        static=True,
    )
    return len(frames), fps, embedded


def _log_video(video_entity: str, video_path: Path) -> int:
    video = rr.AssetVideo(path=video_path)
    rr.log(video_entity, video, static=True)
    frame_timestamps_ns = video.read_frame_timestamps_nanos()
    rr.send_columns(
        video_entity,
        indexes=[
            rr.TimeColumn(
                TIMELINE,
                duration=np.asarray(frame_timestamps_ns, dtype=np.float64) * 1e-9,
            )
        ],
        columns=rr.VideoFrameReference.columns_nanos(frame_timestamps_ns),
    )
    return len(frame_timestamps_ns)


def _log_camera(
    spec: dict[str, object],
    video_path: Path,
    episode_actors: set[str],
) -> tuple[str, int]:
    intrinsics = spec["intrinsics"]
    name = str(spec["name"])
    kind = str(spec["kind"])
    if kind == "fixed":
        pose = spec.get("world_from_camera_cv")
        if not isinstance(pose, dict):
            raise RuntimeError(f"Fixed camera {name!r} has no world pose.")
        rotation = np.column_stack(
            (
                np.asarray(pose["right_world"], dtype=float),
                -np.asarray(pose["up_world"], dtype=float),
                np.asarray(pose["forward_world"], dtype=float),
            )
        )
        camera = f"world/camera/{name}"
        rr.log(
            camera,
            rr.Transform3D(translation=pose["position_m"], mat3x3=rotation),
            static=True,
        )
    elif kind == "actor":
        actor = str(spec.get("actor_suffix", ""))
        if actor not in episode_actors:
            raise RuntimeError(
                f"Episode does not contain camera actor {actor!r} for {name!r}."
            )
        camera = f"{_actor_entity(actor)}/optical"
    else:
        raise RuntimeError(f"Unsupported camera kind {kind!r} for {name!r}.")

    rr.log(
        camera,
        rr.Pinhole(
            width=float(intrinsics["width_px"]),
            height=float(intrinsics["height_px"]),
            focal_length=[intrinsics["fx_px"], intrinsics["fy_px"]],
            principal_point=[intrinsics["cx_px"], intrinsics["cy_px"]],
            camera_xyz=rr.ViewCoordinates.RDF,
            image_plane_distance=0.20 if kind == "fixed" else 0.10,
        ),
        static=True,
    )
    return camera, _log_video(f"{camera}/image", video_path)


def _series_color(column: str, index: int) -> list[int]:
    component_colors = {
        "_x_n": [232, 80, 75, 255],
        "_y_n": [72, 190, 116, 255],
        "_z_n": [70, 128, 235, 255],
        "_x_nm": [232, 80, 75, 255],
        "_y_nm": [72, 190, 116, 255],
        "_z_nm": [70, 128, 235, 255],
    }
    for suffix, color in component_colors.items():
        if column.endswith(suffix):
            return color
    palette = [
        [70, 128, 235, 255],
        [242, 158, 55, 255],
        [84, 186, 112, 255],
        [214, 83, 94, 255],
        [158, 102, 204, 255],
        [63, 180, 183, 255],
        [220, 103, 172, 255],
    ]
    return palette[index % len(palette)]


def _series_label(column: str) -> str:
    parts = column.split("/")
    if parts[0] == "joint":
        return f"{parts[1]} · {parts[-1]}"
    if parts[0] == "contact":
        return f"{'/'.join(parts[1:-1])} · {parts[-1]}"
    return parts[-1]


def _log_telemetry(columns: list[str], values: npt.NDArray[np.float64]) -> int:
    times = values[:, 1]
    time_column = rr.TimeColumn(TIMELINE, duration=times)
    signal_count = 0
    for column_index, column in enumerate(columns[2:], start=2):
        entity = f"signals/{column}"
        rr.log(
            entity,
            rr.SeriesLines(
                names=[_series_label(column)],
                colors=[_series_color(column, column_index)],
            ),
            static=True,
        )
        rr.send_columns(
            entity,
            indexes=[time_column],
            columns=rr.Scalars.columns(scalars=values[:, column_index]),
        )
        signal_count += 1
    return signal_count


def _strongest_per_step(
    contacts: npt.NDArray,
    limit: int,
) -> npt.NDArray:
    if limit <= 0 or len(contacts) == 0:
        return contacts
    kept: list[npt.NDArray] = []
    steps = contacts["step"]
    boundaries = np.flatnonzero(np.diff(steps)) + 1
    for group in np.split(contacts, boundaries):
        if len(group) <= limit:
            kept.append(group)
            continue
        magnitudes = np.linalg.norm(group["force_on_actor_a_world_n"], axis=1)
        indices = np.argpartition(magnitudes, -limit)[-limit:]
        kept.append(group[np.sort(indices)])
    if not kept:
        return contacts[:0]
    return np.concatenate(kept)


def _contact_colors(forces: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
    magnitudes = np.linalg.norm(forces, axis=1)
    strength = np.clip(magnitudes / 15.0, 0.0, 1.0)
    colors = np.empty((len(forces), 4), dtype=np.uint8)
    colors[:, 0] = 255
    colors[:, 1] = np.asarray(205.0 - 150.0 * strength, dtype=np.uint8)
    colors[:, 2] = np.asarray(45.0 - 25.0 * strength, dtype=np.uint8)
    colors[:, 3] = 220
    return colors


def _log_contact_arrows(
    path: Path,
    telemetry_steps: npt.NDArray[np.int64],
    force_scale: float,
    chunk_steps: int,
    max_per_step: int,
) -> tuple[int, int]:
    if force_scale <= 0.0:
        raise SystemExit("--force-arrow-scale must be positive.")
    if chunk_steps <= 0:
        raise SystemExit("--contact-chunk-steps must be positive.")
    if max_per_step < 0:
        raise SystemExit("--max-contact-arrows-per-step cannot be negative.")

    entity = "world/contact_force_arrows"
    source_count = 0
    logged_count = 0
    with h5py.File(path, "r") as archive:
        dataset = archive["contacts"]
        rate_hz = float(archive.attrs["physics_rate_hz"])
        all_contact_steps = np.asarray(dataset["step"], dtype=np.int64)
        if len(all_contact_steps) and np.any(np.diff(all_contact_steps) < 0):
            raise RuntimeError("contacts.h5 must be sorted by step.")
        source_count = len(all_contact_steps)

        first_step = int(telemetry_steps[0])
        final_step = int(telemetry_steps[-1])
        for start in range(first_step, final_step + 1, chunk_steps):
            stop = min(start + chunk_steps, final_step + 1)
            lo = int(np.searchsorted(all_contact_steps, start, side="left"))
            hi = int(np.searchsorted(all_contact_steps, stop, side="left"))
            contacts = dataset[lo:hi]
            contacts = _strongest_per_step(contacts, max_per_step)
            logged_count += len(contacts)

            steps = np.arange(start, stop, dtype=np.int64)
            lengths = np.bincount(
                np.asarray(contacts["step"], dtype=np.int64) - start,
                minlength=len(steps),
            )
            origins = np.asarray(contacts["position_a_world_m"], dtype=np.float32)
            forces = np.asarray(contacts["force_on_actor_a_world_n"], dtype=np.float32)
            vectors = forces * np.float32(force_scale)
            colors = _contact_colors(forces)
            radii = np.full(len(forces), 0.0012, dtype=np.float32)

            rr.send_columns(
                entity,
                indexes=[rr.TimeColumn(TIMELINE, duration=steps / rate_hz)],
                columns=[
                    *rr.Arrows3D.columns(
                        origins=origins,
                        vectors=vectors,
                        colors=colors,
                        radii=radii,
                    ).partition(lengths=lengths),
                ],
            )
    return source_count, logged_count


def _blueprint(
    columns: list[str],
    camera_entities: list[tuple[str, str]],
    joint_metadata: list[dict[str, object]],
    embodiment: str,
) -> rrb.Blueprint:
    torque_by_dof = {
        int(joint["dof_index"]): f"/signals/joint/{joint['name']}/motor_torque_nm"
        for joint in joint_metadata
    }
    hand_dofs = {
        int(joint["dof_index"])
        for joint in joint_metadata
        if str(joint["joint_name"]).startswith(("openarm_right_finger_joint", "joint_"))
    }
    controlled_dofs = {
        int(joint["dof_index"])
        for joint in joint_metadata
        if bool(joint.get("controlled"))
    }
    arm_torques = [
        path
        for dof, path in torque_by_dof.items()
        if dof in controlled_dofs - hand_dofs
    ]
    hand_torques = [path for dof, path in torque_by_dof.items() if dof in hand_dofs]
    parked_torques = [
        path for dof, path in torque_by_dof.items() if dof not in controlled_dofs
    ]
    contact_force_norms = [
        f"/signals/{column}"
        for column in columns
        if column.startswith("contact/") and column.endswith("/force_norm_n")
    ]
    contact_torque_norms = [
        f"/signals/{column}"
        for column in columns
        if column.startswith("contact/") and column.endswith("/torque_com_norm_nm")
    ]

    replay_views = [
        rrb.Spatial3DView(
            origin="/world",
            name="physics replay + contact-force arrows",
        )
    ]
    if camera_entities:
        label, entity = camera_entities[0]
        replay_views.insert(0, rrb.Spatial2DView(origin=f"/{entity}", name=label))

    logical_root = "gripper" if embodiment == "openarm_v2" else "hand"
    replay = rrb.Vertical(
        rrb.Horizontal(*replay_views),
        rrb.Horizontal(
            rrb.TimeSeriesView(
                origin=f"/signals/{logical_root}",
                contents=[f"/signals/{logical_root}/**"],
                name=f"{logical_root} physical forces + joint-side torque",
            ),
            rrb.TimeSeriesView(
                origin="/signals/contact/blue_ball",
                contents=["/signals/contact/blue_ball/**"],
                name="ball aggregate contact wrench",
            ),
        ),
        row_shares=[3, 2],
        name="Replay",
    )
    torques = rrb.Horizontal(
        rrb.TimeSeriesView(
            origin="/signals/joint",
            contents=arm_torques,
            name="controlled arm motor torques [N·m]",
        ),
        rrb.TimeSeriesView(
            origin="/signals/joint",
            contents=hand_torques,
            name="hand/gripper joint torques [N·m]",
        ),
        rrb.TimeSeriesView(
            origin="/signals/joint",
            contents=parked_torques,
            name="uncontrolled joint torques [N·m]",
        ),
        name="Motor torques",
    )
    contacts = rrb.Horizontal(
        rrb.Spatial3DView(
            origin="/world",
            name="individual surface contact forces",
        ),
        rrb.Vertical(
            rrb.TimeSeriesView(
                origin="/signals/contact",
                contents=contact_force_norms,
                name="all actor contact-force norms [N]",
            ),
            rrb.TimeSeriesView(
                origin="/signals/contact",
                contents=contact_torque_norms,
                name="all actor contact-torque norms [N·m]",
            ),
        ),
        column_shares=[1, 1],
        name="Contacts",
    )
    tabs: list[object] = [replay]
    if camera_entities:
        tabs.append(
            rrb.Horizontal(
                *(
                    rrb.Spatial2DView(origin=f"/{entity}", name=label)
                    for label, entity in camera_entities
                ),
                name="Cameras",
            )
        )
    tabs.extend((torques, contacts))
    return rrb.Blueprint(
        rrb.Tabs(*tabs, active_tab=0),
        rrb.TimePanel(state="collapsed", timeline=TIMELINE, playback_speed=1.0),
        collapse_panels=True,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = _parse_args()
    export_dir = args.export_dir.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else export_dir / "ball_bowl.rrd"
    )
    if output.suffix.lower() != ".rrd":
        raise SystemExit("--output must end in .rrd")

    telemetry_path = _required(export_dir, "telemetry.csv")
    telemetry_metadata_path = _required(export_dir, "telemetry_metadata.json")
    contacts_path = _required(export_dir, "contacts.h5")
    episode_path = _required(export_dir, "episode.json")

    columns, telemetry = _load_telemetry(telemetry_path)
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    telemetry_metadata = json.loads(telemetry_metadata_path.read_text(encoding="utf-8"))
    episode_actors = {str(actor) for actor in episode["actors"]}
    specification = episode.get("scenario", {})
    embodiment = str(episode.get("embodiment", "unknown"))
    if args.scene is not None:
        scene_path = args.scene.expanduser().resolve()
    else:
        manifest = str(episode.get("renderManifest", ""))
        if not manifest:
            raise SystemExit("episode.json has no renderManifest; pass --scene.")
        scene_path = REPOSITORY_ROOT / manifest.lstrip("/")
    if not scene_path.is_file():
        raise SystemExit(f"Render manifest is missing: {scene_path}")
    output.parent.mkdir(parents=True, exist_ok=True)

    rr.init("superdex/ball_bowl", strict=True)
    rr.save(output)
    try:
        print("Writing embedded 3D scene and actor replay...")
        replay_frames, replay_fps, embedded_actors = _log_scene(
            episode_path, scene_path
        )
        camera_entities: list[tuple[str, str]] = []
        video_frame_counts: dict[str, int] = {}
        for camera_spec in episode.get("cameras", ()):
            camera_name = str(camera_spec["name"])
            video_path = _optional(export_dir, f"{camera_name}.mp4")
            if video_path is None:
                print(f"Skipping absent {camera_name}.mp4")
                continue
            print(f"Writing {camera_name} video...")
            camera_entity, frame_count = _log_camera(
                camera_spec,
                video_path,
                episode_actors,
            )
            camera_entities.append(
                (str(camera_spec.get("label", camera_name)), camera_entity)
            )
            video_frame_counts[camera_name] = frame_count
        print(f"Writing {len(columns) - 2} dense telemetry signals...")
        signal_count = _log_telemetry(columns, telemetry)

        source_contacts = 0
        logged_contacts = 0
        if not args.skip_contact_arrows:
            print("Writing individual 3D contact-force arrows...")
            source_contacts, logged_contacts = _log_contact_arrows(
                contacts_path,
                np.asarray(telemetry[:, 0], dtype=np.int64),
                args.force_arrow_scale,
                args.contact_chunk_steps,
                args.max_contact_arrows_per_step,
            )

        arrow_note = (
            "omitted by request"
            if args.skip_contact_arrows
            else (
                f"{logged_contacts:,} of {source_contacts:,} individual nonzero "
                f"surface samples, vectors shown at {args.force_arrow_scale:g} m/N"
            )
        )
        video_note = ", ".join(
            f"{name}: {frames:,} frames" for name, frames in video_frame_counts.items()
        )
        info = f"""# Ball-to-bowl physical simulation

- Embodiment: `{embodiment}`; only its right arm/hand chain is motor-controlled.
- Timeline: `{TIMELINE}` in simulated seconds.
- Embedded camera videos: {video_note}.
- Physics replay: {replay_frames:,} frames at {replay_fps:.3f} Hz; {len(embedded_actors)} embedded GLB actors.
- Dense telemetry: {len(telemetry):,} samples and {signal_count} signals at 400 Hz.
- Individual contact arrows: {arrow_note}.
- Randomized scenario: `{json.dumps(specification, separators=(",", ":"))}`

The 3D arrows are deliberately length-scaled for legibility. Exact aggregate
forces and center-of-mass torques are under `signals/contact`; exact logical
digit/jaw ball forces and joint torques are under `signals/hand` or
`signals/gripper`. The companion
`contacts.h5` remains the lossless source for positions, actor pairs, normals,
velocities, quadrature weights, and unscaled per-surface-sample force vectors.

Motor torque is Mochi's simulated joint-side generalized actuator force. The
reported values are not pre-gearbox motor-shaft estimates.
"""
        rr.log(
            "info/readme",
            rr.TextDocument(info, media_type=rr.MediaType.MARKDOWN),
            static=True,
        )
        rr.send_blueprint(
            _blueprint(
                columns,
                camera_entities,
                list(telemetry_metadata["joints"]),
                embodiment,
            )
        )
    finally:
        rr.disconnect()

    size_mib = output.stat().st_size / (1024 * 1024)
    print(f"Wrote {output}")
    print(f"  size: {size_mib:.1f} MiB")
    print(f"  sha256: {_sha256(output)}")
    print(f"  Rerun SDK: {rr.__version__} (expected {EXPECTED_RERUN_VERSION})")


if __name__ == "__main__":
    main()
