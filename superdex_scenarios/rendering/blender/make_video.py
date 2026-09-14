# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Render a recorded episode with Blender and encode it as an MP4.

    python make_video.py --recording DIR --output video.mp4 [render options]

Runs Blender in the background with ``render_episode.py`` (every unknown option
is passed through to it), then encodes the PNG frames with ffmpeg.  Blender is
found through ``--blender``, the ``BLENDER`` environment variable, ``PATH``, or
the macOS application bundle.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

RENDER_SCRIPT = Path(__file__).resolve().parent / "render_episode.py"
BLENDER_CANDIDATES = (
    "/Applications/Blender.app/Contents/MacOS/Blender",
    "/usr/bin/blender",
    "/usr/local/bin/blender",
)


def find_blender(explicit: str | None) -> str:
    for candidate in (explicit, os.environ.get("BLENDER"), shutil.which("blender"), *BLENDER_CANDIDATES):
        if candidate and Path(candidate).exists():
            return candidate
    raise SystemExit("Blender not found; pass --blender or set BLENDER.")


def find_ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as error:  # pragma: no cover
        raise SystemExit("ffmpeg not found; install it or imageio-ffmpeg.") from error


def parse_device_indices(spec: str | None, workers: int) -> list[int | None]:
    """Parse one visible Cycles GPU index per local worker."""

    if spec is None:
        return [None] * workers
    try:
        indices = [int(value.strip()) for value in spec.split(",") if value.strip()]
    except ValueError as error:
        raise SystemExit("--devices must be a comma-separated list of integer GPU indices") from error
    if len(indices) != workers:
        raise SystemExit(f"--devices must contain exactly {workers} index(es)")
    if any(index < 0 for index in indices):
        raise SystemExit("--devices indices must be non-negative")
    return indices


def render_worker_command(
    blender: str,
    recording: Path,
    output: Path,
    fps: float,
    passthrough: list[str],
    worker_index: int,
    worker_count: int,
    device: str,
    device_index: int | None,
    resume: bool,
    overwrite: bool,
) -> list[str]:
    command = [
        blender,
        "-b",
        "-P",
        str(RENDER_SCRIPT),
        "--",
        "--recording",
        str(recording),
        "--output",
        str(output),
        "--fps",
        str(fps),
        "--device",
        device,
        "--worker-index",
        str(worker_index),
        "--worker-count",
        str(worker_count),
    ]
    if device_index is not None:
        command.extend(("--device-index", str(device_index)))
    if resume:
        command.append("--resume")
    if overwrite:
        command.append("--overwrite")
    command.extend(passthrough)
    return command


def run_render_workers(commands: list[list[str]]) -> None:
    """Run Blender workers concurrently and terminate siblings after failure."""

    processes = []
    try:
        for command in commands:
            print("Running:", " ".join(command))
            processes.append(subprocess.Popen(command))

        running = set(range(len(processes)))
        while running:
            for index in tuple(running):
                return_code = processes[index].poll()
                if return_code is None:
                    continue
                running.remove(index)
                if return_code != 0:
                    raise subprocess.CalledProcessError(return_code, commands[index])
            if running:
                time.sleep(0.2)
    except BaseException:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recording", required=True, help="directory written by --record-blender")
    parser.add_argument("--output", required=True, help="MP4 path")
    parser.add_argument("--blender", default=None, help="Blender executable")
    parser.add_argument("--fps", type=float, default=24.0, help="video frame rate (default: 24)")
    parser.add_argument("--frames-dir", default=None, help="keep PNG frames here (default: temporary)")
    parser.add_argument("--crf", type=int, default=18, help="H.264 quality, 0 best to 51 worst")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="concurrent Blender frame workers on this host (default: 1)",
    )
    parser.add_argument(
        "--devices",
        default=None,
        help="comma-separated visible Cycles GPU indices, one per worker",
    )
    parser.add_argument(
        "--device",
        choices=("AUTO", "OPTIX", "CUDA", "METAL", "CPU"),
        default="AUTO",
        type=str.upper,
        help="Cycles backend (default: AUTO; NVIDIA prefers OPTIX, macOS prefers METAL)",
    )
    parser.add_argument("--device-index", type=int, default=None, help="GPU index when using one worker")
    parser.add_argument("--frame-index", type=int, default=None, help="render one recorded frame")
    parser.add_argument("--resume", action="store_true", help="skip existing non-empty PNG frames")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing PNG frames")
    args, passthrough = parser.parse_known_args()

    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    if args.frame_index is not None and args.workers != 1:
        raise SystemExit("--frame-index cannot be combined with multiple workers")
    if args.workers > 1 and args.device_index is not None:
        raise SystemExit("use --devices for per-worker GPU selection when --workers is greater than one")
    if args.workers > 1 and args.devices is None and args.device != "CPU":
        raise SystemExit(
            "multiple GPU workers require --devices, for example --workers 2 --devices 0,1; "
            "this prevents accidentally competing on one GPU"
        )
    device_indices = parse_device_indices(args.devices, args.workers)
    if args.workers == 1 and args.device_index is not None:
        device_indices = [args.device_index]
    if args.frame_index is not None:
        passthrough = [*passthrough, "--frame-index", str(args.frame_index)]

    blender = find_blender(args.blender)
    ffmpeg = find_ffmpeg()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = Path(args.frames_dir).expanduser().resolve() if args.frames_dir else None
    temporary = tempfile.TemporaryDirectory(prefix="superdex-blender-") if frames_dir is None else None
    frames = frames_dir if frames_dir is not None else Path(temporary.name)  # type: ignore[union-attr]
    try:
        recording = Path(args.recording).expanduser().resolve()
        commands = [
            render_worker_command(
                blender=blender,
                recording=recording,
                output=frames,
                fps=args.fps,
                passthrough=passthrough,
                worker_index=worker_index,
                worker_count=args.workers,
                device=args.device,
                device_index=device_indices[worker_index],
                resume=args.resume,
                overwrite=args.overwrite,
            )
            for worker_index in range(args.workers)
        ]
        run_render_workers(commands)
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(args.fps),
                "-i",
                str(frames / "frame_%05d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                str(args.crf),
                "-movflags",
                "+faststart",
                str(output),
            ],
            check=True,
        )
        print(f"Video written: {output}")
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    sys.exit(main())
