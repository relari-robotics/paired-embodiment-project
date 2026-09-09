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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recording", required=True, help="directory written by --record-blender")
    parser.add_argument("--output", required=True, help="MP4 path")
    parser.add_argument("--blender", default=None, help="Blender executable")
    parser.add_argument("--fps", type=float, default=24.0, help="video frame rate (default: 24)")
    parser.add_argument("--frames-dir", default=None, help="keep PNG frames here (default: temporary)")
    parser.add_argument("--crf", type=int, default=18, help="H.264 quality, 0 best to 51 worst")
    args, passthrough = parser.parse_known_args()

    blender = find_blender(args.blender)
    ffmpeg = find_ffmpeg()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = Path(args.frames_dir).expanduser().resolve() if args.frames_dir else None
    temporary = tempfile.TemporaryDirectory(prefix="superdex-blender-") if frames_dir is None else None
    frames = frames_dir if frames_dir is not None else Path(temporary.name)  # type: ignore[union-attr]
    try:
        command = [
            blender,
            "-b",
            "-P",
            str(RENDER_SCRIPT),
            "--",
            "--recording",
            str(Path(args.recording).expanduser().resolve()),
            "--output",
            str(frames),
            "--fps",
            str(args.fps),
            *passthrough,
        ]
        print("Running:", " ".join(command))
        subprocess.run(command, check=True)
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
