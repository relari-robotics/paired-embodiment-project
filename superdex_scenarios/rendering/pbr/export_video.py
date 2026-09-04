#!/usr/bin/env python3
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

"""Render a deterministic desk or wrist-camera MP4 from a physics replay."""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PBR_ROOT = Path(__file__).resolve().parent


class _ReplayHandler(SimpleHTTPRequestHandler):
    """Serve scenario assets and one replay that may live outside the tree."""

    replay_path: Path

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, directory=str(REPOSITORY_ROOT), **kwargs)

    def do_GET(self) -> None:
        if urllib.parse.urlsplit(self.path).path == "/__episode__.json":
            payload = self.replay_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        super().do_GET()

    def log_message(self, *_: object) -> None:
        pass


class _WebDriver:
    """Small WebDriver HTTP client; Selenium is intentionally not required."""

    def __init__(self, executable: str) -> None:
        self.port = self._free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.process = subprocess.Popen(
            [executable, "--port", str(self.port), "--log", "fatal"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.session_id: str | None = None

    @staticmethod
    def _free_port() -> int:
        import socket

        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def _request(
        self,
        method: str,
        path: str,
        payload: object | None = None,
        timeout: float = 60.0,
    ) -> object:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
        value = result.get("value")
        if isinstance(value, dict) and value.get("error"):
            raise RuntimeError(value.get("message", value["error"]))
        return value

    def start(self) -> None:
        payload = {
            "capabilities": {
                "alwaysMatch": {
                    "browserName": "firefox",
                    "moz:firefoxOptions": {
                        "args": ["-headless"],
                        "prefs": {
                            "media.autoplay.default": 0,
                            "webgl.disabled": False,
                            "webgl.force-enabled": True,
                        },
                    },
                }
            }
        }
        deadline = time.monotonic() + 20.0
        while True:
            try:
                value = self._request("POST", "/session", payload)
                assert isinstance(value, dict)
                self.session_id = str(value["sessionId"])
                return
            except (OSError, urllib.error.URLError):
                if self.process.poll() is not None:
                    raise RuntimeError("geckodriver exited before creating a session.")
                if time.monotonic() >= deadline:
                    raise RuntimeError("Timed out while starting headless Firefox.")
                time.sleep(0.1)

    def navigate(self, url: str) -> None:
        assert self.session_id is not None
        self._request("POST", f"/session/{self.session_id}/url", {"url": url})

    def execute(self, script: str, *args: object) -> object:
        assert self.session_id is not None
        return self._request(
            "POST",
            f"/session/{self.session_id}/execute/sync",
            {"script": script, "args": list(args)},
            timeout=120.0,
        )

    def close(self) -> None:
        if self.session_id is not None:
            try:
                self._request("DELETE", f"/session/{self.session_id}")
            except (OSError, urllib.error.URLError, RuntimeError):
                pass
            self.session_id = None
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.process.kill()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episode",
        type=Path,
        default=REPOSITORY_ROOT / "scenarios" / "ball_bowl" / "exports" / "replay.json",
        help="transform replay JSON produced by runner.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output MP4 (default: exports/latest/<camera>.mp4)",
    )
    parser.add_argument(
        "--camera",
        default="desk_zed",
        help="camera name embedded in the replay (default: desk_zed)",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--crf", type=int, default=18)
    return parser.parse_args()


def _wait_for_viewer(driver: _WebDriver, timeout: float = 120.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = driver.execute(
            "return window.superdexExport ? "
            "{ready: window.superdexExport.ready, error: window.superdexExport.error} "
            ": {ready: false, error: null};"
        )
        assert isinstance(state, dict)
        if state.get("error"):
            raise RuntimeError(f"PBR viewer failed: {state['error']}")
        if state.get("ready"):
            metadata = driver.execute("return window.superdexExport.metadata();")
            assert isinstance(metadata, dict)
            return metadata
        time.sleep(0.1)
    raise RuntimeError("Timed out loading PBR models in headless Firefox.")


def _ensure_renderer_dependencies() -> None:
    three_package = PBR_ROOT / "node_modules" / "three" / "package.json"
    if three_package.is_file():
        return
    npm = shutil.which("npm")
    if npm is None:
        raise RuntimeError(
            "Three.js is not installed and npm is unavailable; run npm ci in "
            f"{PBR_ROOT}."
        )
    print("Installing locked PBR renderer dependency with npm ci...")
    subprocess.run(
        [npm, "ci", "--ignore-scripts"],
        cwd=PBR_ROOT,
        check=True,
    )


def export_video(
    episode_path: Path,
    output_path: Path,
    fps: float,
    crf: int,
    camera_name: str = "desk_zed",
) -> None:
    episode_path = episode_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not episode_path.is_file():
        raise FileNotFoundError(f"Replay does not exist: {episode_path}")
    if fps <= 0.0:
        raise ValueError("--fps must be positive")
    if not 0 <= crf <= 51:
        raise ValueError("--crf must be between 0 and 51")
    ffmpeg = shutil.which("ffmpeg")
    geckodriver = shutil.which("geckodriver")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to export MP4 video.")
    if geckodriver is None:
        raise RuntimeError("geckodriver is required to drive the PBR renderer.")
    _ensure_renderer_dependencies()

    replay = json.loads(episode_path.read_text(encoding="utf-8"))
    replay_duration = (len(replay["frames"]) - 1) / float(replay["fps"])
    camera_metadata = next(
        (
            camera
            for camera in replay.get("cameras", ())
            if camera.get("name") == camera_name
        ),
        None,
    )
    if camera_metadata is None:
        available = [camera.get("name") for camera in replay.get("cameras", ())]
        raise ValueError(
            f"Replay has no camera {camera_name!r}; available cameras: {available}"
        )
    intrinsics = camera_metadata["intrinsics"]
    width = int(intrinsics["width_px"])
    height = int(intrinsics["height_px"])
    frame_count = round(replay_duration * fps) + 1
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _ReplayHandler.replay_path = episode_path
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ReplayHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    driver = _WebDriver(geckodriver)
    ffmpeg_process: subprocess.Popen[bytes] | None = None
    try:
        driver.start()
        url = (
            f"http://127.0.0.1:{server.server_port}/superdex_scenarios/rendering/pbr/"
            f"?recording=/__episode__.json&export=1&camera={camera_name}"
        )
        driver.navigate(url)
        _wait_for_viewer(driver)
        configured = driver.execute(
            "return window.superdexExport.configure(arguments[0], arguments[1], arguments[2]);",
            width,
            height,
            camera_name,
        )
        assert isinstance(configured, dict)

        command = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "image2pipe",
            "-framerate",
            f"{fps:g}",
            "-vcodec",
            "png",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        ffmpeg_process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        assert ffmpeg_process.stdin is not None
        print(
            f"Rendering {camera_name} video: {frame_count} frames, "
            f"{width}x{height} at {fps:g} FPS"
        )
        for frame_index in range(frame_count):
            seconds = min(frame_index / fps, replay_duration)
            data_url = driver.execute(
                "return window.superdexExport.renderAt(arguments[0]);", seconds
            )
            if not isinstance(data_url, str) or not data_url.startswith(
                "data:image/png;base64,"
            ):
                raise RuntimeError("PBR renderer returned an invalid PNG frame.")
            ffmpeg_process.stdin.write(base64.b64decode(data_url.split(",", 1)[1]))
            if frame_index % max(1, round(fps)) == 0:
                print(
                    f"  frame {frame_index + 1:4d}/{frame_count} "
                    f"({seconds:5.2f}/{replay_duration:5.2f} s)"
                )
        ffmpeg_process.stdin.close()
        return_code = ffmpeg_process.wait()
        if return_code != 0:
            assert ffmpeg_process.stderr is not None
            error = ffmpeg_process.stderr.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"ffmpeg failed with code {return_code}: {error}")

        metadata = {
            "format": "superdex-camera-video-v2",
            "video": output_path.name,
            "source_episode": str(episode_path),
            "scenario": replay.get("scenario"),
            "camera": camera_metadata,
            "renderer": "Three.js PBR in headless Firefox",
            "codec": "H.264/yuv420p",
            "crf": crf,
            "width_px": width,
            "height_px": height,
            "fps": fps,
            "frames": frame_count,
            "duration_s": replay_duration,
            "viewer": configured,
        }
        output_path.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Video exported: {output_path}")
    finally:
        if ffmpeg_process is not None and ffmpeg_process.poll() is None:
            ffmpeg_process.kill()
        driver.close()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5.0)


def main() -> None:
    args = _parse_args()
    output = (
        args.output
        if args.output is not None
        else REPOSITORY_ROOT
        / "scenarios"
        / "ball_bowl"
        / "exports"
        / "latest"
        / f"{args.camera}.mp4"
    )
    export_video(args.episode, output, args.fps, args.crf, args.camera)


if __name__ == "__main__":
    main()
