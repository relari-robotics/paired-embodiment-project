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

"""Embodiment-neutral simulator and CLI for the ball-and-bowl scenario.

The runner builds the scene for the selected embodiment, asks that
embodiment's :class:`~scenarios.ball_bowl.episode.EpisodePolicy` to plan, and
then executes the episode through one shared pipeline: compliant pose
controller, physics stepping, optional viewer, transform/telemetry/phase
recording, the success check, and export.  Nothing here knows whether it is
driving a two-jaw gripper or a 27-DOF hand.

The bowl is a dynamic body unless ``--fixed`` is given: that flag runs the
original regression scene, whose bowl is bolted to the desk.  ``--move-bowl``
(or a layout with a ``bowl_target_xy``) selects the variant in which the bowl
has to be moved to a target first; success then also requires the bowl to rest
upright there, so the bowl stays movable even with ``--fixed``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import secrets
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
from superdex import physics, robotics

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from superdex_scenarios.recording import PhaseLog, TelemetryRecorder, TransformRecorder
from superdex_scenarios.replay.cli import (
    add_replay_arguments,
    add_route_curves,
    load_layout,
    make_replay_policy,
    park_objects,
    split_routes,
    validate_replay_arguments,
    write_result,
)
from superdex_scenarios.simulation import (
    KinematicExecutor,
    PoseExecutor,
    create_pose_controller,
)

if __package__:
    from . import scenario as task
    from .cameras import camera_payloads
    from .embodiments import DEFAULT_EMBODIMENT, EMBODIMENTS
    from .episode import EpisodePolicy, PolicyOptions, load_policy_class
else:
    from scenarios.ball_bowl import scenario as task
    from scenarios.ball_bowl.cameras import camera_payloads
    from scenarios.ball_bowl.embodiments import DEFAULT_EMBODIMENT, EMBODIMENTS
    from scenarios.ball_bowl.episode import (
        EpisodePolicy,
        PolicyOptions,
        load_policy_class,
    )

NP_REAL = np.float64 if physics.uses_double_precision() else np.float32


def _xyzw(rotation: physics.Quaternion) -> npt.NDArray[np.float64]:
    return np.array([rotation[0], rotation[1], rotation[2], rotation[3]], dtype=float)


def make_policy(
    scenario: task.BallBowlScenario, options: PolicyOptions
) -> EpisodePolicy:
    """Instantiate the registered policy for the scenario's embodiment and task variant."""
    policy_class = load_policy_class(
        scenario.embodiment.policy_for(scenario.specification)
    )
    return policy_class(scenario, options)


class EpisodeRunner:
    """Shared execution helpers a policy uses to drive one episode.

    Wraps the :class:`PoseExecutor` with hand-pose blending, the semantic
    phase log, and the task-level checks (grasp alignment, physical lift).
    """

    def __init__(
        self,
        scenario: task.BallBowlScenario,
        controller: robotics.ControllerBase | None,
        controller_target: robotics.ControllerMochiArticulatedPoseTarget | None,
        viewer: object | None,
        real_time: bool = False,
        allow_failed_grasp: bool = False,
        frame_callback=None,
        step_callback=None,
        phase_sequence: tuple[str, ...] | None = None,
        kinematic: bool = False,
    ) -> None:
        self.scenario = scenario
        self.scene = scenario.scene
        self.info = scenario.bot_info
        self.workcell = scenario.workcell
        self.allow_failed_grasp = allow_failed_grasp
        self.objects_parked = False
        if kinematic:
            self.executor: PoseExecutor = KinematicExecutor(
                self.scene,
                self.info.actor,
                task.TIME_STEP,
                task.RENDER_EVERY_STEPS,
                viewer=viewer,
                real_time=real_time,
                frame_callback=frame_callback,
                step_callback=step_callback,
            )
        else:
            assert controller is not None and controller_target is not None
            self.executor = PoseExecutor(
                self.scene,
                controller,
                controller_target,
                task.TIME_STEP,
                task.RENDER_EVERY_STEPS,
                viewer=viewer,
                real_time=real_time,
                frame_callback=frame_callback,
                step_callback=step_callback,
            )
        self.ee = self.info.link_actor(self.scene, f"/{self.info.end_effector_link}")
        self.phases = PhaseLog(
            self.info.embodiment_id,
            self._phase_sample,
            phase_sequence or PhaseLog.phase_sequence,
        )

    # -- state queries ----------------------------------------------------

    def ball_position(self) -> npt.NDArray[np.float64]:
        return np.asarray(
            self.workcell.ball.get_root_transform().translation, dtype=float
        )

    def bowl_position(self) -> npt.NDArray[np.float64]:
        return np.asarray(
            self.workcell.bowl.get_root_transform().translation, dtype=float
        )

    def bowl_quaternion(self) -> npt.NDArray[np.float64]:
        return _xyzw(self.workcell.bowl.get_root_transform().rotation)

    def grasp_point_pose(
        self,
        link: str | None = None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """World position and (x, y, z, w) quaternion of the embodiment's grasp point.

        ``link`` names another end-effector link carrying the same local grasp
        point (the second gripper of a two-armed embodiment).
        """
        ee = self.ee if link is None else self.info.link_actor(self.scene, f"/{link}")
        transform = ee.get_root_transform() * physics.TransformRT(
            translation=self.info.grasp_point_local
        )
        return np.asarray(transform.translation, dtype=float), _xyzw(transform.rotation)

    def measured_pose(self) -> npt.NDArray[np.float64]:
        pose = physics.DynamicArrayReal(self.info.actor.get_num_dofs())
        self.info.actor.get_articulated_pose(pose)
        return np.asarray(pose, dtype=float)

    def _phase_sample(self):
        position, quaternion = self.grasp_point_pose()
        step = self.executor.step_count
        return step, step * task.TIME_STEP, position, quaternion, self.ball_position()

    # -- phase bookkeeping -------------------------------------------------

    def phase(self, name: str) -> None:
        """Mark the start of a semantic task phase (see ``episode.TASK_PHASES``)."""
        self.phases.begin(name)
        if os.environ.get("SUPERDEX_PHASE_DEBUG"):
            sample = self.phases.records[-1].start
            print(
                f"  phase {name} @ {sample.time_s:.2f}s: grasp point "
                f"{np.round(sample.grasp_point_world_m, 3).tolist()}, ball "
                f"{np.round(sample.object_position_m, 3).tolist()}"
            )

    # -- motion helpers ----------------------------------------------------

    def _target_pose(
        self,
        arm_pose: npt.ArrayLike,
        hand: str | npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        return self.info.target_pose(arm_pose, hand)

    def follow(
        self,
        path: npt.NDArray[np.float64],
        duration: float,
        hand_start: str | npt.ArrayLike,
        hand_end: str | npt.ArrayLike | None = None,
    ) -> bool:
        """Follow an arm path while blending the hand from one pose to another."""
        if isinstance(hand_start, str):
            start = self.info.hand_poses[hand_start]
        else:
            start = np.asarray(hand_start, dtype=float)
        if hand_end is None:
            end = start
        elif isinstance(hand_end, str):
            end = self.info.hand_poses[hand_end]
        else:
            end = np.asarray(hand_end, dtype=float)
        fractions = np.linspace(0.0, 1.0, len(path))
        full_path = np.vstack(
            [
                self._target_pose(arm, (1.0 - fraction) * start + fraction * end)
                for arm, fraction in zip(path, fractions)
            ]
        )
        return self.executor.follow(full_path, duration)

    def hold(
        self, arm_pose: npt.ArrayLike, hand: str | npt.ArrayLike, duration: float
    ) -> bool:
        return self.executor.hold(self._target_pose(arm_pose, hand), duration)

    def hold_pose(self, pose: npt.ArrayLike, duration: float) -> bool:
        """Hold a full articulation pose (all DOFs)."""
        return self.executor.hold(pose, duration)

    # -- task checks -------------------------------------------------------

    def print_tracking_error(self, planned_arm: npt.ArrayLike) -> None:
        actual_arm = self.measured_pose()[self.info.arm_dofs]
        print(
            "  arm tracking error: "
            f"{np.round(actual_arm - np.asarray(planned_arm, dtype=float), 3).tolist()}"
        )

    def check_bowl_grasp_alignment(self, rim_point: npt.ArrayLike, link: str) -> None:
        """Refuse to pinch when the grasp point of ``link`` is not on the planned rim point."""
        grasp_position, _ = self.grasp_point_pose(link)
        # The bowl may have been nudged; measure against where its rim actually is.
        shift = self.bowl_position() - self.scenario.specification.bowl_start
        actual_rim = np.asarray(rim_point, dtype=float) + [shift[0], shift[1], 0.0]
        error = float(np.linalg.norm(actual_rim - grasp_position))
        print(
            "  bowl grasp: rim/grasp point = "
            f"{np.round(actual_rim, 3).tolist()} / "
            f"{np.round(grasp_position, 3).tolist()}"
        )
        if error > 0.03:
            message = f"Grasp point missed the bowl rim by {error:.3f} m."
            if not self.allow_failed_grasp:
                raise RuntimeError(message + " Refusing to pinch.")
            print(f"  WARNING: {message} Continuing diagnostic episode.")

    def verify_bowl_moved(self) -> None:
        """Prove that contact alone brought the bowl to its target region."""
        bowl = self.bowl_position()
        quaternion = self.bowl_quaternion()
        at_target = self.scenario.specification.bowl_at_target(bowl, quaternion)
        print(
            f"  bowl moved to {np.round(bowl, 3).tolist()} (tilt "
            f"{task.bowl_tilt_deg(quaternion):.1f} deg), at_target={at_target}"
        )
        if not at_target:
            message = "Contact-only drag did not bring the bowl to its target."
            if not self.allow_failed_grasp:
                raise RuntimeError(message)
            print(f"  WARNING: {message} Continuing diagnostic episode.")
            return
        self.phases.event("bowl_at_target")

    def check_grasp_alignment(self) -> None:
        """Refuse to close when the grasp point is not on the ball."""
        ball_position = self.ball_position()
        grasp_position, _ = self.grasp_point_pose()
        print(
            "  grasp: ball/grasp point = "
            f"{np.round(ball_position, 3).tolist()} / "
            f"{np.round(grasp_position, 3).tolist()}"
        )
        grasp_error = float(np.linalg.norm(ball_position - grasp_position))
        if self.info.grasp_tolerance_m is None:
            raise ValueError("The embodiment must define grasp_tolerance_m.")
        if grasp_error > self.info.grasp_tolerance_m:
            raise RuntimeError(
                f"Grasp point missed the ball by {grasp_error:.3f} m; refusing to close."
            )

    def verify_physical_grasp(self) -> None:
        """Prove that contact alone lifted the ball off the desk."""
        ball_position = self.ball_position()
        minimum_lifted_z = task.DESK_TOP_Z + task.BALL_RADIUS + 0.055
        if ball_position[2] < minimum_lifted_z:
            message = f"Contact-only grasp failed to lift the ball; ball z={ball_position[2]:.3f} m."
            if not self.allow_failed_grasp:
                raise RuntimeError(message)
            print(f"  WARNING: {message} Continuing diagnostic episode.")
            return
        print(
            f"  physical grasp verified: ball at {np.round(ball_position, 3).tolist()}"
        )
        self.phases.event("grasp_verified")

    def print_release_position(self) -> None:
        print(
            f"  physical release begins: ball at {np.round(self.ball_position(), 3).tolist()}"
        )

    # -- episode control ---------------------------------------------------

    def reset_episode(self, home_pose: npt.ArrayLike) -> None:
        """Restore embodiment, ball, and bowl state for an exact debugger replay."""
        self.info.actor.set_articulated_pose_from_joints(
            np.asarray(home_pose, dtype=NP_REAL)
        )
        self.info.actor.set_articulated_joint_velocities(
            np.zeros(self.info.actor.get_num_dofs(), dtype=NP_REAL)
        )
        if self.objects_parked:
            return
        self.workcell.ball.set_root_transform(
            physics.TransformRT(translation=self.scenario.specification.ball_start)
        )
        self.workcell.ball.set_velocity(
            linear_vel=[0.0, 0.0, 0.0], angular_vel=[0.0, 0.0, 0.0]
        )
        if self.scenario.specification.bowl_static:
            return
        self.workcell.bowl.set_root_transform(
            physics.TransformRT(translation=self.scenario.specification.bowl_start)
        )
        self.workcell.bowl.set_velocity(
            linear_vel=[0.0, 0.0, 0.0], angular_vel=[0.0, 0.0, 0.0]
        )

    def park_objects(self) -> None:
        """Move the ball and bowl off the desk for an object-free run."""
        park_objects([self.workcell.ball, self.workcell.bowl])
        self.objects_parked = True

    def run(self, policy: EpisodePolicy) -> bool:
        completed = policy.run(self)
        self.phases.finish_episode(completed)
        return completed

    def close(self) -> None:
        pass


DEFAULT_LOOK_FROM = (0.95, -1.15, 0.92)
DEFAULT_LOOK_AT = (-0.02, 0.0, 0.36)
CALIBRATED_CAMERAS = ("desk_gemini_335", "wrist_right", "wrist_left")
"""Cameras embedded in every replay; usable as --camera with the PBR video renderer."""
PBR_FREE_CAMERA_FOV_DEG = 40.0
CAMERA_PRESETS: dict[
    str, tuple[tuple[float, float, float], tuple[float, float, float]]
] = {
    "workcell": (DEFAULT_LOOK_FROM, DEFAULT_LOOK_AT),
    "desk": ((0.55, -0.75, 0.72), (-0.02, -0.06, 0.42)),
    "right": ((0.48, -0.76, 0.68), (0.0, -0.16, 0.43)),
    "left": ((0.48, 0.76, 0.68), (0.0, 0.16, 0.43)),
    "front": ((0.95, 0.0, 0.75), (-0.05, 0.0, 0.40)),
    # Over the robot's shoulders, looking down at the desk: left and right match
    # a teleoperator's own hands.
    "first_person": ((-0.62, 0.0, 1.15), (0.02, 0.0, 0.40)),
}
CAMERA_VIEW_LIMITS_COLOR = (1.0, 0.55, 0.0)
TELEOP_WORKSPACE_COLORS = {"right": (0.31, 0.86, 0.31), "left": (0.35, 0.67, 1.0)}


def box_wireframe(low, high) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int_]]:
    """Corner nodes and the twelve edges of an axis-aligned box."""

    low, high = np.asarray(low, dtype=float), np.asarray(high, dtype=float)
    nodes = np.array(
        [
            [(low, high)[(index >> axis) & 1][axis] for axis in range(3)]
            for index in range(8)
        ]
    )
    edges = [
        (a, a | (1 << axis)) for a in range(8) for axis in range(3) if not a & (1 << axis)
    ]
    return nodes, np.asarray(edges, dtype=int)


def create_viewer(
    scenario: task.BallBowlScenario,
    points: npt.NDArray[np.float64],
    *,
    show_trajectory: bool = True,
    look_from: tuple[float, float, float] = DEFAULT_LOOK_FROM,
    look_at: tuple[float, float, float] = DEFAULT_LOOK_AT,
    size: tuple[int, int] | None = None,
    offscreen: bool = False,
):
    from superdex.physics.utils.coordinate_systems import CoordinateSystem
    from superdex.physics.viewer import Viewer, ViewerCfg

    viewer = Viewer(
        ViewerCfg(
            coordinate_system=CoordinateSystem(right="-Y", up="+Z", forward="+X"),
            start_paused=False,
            size=size,
            offscreen=offscreen,
        )
    )
    viewer.set_scene(scenario.scene)
    hidden_links = scenario.bot_info.hidden_render_link_names
    if hidden_links:
        viewer.set_excluded_actors(
            [f"*{link_name}" for link_name in sorted(hidden_links)]
        )
    viewer.set_camera_view(look_from=list(look_from), look_at=list(look_at))
    if show_trajectory and len(points) > 1:
        add_route_curves(viewer, points)
    viewer.render()
    for actor, color in (
        *((actor, task.WOOD_COLOR) for actor in scenario.workcell.desk_actors),
        (scenario.workcell.bowl, scenario.specification.bowl_color.rgb),
        (scenario.workcell.ball, scenario.specification.ball_color.rgb),
    ):
        renderer = viewer.get_actor_renderer(actor)
        if renderer is not None and hasattr(renderer, "set_front_face_color"):
            renderer.set_front_face_color(color)
    return viewer


class VideoWriter:
    """Encode offscreen viewer frames to an MP4 through ffmpeg as the episode plays.

    Wraps the viewer so :class:`PoseExecutor` keeps calling ``render()`` and
    ``user_requested_close()`` unchanged; every ``stride``-th rendered frame is
    piped to ffmpeg as raw RGBA.
    """

    def __init__(
        self, viewer, path: Path, *, fps: float, stride: int, size: tuple[int, int]
    ) -> None:
        self.viewer = viewer
        self.path = path
        self.stride = max(1, stride)
        self.count = 0
        self.written = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        self._process = subprocess.Popen(
            [
                _ffmpeg_executable(),
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgba",
                "-s",
                f"{size[0]}x{size[1]}",
                "-r",
                f"{fps:.4f}",
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "18",
                "-movflags",
                "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )

    def render(self):
        frame = self.viewer.render()
        self.count += 1
        if frame is not None and (self.count - 1) % self.stride == 0:
            assert self._process.stdin is not None
            self._process.stdin.write(
                np.ascontiguousarray(frame, dtype=np.uint8).tobytes()
            )
            self.written += 1
        return frame

    def user_requested_close(self) -> bool:
        return False

    def close(self) -> None:
        if self._process.stdin is not None:
            self._process.stdin.close()
        self._process.wait()
        self.viewer.close()
        print(f"Video written: {self.path} ({self.written} frames)")

    def __getattr__(self, name: str):
        return getattr(self.viewer, name)


def _ffmpeg_executable() -> str:
    import shutil

    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as error:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "ffmpeg is required for --video; install it or imageio-ffmpeg."
        ) from error


def default_headless_export_dir(seed: int | None) -> Path:
    """Return a unique local output directory for one automatic headless run."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    task_id = "fixed" if seed is None else f"seed_{seed}"
    return (
        Path(__file__).resolve().parent / "exports" / "runs" / f"{timestamp}_{task_id}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embodiment",
        choices=sorted(EMBODIMENTS),
        default=None,
        help=f"embodiment to run (default: {DEFAULT_EMBODIMENT})",
    )
    parser.add_argument(
        "--human",
        action="store_true",
        help="shorthand for --embodiment human_right_hand (an earlier project's scaffold)",
    )
    randomization = parser.add_mutually_exclusive_group()
    randomization.add_argument(
        "--seed",
        type=lambda value: int(value, 0),
        help=(
            "reproduce a randomized scenario with this integer seed; if omitted, "
            "a new seed is generated"
        ),
    )
    randomization.add_argument(
        "--fixed",
        action="store_true",
        help=(
            "run the original fixed blue-ball/gray-bowl regression scenario, with "
            "the bowl bolted to the desk (it is movable in every other run, and "
            "with --move-bowl)"
        ),
    )
    parser.add_argument(
        "--move-bowl",
        action="store_true",
        help=(
            "bimanual variant: the bowl starts on the robot's left and has to be "
            "moved to a target before the ball goes in (a --layout with "
            "bowl_target_xy selects it too)"
        ),
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help=(
            "run without a viewer and write a full bundle to a unique exports/runs "
            "directory unless another output mode is selected"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run one complete headless episode and validation without writing files",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="build the scene, run the embodiment's planner, validate it, then exit",
    )
    parser.add_argument(
        "--no-trajopt",
        action="store_true",
        help=(
            "ask the policy for direct Cartesian references instead of trajectory "
            "optimization; useful for isolating optimizer failures"
        ),
    )
    parser.add_argument(
        "--allow-failed-grasp",
        action="store_true",
        help=(
            "complete and export the entire motion even if contact validation or "
            "the physical lift fails"
        ),
    )
    parser.add_argument(
        "--teleop",
        action="store_true",
        help="drive both OpenArm arms from live Kyber MediaPipe packets",
    )
    parser.add_argument(
        "--teleop-endpoint",
        default="127.0.0.1:7447",
        help="UDP bind endpoint for --teleop (default: 127.0.0.1:7447)",
    )
    parser.add_argument(
        "--teleop-mapping",
        type=Path,
        default=Path(__file__).resolve().parent
        / "embodiments/openarm_v2/teleop_mapping.json",
        help="per-arm desk-to-workspace mapping JSON",
    )
    parser.add_argument(
        "--teleop-duration",
        type=float,
        default=0.0,
        help="teleoperation duration in seconds; zero runs until stopped",
    )
    parser.add_argument(
        "--snapshot",
        nargs="?",
        const=str(Path(__file__).resolve().parent / "exports" / "snapshot.png"),
        metavar="PATH",
        help=(
            "render the planned pre-grasp pose in the simulator and save one PNG "
            "(default: exports/snapshot.png)"
        ),
    )
    parser.add_argument(
        "--debugger",
        action="store_true",
        help=(
            "open the live Mochi debugger and execute the episode at wall-clock "
            "speed using its Filament renderer"
        ),
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="replay successful episodes while the live debugger remains connected",
    )
    parser.add_argument(
        "--record-pbr",
        nargs="?",
        const=str(Path(__file__).resolve().parent / "exports" / "replay.json"),
        metavar="PATH",
        help=(
            "run one headless physics episode and record actor transforms for "
            "the external PBR viewer (default: exports/replay.json)"
        ),
    )
    parser.add_argument(
        "--export-dir",
        nargs="?",
        const=str(Path(__file__).resolve().parent / "exports" / "latest"),
        metavar="PATH",
        help=(
            "run one headless episode and export all embodiment cameras, "
            "400 Hz motor/contact telemetry, individual contacts, transforms, "
            "and the task phase log (default: exports/latest)"
        ),
    )
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="with --headless or --export-dir, write physics data but skip MP4s",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=30.0,
        help="exported MP4 frame rate (default: 30)",
    )
    parser.add_argument(
        "--video-crf",
        type=int,
        default=18,
        help="exported H.264 quality, 0 best to 51 worst (default: 18)",
    )
    parser.add_argument(
        "--video",
        metavar="PATH",
        help=(
            "render the episode to an MP4 without a window: photorealistic PBR by "
            "default (--video-renderer pbr), or the plain SuperDex viewer "
            "(--video-renderer viewer); combine with --camera"
        ),
    )
    parser.add_argument(
        "--video-renderer",
        choices=("pbr", "viewer"),
        default="pbr",
        help=(
            "pbr: Three.js physically based renderer with the robot's render meshes "
            "(needs Firefox, geckodriver, and Node); viewer: the SuperDex viewer used "
            "by the interactive window (default: pbr)"
        ),
    )
    parser.add_argument(
        "--video-size",
        default="1920x1080",
        metavar="WxH",
        help="offscreen render size for --video (default: 1920x1080)",
    )
    parser.add_argument(
        "--camera",
        choices=sorted(CAMERA_PRESETS) + list(CALIBRATED_CAMERAS),
        default="workcell",
        help=(
            "viewpoint for the interactive window and --video: a preset "
            f"({', '.join(sorted(CAMERA_PRESETS))}) or, for the PBR video only, one of "
            "the scene's calibrated cameras (desk_gemini_335, wrist_right, wrist_left); "
            "default: workcell"
        ),
    )
    parser.add_argument(
        "--no-trajectory",
        action="store_true",
        help="hide the planned grasp-point trajectory curve in the viewer",
    )
    add_replay_arguments(parser)
    args = parser.parse_args()
    if args.human and args.embodiment not in (None, "human_right_hand"):
        parser.error("--human conflicts with --embodiment")
    args.embodiment = (
        "human_right_hand"
        if args.human
        else (args.embodiment or DEFAULT_EMBODIMENT)
    )
    return args


def validate_args(args: argparse.Namespace) -> bool:
    """Reject incompatible flag combinations; return whether headless export is automatic."""
    validate_replay_arguments(args)
    if (
        args.move_bowl
        and EMBODIMENTS[args.embodiment].move_bowl_policy is None
        and args.replay is None
    ):
        raise SystemExit(
            f"--move-bowl needs a two-armed embodiment; {args.embodiment!r} has no "
            "policy that moves the bowl"
        )
    if args.teleop:
        if args.embodiment != "openarm_v2_bimanual":
            raise SystemExit("--teleop requires --embodiment openarm_v2_bimanual")
        if args.teleop_duration < 0:
            raise SystemExit("--teleop-duration must be non-negative")
        if (
            args.replay is not None
            or args.plan_only
            or args.snapshot is not None
            or args.kinematic
        ):
            raise SystemExit(
                "--teleop cannot be combined with replay, --plan-only, --snapshot, or --kinematic"
            )
        if args.teleop_duration == 0 and (
            args.headless
            or args.dry_run
            or args.record_pbr is not None
            or args.export_dir is not None
            or args.video is not None
        ):
            raise SystemExit(
                "non-interactive --teleop runs require a positive --teleop-duration"
            )
    if args.camera in CALIBRATED_CAMERAS and not (
        args.video is not None and args.video_renderer == "pbr"
    ):
        raise SystemExit(
            f"--camera {args.camera} is a calibrated scene camera; it needs --video with "
            "the pbr renderer. Use a preset (workcell, desk, right, left, front) otherwise."
        )
    if args.video is not None and args.video_renderer == "pbr":
        import shutil

        if shutil.which("geckodriver") is None:
            raise SystemExit(
                "The photorealistic --video renderer drives headless Firefox through "
                "geckodriver, which is not on PATH. Install Firefox and geckodriver "
                "(macOS: brew install geckodriver), or pass --video-renderer viewer."
            )
    if args.video is not None and (
        args.headless
        or args.dry_run
        or args.plan_only
        or args.debugger
        or args.loop
        or args.snapshot is not None
    ):
        raise SystemExit(
            "--video cannot be combined with --headless, --dry-run, --plan-only, "
            "--debugger, --loop, or --snapshot"
        )
    if args.loop and not args.debugger:
        raise SystemExit("--loop requires --debugger")
    if args.headless and (args.debugger or args.loop):
        raise SystemExit("--headless cannot be combined with --debugger or --loop")
    if args.dry_run and (
        args.debugger
        or args.loop
        or args.record_pbr is not None
        or args.export_dir is not None
        or args.skip_video
    ):
        raise SystemExit(
            "--dry-run cannot be combined with --debugger, --loop, --record-pbr, "
            "--export-dir, or --skip-video"
        )
    if args.plan_only and (
        args.debugger
        or args.loop
        or args.record_pbr is not None
        or args.export_dir is not None
        or args.skip_video
    ):
        raise SystemExit(
            "--plan-only cannot be combined with simulation or export options"
        )
    if args.snapshot is not None and (
        args.headless
        or args.dry_run
        or args.plan_only
        or args.debugger
        or args.loop
        or args.record_pbr is not None
        or args.export_dir is not None
        or args.skip_video
    ):
        raise SystemExit(
            "--snapshot cannot be combined with simulation or export options"
        )
    if args.export_dir is not None and (
        args.debugger or args.loop or args.record_pbr is not None
    ):
        raise SystemExit(
            "--export-dir cannot be combined with --debugger, --loop, or --record-pbr"
        )
    automatic_headless_export = (
        args.headless
        and not args.dry_run
        and not args.plan_only
        and not args.debugger
        and args.record_pbr is None
        and args.export_dir is None
    )
    if args.skip_video and args.export_dir is None and not automatic_headless_export:
        raise SystemExit("--skip-video requires --export-dir or --headless")
    return automatic_headless_export


def build_and_plan(
    context: robotics.RoboticsContext,
    args: argparse.Namespace,
    seed: int | None,
    options: PolicyOptions,
) -> tuple[task.BallBowlScenario, EpisodePolicy]:
    """Build the scene for the selected embodiment and run its planner."""

    def planner(scenario: task.BallBowlScenario) -> EpisodePolicy:
        if args.teleop:
            from scenarios.ball_bowl.embodiments.openarm_v2.teleop_policy import (
                TeleopPolicy,
            )

            policy: EpisodePolicy = TeleopPolicy(
                scenario,
                options,
                endpoint=args.teleop_endpoint,
                mapping_path=args.teleop_mapping,
                duration_s=args.teleop_duration,
            )
        elif args.replay is not None:
            policy: EpisodePolicy = make_replay_policy(scenario, options, args)
        else:
            policy = make_policy(scenario, options)
        policy.plan()
        return policy

    layout = load_layout(args.layout)
    if args.fixed or layout is not None:
        specification = (
            task.ScenarioSpecification.from_layout(layout)
            if layout is not None
            else task.ScenarioSpecification.fixed(move_bowl=args.move_bowl)
        )
        # Only --fixed bolts the bowl down, and never a bowl that has to be moved.
        if args.fixed and specification.bowl_target_xy is None:
            specification = dataclasses.replace(specification, bowl_static=True)
        scenario = task.BallBowlScenario.build(context, args.embodiment, specification)
        print(
            "Scenario: "
            + (
                "layout file " + args.layout
                if layout is not None
                else "fixed regression configuration"
            )
            + f"; bowl: {'static' if specification.bowl_static else 'movable'}"
            + f"; embodiment: {scenario.embodiment_id}"
        )
        try:
            return scenario, planner(scenario)
        except Exception:
            scenario.close()
            raise
    assert seed is not None
    print(f"Scenario random seed: {seed}")
    return task.BallBowlScenario.build_randomized(
        context, seed, args.embodiment, planner=planner, move_bowl=args.move_bowl
    )


def main() -> None:
    args = parse_args()
    automatic_headless_export = validate_args(args)
    selected_seed = (
        None
        if args.fixed or args.layout is not None
        else (args.seed if args.seed is not None else secrets.randbits(63))
    )
    if args.export_dir is not None:
        export_dir = Path(args.export_dir).expanduser().resolve()
    elif automatic_headless_export:
        export_dir = default_headless_export_dir(selected_seed).resolve()
        print(f"Headless export directory: {export_dir}")
    else:
        export_dir = None
    export_episode_path = export_dir / "episode.json" if export_dir else None
    export_video_paths: dict[str, Path] | None = None
    export_ready = False
    video_episode_path: Path | None = None
    options = PolicyOptions(
        optimize_trajectory=not args.no_trajopt,
        allow_failed_grasp=args.allow_failed_grasp,
    )

    physics.initialize(num_worker_threads=0)
    replay_failed = False
    scenario: task.BallBowlScenario | None = None
    viewer = None
    runner: EpisodeRunner | None = None
    recorder: TransformRecorder | None = None
    telemetry: TelemetryRecorder | None = None
    try:
        if args.debugger:
            # Attach before constructing the scene so the live debugger receives
            # every scene/actor creation event instead of relying on a late snapshot.
            # The task is authored X-forward/Y-left/Z-up (FLU); the debug server's
            # standalone default is OpenGL Y-up, so declare the simulation convention
            # before starting the server.
            debug_server = physics.get_debug_server()
            if not debug_server.has_started():
                debug_server.set_coordinate_space(
                    physics.CoordinateSpace(
                        axes=physics.CoordinateSpaceAxes.FLU,
                        units_per_meter=1.0,
                    )
                )
            if not physics.debugger.attach(timeout_seconds=10.0):
                raise RuntimeError("Could not attach the Mochi physics debugger.")
        context = robotics.create_context()
        try:
            scenario, policy = build_and_plan(context, args, selected_seed, options)
        except NotImplementedError as error:
            # The blank project policy: report the scaffold state and stop.
            if args.plan_only:
                print(
                    f"Embodiment {args.embodiment!r} scene built; policy unimplemented: {error}"
                )
                return
            raise
        task_payload = scenario.specification.to_dict()
        cameras = camera_payloads(scenario.bot_info)
        scenario_payload = {"embodiment": scenario.embodiment_id, **task_payload}
        print(json.dumps(scenario_payload, indent=2))

        points = policy.trajectory_points()
        if args.plan_only:
            routes = split_routes(points)
            print(
                f"Plan valid: {sum(len(r) for r in routes)} displayed trajectory points "
                f"in {len(routes)} route(s)."
            )
            return
        if args.snapshot is not None:
            scenario.bot_info.actor.set_articulated_pose_from_joints(
                np.asarray(policy.preshape_pose(), dtype=NP_REAL)
            )
            viewer = create_viewer(scenario, points)
            viewer.set_camera_view(
                look_from=[0.48, -0.76, 0.68],
                look_at=[0.0, -0.16, 0.43],
            )
            viewer.render()
            import polyscope as ps

            snapshot_path = Path(args.snapshot).expanduser().resolve()
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            ps.screenshot(str(snapshot_path), transparent_bg=False, include_UI=False)
            print(f"Simulator snapshot: {snapshot_path}")
            return

        if export_dir is not None:
            export_video_paths = {
                str(camera["name"]): export_dir / f"{camera['name']}.mp4"
                for camera in cameras
            }
        if args.kinematic:
            controller, controller_target = None, None
        else:
            controller, controller_target = create_pose_controller(scenario.bot_info)
        interactive = (
            not args.headless
            and not args.dry_run
            and not args.debugger
            and args.record_pbr is None
            and export_dir is None
            and args.video is None
        )
        pbr_video = args.video is not None and args.video_renderer == "pbr"
        if interactive or (args.video is not None and not pbr_video):
            look_from, look_at = CAMERA_PRESETS[args.camera]
            video_size = None
            if args.video is not None:
                width, height = (int(v) for v in args.video_size.lower().split("x"))
                video_size = (width, height)
            viewer = create_viewer(
                scenario,
                points,
                show_trajectory=not args.no_trajectory,
                look_from=look_from,
                look_at=look_at,
                size=video_size,
                offscreen=args.video is not None,
            )
            if args.teleop:
                # Where the real desk camera can see a hand: from the desk top to
                # the top of the teleop workspace.
                from scenarios.ball_bowl.cameras import desk_camera_view_limits

                mapping = json.loads(Path(args.teleop_mapping).read_text(encoding="utf-8"))
                top = max(arm["target_max_world_m"][2] for arm in mapping["arms"].values())
                nodes, edges = desk_camera_view_limits(task.DESK_TOP_Z + 0.002, top)
                viewer.add_curve_network(
                    "camera_view_limits",
                    nodes=nodes,
                    edges=edges,
                    radius=0.002,
                    color=np.asarray(CAMERA_VIEW_LIMITS_COLOR),
                )
                # Where each gripper can follow: outside its box the target is
                # clipped, even while the camera still sees the hand.
                for side, arm in mapping["arms"].items():
                    nodes, edges = box_wireframe(
                        arm["target_min_world_m"], arm["target_max_world_m"]
                    )
                    viewer.add_curve_network(
                        f"teleop_workspace_{side}",
                        nodes=nodes,
                        edges=edges,
                        radius=0.0012,
                        color=np.asarray(TELEOP_WORKSPACE_COLORS[side]),
                    )
            if args.video is not None:
                render_fps = 1.0 / (task.TIME_STEP * task.RENDER_EVERY_STEPS)
                stride = max(1, round(render_fps / args.video_fps))
                viewer = VideoWriter(
                    viewer,
                    Path(args.video).expanduser().resolve(),
                    fps=render_fps / stride,
                    stride=stride,
                    size=video_size,
                )

        video_episode_path = (
            Path(args.video).expanduser().resolve().with_suffix(".episode.json")
            if pbr_video
            else None
        )
        if args.record_pbr is not None or export_dir is not None or pbr_video:
            recorder = TransformRecorder(
                scenario.scene,
                time_step=task.TIME_STEP,
                render_every_steps=task.RENDER_EVERY_STEPS,
                task=task_payload,
                embodiment=scenario.embodiment_id,
                cameras=cameras,
                render_manifest=scenario.embodiment.render_manifest,
                render_overrides={
                    "blue_ball": {
                        "scale": [task.BALL_RADIUS] * 3,
                        "color": list(scenario.specification.ball_color.rgb),
                    },
                    "gray_bowl": {
                        "scale": scenario.specification.bowl_scale.tolist(),
                        "color": list(scenario.specification.bowl_color.rgb),
                    },
                },
            )
            recorder.capture()
        if export_dir is not None:
            export_dir.mkdir(parents=True, exist_ok=True)
            (export_dir / "scenario.json").write_text(
                json.dumps(scenario_payload, indent=2) + "\n",
                encoding="utf-8",
            )
            if not args.kinematic:
                telemetry = TelemetryRecorder(
                    scenario,
                    export_dir,
                    time_step=task.TIME_STEP,
                    cameras=cameras,
                )

        runner = EpisodeRunner(
            scenario,
            controller,
            controller_target,
            viewer,
            # Interactive playback should be watchable. Headless/export modes
            # remain unthrottled, while either live viewer tracks wall time.
            real_time=args.teleop
            or args.debugger
            or (viewer is not None and args.video is None),
            allow_failed_grasp=args.allow_failed_grasp,
            frame_callback=recorder.capture if recorder is not None else None,
            step_callback=telemetry.record_sample if telemetry is not None else None,
            phase_sequence=getattr(policy, "phase_sequence", None),
            kinematic=args.kinematic,
        )
        home_pose = policy.home_pose()
        # Every embodiment starts parked exactly where its policy says home is.
        runner.reset_episode(home_pose)
        if args.no_objects:
            runner.park_objects()
            print("Task objects parked away from the desk; success is not evaluated.")
        if args.debugger:
            # Give the debugger time to display and frame the connected scene before
            # the useful motion begins.
            runner.hold_pose(home_pose, 1.5)
        uncontrolled = np.ones(scenario.bot_info.actor.get_num_dofs(), dtype=bool)
        uncontrolled[scenario.bot_info.controlled_dofs] = False
        episode_number = 0
        while True:
            episode_number += 1
            completed = runner.run(policy)
            if args.debugger and completed:
                runner.hold_pose(home_pose, 1.0)
            final_ball = runner.ball_position()
            final_bowl = runner.bowl_position()
            final_bowl_quaternion = runner.bowl_quaternion()
            final_pose = runner.measured_pose()
            parked_drift = (
                float(
                    np.max(
                        np.abs(
                            final_pose[uncontrolled]
                            - scenario.bot_info.default_pose[uncontrolled]
                        )
                    )
                )
                if np.any(uncontrolled)
                else 0.0
            )
            spec = scenario.specification
            # The bowl is dynamic: judge the ball against where the bowl ended up.
            in_bowl = spec.contains_ball(final_ball, final_bowl)
            bowl_at_target: bool | None = (
                None
                if spec.bowl_target_xy is None
                else spec.bowl_at_target(final_bowl, final_bowl_quaternion)
            )
            task_done = in_bowl and bowl_at_target is not False
            print(
                f"Episode {episode_number} "
                f"{'complete' if completed else 'stopped'}; bowl at "
                f"{np.round(final_bowl, 3).tolist()} (tilt "
                f"{task.bowl_tilt_deg(final_bowl_quaternion):.1f} deg"
                + ("" if bowl_at_target is None else f", at_target={bowl_at_target}")
                + f"), ball at {np.round(final_ball, 3).tolist()}, in_bowl={in_bowl}, "
                f"parked-side drift={parked_drift:.2e} rad."
            )
            evaluated = not (args.no_objects or args.kinematic)
            success: bool | None = (
                None if args.teleop or not evaluated else bool(completed and task_done)
            )
            result_payload = {
                "task": "ball_bowl",
                "mode": (
                    "teleop"
                    if args.teleop
                    else ("replay" if args.replay is not None else "policy")
                ),
                "kinematic": bool(args.kinematic),
                "objects_present": not args.no_objects,
                "completed": bool(completed),
                "success": success,
                "in_bowl": bool(in_bowl) if evaluated else None,
                "bowl_at_target": bowl_at_target if evaluated else None,
                "final_ball_position_m": np.round(final_ball, 4).tolist(),
                "bowl_position_m": spec.bowl_start.tolist(),
                "final_bowl_position_m": np.round(final_bowl, 4).tolist(),
                "final_bowl_tilt_deg": round(
                    task.bowl_tilt_deg(final_bowl_quaternion), 2
                ),
                "bowl_target_m": None
                if spec.bowl_target_xy is None
                else [*spec.bowl_target_xy, task.DESK_TOP_Z],
                "parked_side_drift_rad": parked_drift,
                "simulated_time_s": runner.executor.step_count * task.TIME_STEP,
                "scenario": scenario_payload,
            }
            if hasattr(policy, "summary"):
                result_payload["replay"] = policy.summary()
            if args.replay is not None or args.result is not None:
                write_result(result_payload, export_dir, args.result)
            if recorder is not None:
                recorder.capture()
                if export_episode_path is not None:
                    recording_path = export_episode_path
                elif args.record_pbr is not None:
                    recording_path = Path(args.record_pbr).expanduser().resolve()
                else:
                    assert video_episode_path is not None
                    recording_path = video_episode_path
                recorder.save(recording_path)
            if export_dir is not None:
                runner.phases.save(export_dir / "phases.json")
            if (
                (args.headless or export_dir is not None)
                and completed
                and not task_done
                and not args.allow_failed_grasp
                and args.replay is None
                and not args.teleop
                and not args.no_objects
            ):
                raise RuntimeError(
                    "Headless episode completed without the ball in the bowl"
                    + ("." if bowl_at_target is None else " at its target.")
                )
            if export_dir is not None:
                export_ready = completed and (
                    task_done
                    or args.allow_failed_grasp
                    or args.replay is not None
                    or args.no_objects
                )
            if args.replay is not None and success is False:
                replay_failed = True
            if not (
                args.loop
                and completed
                and task_done
                and physics.debugger.is_attached()
            ):
                break
            runner.hold_pose(home_pose, 1.5)
            runner.reset_episode(home_pose)
            runner.hold_pose(home_pose, 1.0)
            print("Replaying episode...")
    finally:
        if runner is not None:
            runner.close()
        if viewer is not None:
            viewer.close()
        if telemetry is not None:
            telemetry.close()
        if scenario is not None:
            scenario.close()
        physics.shutdown()

    if export_ready and not args.skip_video:
        assert export_episode_path is not None
        assert export_video_paths is not None
        exporter = (
            REPOSITORY_ROOT
            / "superdex_scenarios"
            / "rendering"
            / "pbr"
            / "export_video.py"
        )
        for camera, video_path in export_video_paths.items():
            subprocess.run(
                [
                    sys.executable,
                    str(exporter),
                    "--episode",
                    str(export_episode_path),
                    "--camera",
                    camera,
                    "--output",
                    str(video_path),
                    "--fps",
                    str(args.video_fps),
                    "--crf",
                    str(args.video_crf),
                ],
                check=True,
            )
    if args.video is not None and args.video_renderer == "pbr":
        exporter = (
            REPOSITORY_ROOT
            / "superdex_scenarios"
            / "rendering"
            / "pbr"
            / "export_video.py"
        )
        episode_for_video = (
            export_episode_path
            if export_episode_path is not None
            else video_episode_path
        )
        assert episode_for_video is not None
        command = [
            sys.executable,
            str(exporter),
            "--episode",
            str(episode_for_video),
            "--output",
            str(Path(args.video).expanduser().resolve()),
            "--fps",
            str(args.video_fps),
            "--crf",
            str(args.video_crf),
        ]
        if args.camera in CALIBRATED_CAMERAS:
            command += ["--camera", args.camera]
        else:
            look_from, look_at = CAMERA_PRESETS[args.camera]
            command += [
                "--look-from",
                *map(str, look_from),
                "--look-at",
                *map(str, look_at),
                "--fov",
                str(PBR_FREE_CAMERA_FOV_DEG),
                "--size",
                args.video_size,
            ]
        subprocess.run(command, check=True)
    if replay_failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
