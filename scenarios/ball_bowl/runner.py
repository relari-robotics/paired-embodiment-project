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
recording, the in-bowl success check, and export.  Nothing here knows whether
it is driving a two-jaw gripper or a 27-DOF hand.
"""

from __future__ import annotations

import argparse
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
from superdex_scenarios.simulation import PoseExecutor, create_pose_controller

if __package__:
    from . import scenario as task
    from .cameras import camera_payloads
    from .episode import EpisodePolicy, PolicyOptions, load_policy_class
else:
    from scenarios.ball_bowl import scenario as task
    from scenarios.ball_bowl.cameras import camera_payloads
    from scenarios.ball_bowl.episode import EpisodePolicy, PolicyOptions, load_policy_class

NP_REAL = np.float64 if physics.uses_double_precision() else np.float32


def make_policy(scenario: task.BallBowlScenario, options: PolicyOptions) -> EpisodePolicy:
    """Instantiate the registered policy for the scenario's embodiment."""
    policy_class = load_policy_class(scenario.embodiment.policy)
    return policy_class(scenario, options)


class EpisodeRunner:
    """Shared execution helpers a policy uses to drive one episode.

    Wraps the :class:`PoseExecutor` with hand-pose blending, the semantic
    phase log, and the task-level checks (grasp alignment, physical lift).
    """

    def __init__(
        self,
        scenario: task.BallBowlScenario,
        controller: robotics.ControllerBase,
        controller_target: robotics.ControllerMochiArticulatedPoseTarget,
        viewer: object | None,
        real_time: bool = False,
        allow_failed_grasp: bool = False,
        frame_callback=None,
        step_callback=None,
    ) -> None:
        self.scenario = scenario
        self.scene = scenario.scene
        self.info = scenario.bot_info
        self.workcell = scenario.workcell
        self.allow_failed_grasp = allow_failed_grasp
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
        self.phases = PhaseLog(self.info.embodiment_id, self._phase_sample)

    # -- state queries ----------------------------------------------------

    def ball_position(self) -> npt.NDArray[np.float64]:
        return np.asarray(self.workcell.ball.get_root_transform().translation, dtype=float)

    def grasp_point_pose(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """World position and (x, y, z, w) quaternion of the embodiment's grasp point."""
        transform = self.ee.get_root_transform() * physics.TransformRT(
            translation=self.info.grasp_point_local
        )
        rotation = transform.rotation
        quaternion = np.array(
            [rotation[0], rotation[1], rotation[2], rotation[3]], dtype=float
        )
        return np.asarray(transform.translation, dtype=float), quaternion

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
            message = (
                "Contact-only grasp failed to lift the ball; "
                f"ball z={ball_position[2]:.3f} m."
            )
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
            "  physical release begins: ball at "
            f"{np.round(self.ball_position(), 3).tolist()}"
        )

    # -- episode control ---------------------------------------------------

    def reset_episode(self, home_pose: npt.ArrayLike) -> None:
        """Restore embodiment and ball state for an exact debugger replay."""
        self.info.actor.set_articulated_pose_from_joints(
            np.asarray(home_pose, dtype=NP_REAL)
        )
        self.info.actor.set_articulated_joint_velocities(
            np.zeros(self.info.actor.get_num_dofs(), dtype=NP_REAL)
        )
        self.workcell.ball.set_root_transform(
            physics.TransformRT(translation=self.scenario.specification.ball_start)
        )
        self.workcell.ball.set_velocity(
            linear_vel=[0.0, 0.0, 0.0], angular_vel=[0.0, 0.0, 0.0]
        )

    def run(self, policy: EpisodePolicy) -> bool:
        completed = policy.run(self)
        self.phases.finish_episode(completed)
        return completed

    def close(self) -> None:
        pass


def create_viewer(
    scenario: task.BallBowlScenario,
    points: npt.NDArray[np.float64],
    *,
    show_trajectory: bool = True,
):
    from superdex.physics.utils.coordinate_systems import CoordinateSystem
    from superdex.physics.viewer import Viewer, ViewerCfg

    viewer = Viewer(
        ViewerCfg(
            coordinate_system=CoordinateSystem(right="-Y", up="+Z", forward="+X"),
            start_paused=False,
        )
    )
    viewer.set_scene(scenario.scene)
    hidden_links = scenario.bot_info.hidden_render_link_names
    if hidden_links:
        viewer.set_excluded_actors(
            [f"*{link_name}" for link_name in sorted(hidden_links)]
        )
    viewer.set_camera_view(
        look_from=[0.95, -1.15, 0.92],
        look_at=[-0.02, 0.0, 0.36],
    )
    if show_trajectory and len(points) > 1:
        edges = np.column_stack([np.arange(len(points) - 1), np.arange(1, len(points))])
        viewer.add_curve_network(
            "collision_aware_trajectory",
            nodes=points,
            edges=edges,
            radius=0.0025,
            color=task.TRAJECTORY_COLOR,
        )
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
        choices=sorted(task.EMBODIMENTS),
        default=None,
        help=f"embodiment to run (default: {task.DEFAULT_EMBODIMENT})",
    )
    parser.add_argument(
        "--human",
        action="store_true",
        help="shorthand for --embodiment human_right_hand (the project scaffold)",
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
        help="run the original fixed blue-ball/gray-bowl regression scenario",
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
    args = parser.parse_args()
    if args.human and args.embodiment not in (None, "human_right_hand"):
        parser.error("--human conflicts with --embodiment")
    args.embodiment = (
        "human_right_hand" if args.human else (args.embodiment or task.DEFAULT_EMBODIMENT)
    )
    return args


def validate_args(args: argparse.Namespace) -> bool:
    """Reject incompatible flag combinations; return whether headless export is automatic."""
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
        policy = make_policy(scenario, options)
        policy.plan()
        return policy

    if args.fixed:
        scenario = task.BallBowlScenario.build(context, args.embodiment)
        print(
            "Scenario: fixed regression configuration; embodiment: "
            f"{scenario.embodiment_id}"
        )
        try:
            return scenario, planner(scenario)
        except Exception:
            scenario.close()
            raise
    assert seed is not None
    print(f"Scenario random seed: {seed}")
    return task.BallBowlScenario.build_randomized(
        context, seed, args.embodiment, planner=planner
    )


def main() -> None:
    args = parse_args()
    automatic_headless_export = validate_args(args)
    selected_seed = (
        None
        if args.fixed
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
    options = PolicyOptions(
        optimize_trajectory=not args.no_trajopt,
        allow_failed_grasp=args.allow_failed_grasp,
    )

    physics.initialize(num_worker_threads=0)
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
                    f"Embodiment {args.embodiment!r} scene built; "
                    f"policy unimplemented: {error}"
                )
                return
            raise
        task_payload = scenario.specification.to_dict()
        cameras = camera_payloads(scenario.bot_info)
        scenario_payload = {"embodiment": scenario.embodiment_id, **task_payload}
        print(json.dumps(scenario_payload, indent=2))

        points = policy.trajectory_points()
        if args.plan_only:
            print(f"Plan valid: {len(points)} displayed trajectory points.")
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
        controller, controller_target = create_pose_controller(scenario.bot_info)
        if (
            not args.headless
            and not args.dry_run
            and not args.debugger
            and args.record_pbr is None
            and export_dir is None
        ):
            viewer = create_viewer(scenario, points)

        if args.record_pbr is not None or export_dir is not None:
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
            real_time=args.debugger or viewer is not None,
            allow_failed_grasp=args.allow_failed_grasp,
            frame_callback=recorder.capture if recorder is not None else None,
            step_callback=telemetry.record_sample if telemetry is not None else None,
        )
        home_pose = policy.home_pose()
        # Every embodiment starts parked exactly where its policy says home is.
        runner.reset_episode(home_pose)
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
            in_bowl = scenario.specification.contains_ball(final_ball)
            print(
                f"Episode {episode_number} "
                f"{'complete' if completed else 'stopped'}; ball at "
                f"{np.round(final_ball, 3).tolist()}, in_bowl={in_bowl}, "
                f"parked-side drift={parked_drift:.2e} rad."
            )
            if recorder is not None:
                recorder.capture()
                recording_path = (
                    export_episode_path
                    if export_episode_path is not None
                    else Path(args.record_pbr).expanduser().resolve()
                )
                recorder.save(recording_path)
            if export_dir is not None:
                runner.phases.save(export_dir / "phases.json")
            if (
                (args.headless or export_dir is not None)
                and completed
                and not in_bowl
                and not args.allow_failed_grasp
            ):
                raise RuntimeError(
                    "Headless episode completed without landing in the bowl."
                )
            if export_dir is not None:
                export_ready = completed and (in_bowl or args.allow_failed_grasp)
            if not (
                args.loop and completed and in_bowl and physics.debugger.is_attached()
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


if __name__ == "__main__":
    main()
