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

"""Embodiment-neutral simulator and CLI for the sponge-and-plate scenario.

The runner builds the scene for the selected embodiment, asks that
embodiment's :class:`~scenarios.sponge_plate.episode.EpisodePolicy` to plan,
and executes the episode through the shared pipeline: compliant pose
controller, physics stepping, optional viewer, transform/telemetry/phase
recording, the plate cleanliness map, and export.  Success means the sponge
wiped at least ``CLEAN_SUCCESS_COVERAGE`` of the dish floor under pressure and
was put back on the desk.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
from superdex import physics, robotics

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scenarios.ball_bowl.cameras import camera_payloads
from superdex_scenarios.recording import PhaseLog, TelemetryRecorder, TransformRecorder
from superdex_scenarios.simulation import PoseExecutor, create_pose_controller

if __package__:
    from . import scenario as task
    from .episode import TASK_PHASES, EpisodePolicy, PolicyOptions, load_policy_class
else:
    from scenarios.sponge_plate import scenario as task
    from scenarios.sponge_plate.episode import (
        TASK_PHASES,
        EpisodePolicy,
        PolicyOptions,
        load_policy_class,
    )

NP_REAL = np.float64 if physics.uses_double_precision() else np.float32


def make_policy(scenario: task.SpongePlateScenario, options: PolicyOptions) -> EpisodePolicy:
    """Instantiate the registered policy for the scenario's embodiment."""
    policy_class = load_policy_class(scenario.embodiment.policy)
    return policy_class(scenario, options)


class CleanlinessTracker:
    """Feed sponge-to-plate contact samples into the :class:`CleanlinessMap` every step."""

    def __init__(self, scenario: task.SpongePlateScenario) -> None:
        self.scene = scenario.scene
        self.sponge = scenario.workcell.sponge
        self.plate_handle = scenario.workcell.plate.get_handle()
        self.map = task.CleanlinessMap(scenario.specification)
        self._query = self.sponge.register_query(physics.QueryType.CONTACT_POINTS)
        self.plate_normal_force = 0.0
        self.peak_plate_normal_force = 0.0

    def record_sample(self, step: int, _target_pose: npt.ArrayLike) -> None:
        positions, forces, speeds = [], [], []
        for point in self.sponge.get_contact_points_world():
            if point.actor_b != self.plate_handle and point.actor_a != self.plate_handle:
                continue
            force = np.asarray(point.force, dtype=float)
            normal = np.asarray(point.normal, dtype=float)
            normal_force = abs(float(np.dot(force, normal)))
            velocity = np.asarray(point.point_velocity_a, dtype=float)
            tangential = velocity - np.dot(velocity, normal) * normal
            positions.append(np.asarray(point.pos_b, dtype=float))
            forces.append(normal_force)
            speeds.append(float(np.linalg.norm(tangential)))
        self.plate_normal_force = float(np.sum(forces)) if forces else 0.0
        self.peak_plate_normal_force = max(self.peak_plate_normal_force, self.plate_normal_force)
        self.map.update(
            step,
            np.asarray(positions, dtype=float).reshape(-1, 3),
            np.asarray(forces, dtype=float),
            np.asarray(speeds, dtype=float),
        )

    def close(self) -> None:
        if self._query is not None:
            self.sponge.cancel_query(self._query)
            self._query = None


class EpisodeRunner:
    """Shared execution helpers a policy uses to drive one episode."""

    def __init__(
        self,
        scenario: task.SpongePlateScenario,
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
        self.cleanliness = CleanlinessTracker(scenario)

        def on_step(step: int, target_pose: npt.NDArray[np.float64]) -> None:
            self.cleanliness.record_sample(step, target_pose)
            if step_callback is not None:
                step_callback(step, target_pose)

        self.executor = PoseExecutor(
            self.scene,
            controller,
            controller_target,
            task.TIME_STEP,
            task.RENDER_EVERY_STEPS,
            viewer=viewer,
            real_time=real_time,
            frame_callback=frame_callback,
            step_callback=on_step,
        )
        self.ee = self.info.link_actor(self.scene, f"/{self.info.end_effector_link}")
        self.phases = PhaseLog(self.info.embodiment_id, self._phase_sample, TASK_PHASES)

    # -- state queries ----------------------------------------------------

    def sponge_position(self) -> npt.NDArray[np.float64]:
        return task.sponge_position(self.workcell.sponge)

    def sponge_bounds(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        bounds = self.workcell.sponge.get_aabb_world()
        return np.asarray(bounds.min, dtype=float), np.asarray(bounds.max, dtype=float)

    def grasp_point_pose(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
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
        return step, step * task.TIME_STEP, position, quaternion, self.sponge_position()

    @property
    def coverage(self) -> float:
        return self.cleanliness.map.coverage

    # -- phase bookkeeping -------------------------------------------------

    def phase(self, name: str) -> None:
        self.phases.begin(name)
        if os.environ.get("SUPERDEX_PHASE_DEBUG"):
            sample = self.phases.records[-1].start
            print(
                f"  phase {name} @ {sample.time_s:.2f}s: grasp point "
                f"{np.round(sample.grasp_point_world_m, 3).tolist()}, sponge "
                f"{np.round(sample.object_position_m, 3).tolist()}, "
                f"coverage {self.coverage:.2f}"
            )

    # -- motion helpers ----------------------------------------------------

    def _target_pose(self, arm_pose: npt.ArrayLike, hand: str | npt.ArrayLike):
        return self.info.target_pose(arm_pose, hand)

    def follow(
        self,
        path: npt.NDArray[np.float64],
        duration: float,
        hand_start: str | npt.ArrayLike,
        hand_end: str | npt.ArrayLike | None = None,
    ) -> bool:
        start = (
            self.info.hand_poses[hand_start]
            if isinstance(hand_start, str)
            else np.asarray(hand_start, dtype=float)
        )
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

    def hold(self, arm_pose: npt.ArrayLike, hand: str | npt.ArrayLike, duration: float) -> bool:
        return self.executor.hold(self._target_pose(arm_pose, hand), duration)

    def hold_pose(self, pose: npt.ArrayLike, duration: float) -> bool:
        return self.executor.hold(pose, duration)

    # -- task checks -------------------------------------------------------

    def print_tracking_error(self, planned_arm: npt.ArrayLike) -> None:
        actual_arm = self.measured_pose()[self.info.arm_dofs]
        print(
            "  arm tracking error: "
            f"{np.round(actual_arm - np.asarray(planned_arm, dtype=float), 3).tolist()}"
        )

    def check_grasp_alignment(self) -> None:
        """Refuse to close when the grasp point is not on the sponge's upper half."""
        target = self.sponge_position() + np.array([0.0, 0.0, task.GRASP_UP])
        grasp_position, _ = self.grasp_point_pose()
        print(
            "  grasp: sponge target/grasp point = "
            f"{np.round(target, 3).tolist()} / {np.round(grasp_position, 3).tolist()}"
        )
        grasp_error = float(np.linalg.norm(target - grasp_position))
        if self.info.grasp_tolerance_m is None:
            raise ValueError("The embodiment must define grasp_tolerance_m.")
        if grasp_error > self.info.grasp_tolerance_m:
            raise RuntimeError(
                f"Grasp point missed the sponge by {grasp_error:.3f} m; refusing to close."
            )

    def verify_physical_grasp(self) -> None:
        """Prove that contact alone lifted the sponge off the desk."""
        low, high = self.sponge_bounds()
        minimum_lifted_z = task.DESK_TOP_Z + 0.05
        if low[2] < minimum_lifted_z:
            message = (
                "Contact-only grasp failed to lift the sponge; "
                f"sponge bottom z={low[2]:.3f} m."
            )
            if not self.allow_failed_grasp:
                raise RuntimeError(message)
            print(f"  WARNING: {message} Continuing diagnostic episode.")
            return
        print(
            f"  physical grasp verified: sponge at {np.round(self.sponge_position(), 3).tolist()}, "
            f"height {1000 * (high[2] - low[2]):.0f} mm"
        )
        self.phases.event("grasp_verified")

    def verify_press(self) -> None:
        """Report the normal force the sponge exerts on the plate after lowering."""
        force = self.cleanliness.plate_normal_force
        low, _ = self.sponge_bounds()
        print(
            f"  press: sponge-on-plate normal force {force:.2f} N, sponge bottom "
            f"{1000 * (low[2] - self.scenario.specification.dish_floor_z):+.1f} mm from dish floor"
        )
        if force < 0.5:
            message = f"Sponge is not pressed onto the plate ({force:.2f} N)."
            if not self.allow_failed_grasp:
                raise RuntimeError(message)
            print(f"  WARNING: {message} Continuing diagnostic episode.")
            return
        self.phases.event("press_verified", normal_force_n=force)

    def print_wipe_progress(self, stroke: int) -> None:
        print(
            f"  wipe stroke {stroke}: coverage {100 * self.coverage:.0f}%, plate normal "
            f"force {self.cleanliness.plate_normal_force:.2f} N, sponge at "
            f"{np.round(self.sponge_position(), 3).tolist()}"
        )
        self.phases.event("wipe_stroke", stroke=stroke, coverage=self.coverage)

    def print_release_position(self) -> None:
        print(
            "  physical release begins: sponge at "
            f"{np.round(self.sponge_position(), 3).tolist()}"
        )

    # -- episode control ---------------------------------------------------

    def reset_episode(self, home_pose: npt.ArrayLike) -> None:
        """Restore embodiment and sponge state for an exact replay."""
        self.info.actor.set_articulated_pose_from_joints(
            np.asarray(home_pose, dtype=NP_REAL)
        )
        self.info.actor.set_articulated_joint_velocities(
            np.zeros(self.info.actor.get_num_dofs(), dtype=NP_REAL)
        )
        sponge = self.workcell.sponge
        nodes, _ = task.sponge_tet_mesh()
        sponge.set_root_transform(
            physics.TransformRT(translation=self.scenario.specification.sponge_start)
        )
        sponge.set_node_positions_local(np.asarray(nodes, dtype=NP_REAL).ravel())
        sponge.set_node_velocities_local(np.zeros(nodes.size, dtype=NP_REAL))
        self.cleanliness.map = task.CleanlinessMap(self.scenario.specification)

    def run(self, policy: EpisodePolicy) -> bool:
        completed = policy.run(self)
        self.phases.finish_episode(completed)
        return completed

    def close(self) -> None:
        self.cleanliness.close()


DEFAULT_LOOK_FROM = (0.42, -0.62, 0.66)
DEFAULT_LOOK_AT = (-0.04, -0.06, 0.40)
CAMERA_PRESETS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "workcell": (DEFAULT_LOOK_FROM, DEFAULT_LOOK_AT),
    "plate": ((0.22, -0.34, 0.52), (-0.05, 0.0, 0.40)),
    "sponge": ((0.28, -0.42, 0.50), (0.0, -0.20, 0.41)),
    "side": ((-0.05, -0.55, 0.47), (-0.05, 0.0, 0.41)),
}


def create_viewer(
    scenario: task.SpongePlateScenario,
    points: npt.NDArray[np.float64],
    *,
    show_trajectory: bool = True,
    look_from: tuple[float, float, float] = DEFAULT_LOOK_FROM,
    look_at: tuple[float, float, float] = DEFAULT_LOOK_AT,
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
        viewer.set_excluded_actors([f"*{name}" for name in sorted(hidden_links)])
    viewer.set_camera_view(look_from=list(look_from), look_at=list(look_at))
    if show_trajectory and len(points) > 1:
        edges = np.column_stack([np.arange(len(points) - 1), np.arange(1, len(points))])
        viewer.add_curve_network(
            "planned_trajectory",
            nodes=points,
            edges=edges,
            radius=0.0025,
            color=task.TRAJECTORY_COLOR,
        )
    viewer.render()
    for actor, color in (
        *((actor, task.WOOD_COLOR) for actor in scenario.workcell.desk_actors),
        (scenario.workcell.plate, task.PLATE_COLOR),
        (scenario.workcell.sponge, scenario.specification.sponge_color.rgb),
    ):
        renderer = viewer.get_actor_renderer(actor)
        if renderer is not None and hasattr(renderer, "set_front_face_color"):
            renderer.set_front_face_color(color)
    return viewer


class FrameSaver:
    """Save numbered viewer screenshots while an interactive episode plays."""

    def __init__(self, directory: Path, every: int) -> None:
        self.directory = directory
        self.every = max(1, every)
        self.count = 0
        self.saved = 0
        directory.mkdir(parents=True, exist_ok=True)

    def __call__(self) -> None:
        self.count += 1
        if self.count % self.every:
            return
        import polyscope as ps

        ps.screenshot(
            str(self.directory / f"frame_{self.saved:05d}.png"),
            transparent_bg=False,
            include_UI=False,
        )
        self.saved += 1


def default_headless_export_dir(seed: int | None) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    task_id = "fixed" if seed is None else f"seed_{seed}"
    return Path(__file__).resolve().parent / "exports" / "runs" / f"{timestamp}_{task_id}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embodiment",
        choices=sorted(task.EMBODIMENTS),
        default=task.DEFAULT_EMBODIMENT,
        help=f"embodiment to run (default: {task.DEFAULT_EMBODIMENT})",
    )
    randomization = parser.add_mutually_exclusive_group()
    randomization.add_argument(
        "--seed",
        type=lambda value: int(value, 0),
        help="reproduce a randomized scenario with this seed; if omitted, a new seed is generated",
    )
    randomization.add_argument(
        "--fixed", action="store_true", help="run the deterministic regression scenario"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="run without a viewer and write a bundle to a unique exports/runs directory",
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
        help="use direct Cartesian references instead of trajectory optimization",
    )
    parser.add_argument(
        "--allow-failed-grasp",
        action="store_true",
        help="complete and export the entire motion even if a physical check fails",
    )
    parser.add_argument(
        "--snapshot",
        nargs="?",
        const=str(Path(__file__).resolve().parent / "exports" / "snapshot.png"),
        metavar="PATH",
        help="render the planned pre-grasp pose and save one PNG",
    )
    parser.add_argument(
        "--frames",
        metavar="DIR",
        help="with the interactive viewer, also save a screenshot every --frame-every rendered frames",
    )
    parser.add_argument(
        "--frame-every", type=int, default=10, help="viewer frames between saved screenshots"
    )
    parser.add_argument(
        "--camera",
        choices=sorted(CAMERA_PRESETS),
        default="workcell",
        help="viewer camera preset (default: workcell; 'plate' and 'side' zoom on the wipe)",
    )
    parser.add_argument(
        "--no-trajectory",
        action="store_true",
        help="hide the planned grasp-point trajectory curve in the viewer",
    )
    parser.add_argument(
        "--debugger",
        action="store_true",
        help="open the live Mochi debugger and execute the episode at wall-clock speed",
    )
    parser.add_argument(
        "--loop", action="store_true", help="replay successful episodes while the debugger stays attached"
    )
    parser.add_argument(
        "--record-pbr",
        nargs="?",
        const=str(Path(__file__).resolve().parent / "exports" / "replay.json"),
        metavar="PATH",
        help="run one headless episode and record rigid actor transforms",
    )
    parser.add_argument(
        "--export-dir",
        nargs="?",
        const=str(Path(__file__).resolve().parent / "exports" / "latest"),
        metavar="PATH",
        help=(
            "run one headless episode and export telemetry, contacts, transforms, "
            "the phase log, and the plate cleanliness map (default: exports/latest)"
        ),
    )
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="accepted for parity with the ball-and-bowl runner; this scenario never renders MP4s",
    )
    args = parser.parse_args()
    return args


def validate_args(args: argparse.Namespace) -> bool:
    if args.loop and not args.debugger:
        raise SystemExit("--loop requires --debugger")
    if args.headless and (args.debugger or args.loop):
        raise SystemExit("--headless cannot be combined with --debugger or --loop")
    exclusive = (
        args.debugger or args.loop or args.record_pbr is not None or args.export_dir is not None
    )
    if args.dry_run and (exclusive or args.skip_video):
        raise SystemExit("--dry-run cannot be combined with simulation or export options")
    if args.plan_only and (exclusive or args.skip_video):
        raise SystemExit("--plan-only cannot be combined with simulation or export options")
    if args.snapshot is not None and (
        args.headless or args.dry_run or args.plan_only or exclusive or args.skip_video
    ):
        raise SystemExit("--snapshot cannot be combined with simulation or export options")
    if args.export_dir is not None and (args.debugger or args.loop or args.record_pbr is not None):
        raise SystemExit("--export-dir cannot be combined with --debugger, --loop, or --record-pbr")
    if args.frames is not None and (
        args.headless or args.dry_run or args.plan_only or args.debugger or exclusive
    ):
        raise SystemExit("--frames requires the interactive viewer (no headless/export options)")
    return (
        args.headless
        and not args.dry_run
        and not args.plan_only
        and not args.debugger
        and args.record_pbr is None
        and args.export_dir is None
    )


def build_and_plan(
    context: robotics.RoboticsContext,
    args: argparse.Namespace,
    seed: int | None,
    options: PolicyOptions,
) -> tuple[task.SpongePlateScenario, EpisodePolicy]:
    def planner(scenario: task.SpongePlateScenario) -> EpisodePolicy:
        policy = make_policy(scenario, options)
        policy.plan()
        return policy

    if args.fixed:
        scenario = task.SpongePlateScenario.build(context, args.embodiment)
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
    return task.SpongePlateScenario.build_randomized(
        context, seed, args.embodiment, planner=planner
    )


def main() -> None:
    args = parse_args()
    automatic_headless_export = validate_args(args)
    selected_seed = (
        None if args.fixed else (args.seed if args.seed is not None else secrets.randbits(63))
    )
    if args.export_dir is not None:
        export_dir = Path(args.export_dir).expanduser().resolve()
    elif automatic_headless_export:
        export_dir = default_headless_export_dir(selected_seed).resolve()
        print(f"Headless export directory: {export_dir}")
    else:
        export_dir = None
    export_episode_path = export_dir / "episode.json" if export_dir else None
    options = PolicyOptions(
        optimize_trajectory=not args.no_trajopt,
        allow_failed_grasp=args.allow_failed_grasp,
    )

    physics.initialize(num_worker_threads=0)
    scenario: task.SpongePlateScenario | None = None
    viewer = None
    runner: EpisodeRunner | None = None
    recorder: TransformRecorder | None = None
    telemetry: TelemetryRecorder | None = None
    try:
        if args.debugger:
            debug_server = physics.get_debug_server()
            if not debug_server.has_started():
                debug_server.set_coordinate_space(
                    physics.CoordinateSpace(
                        axes=physics.CoordinateSpaceAxes.FLU, units_per_meter=1.0
                    )
                )
            if not physics.debugger.attach(timeout_seconds=10.0):
                raise RuntimeError("Could not attach the Mochi physics debugger.")
        context = robotics.create_context()
        try:
            scenario, policy = build_and_plan(context, args, selected_seed, options)
        except NotImplementedError as error:
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
            viewer.set_camera_view(look_from=[0.48, -0.76, 0.68], look_at=[0.0, -0.16, 0.43])
            viewer.render()
            import polyscope as ps

            snapshot_path = Path(args.snapshot).expanduser().resolve()
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            ps.screenshot(str(snapshot_path), transparent_bg=False, include_UI=False)
            print(f"Simulator snapshot: {snapshot_path}")
            return

        controller, controller_target = create_pose_controller(scenario.bot_info)
        interactive = (
            not args.headless
            and not args.dry_run
            and not args.debugger
            and args.record_pbr is None
            and export_dir is None
        )
        if interactive:
            look_from, look_at = CAMERA_PRESETS[args.camera]
            viewer = create_viewer(
                scenario,
                points,
                show_trajectory=not args.no_trajectory,
                look_from=look_from,
                look_at=look_at,
            )

        frame_callbacks = []
        if args.record_pbr is not None or export_dir is not None:
            recorder = TransformRecorder(
                scenario.scene,
                time_step=task.TIME_STEP,
                render_every_steps=task.RENDER_EVERY_STEPS,
                task=task_payload,
                embodiment=scenario.embodiment_id,
                cameras=cameras,
                render_manifest="",
                render_overrides={
                    "soft_sponge": {
                        "scale": task.SPONGE_SIZE.tolist(),
                        "color": list(scenario.specification.sponge_color.rgb),
                    },
                    "ceramic_plate": {"color": task.PLATE_COLOR.tolist()},
                },
            )
            recorder.capture()
            frame_callbacks.append(recorder.capture)
        if args.frames is not None:
            frame_callbacks.append(FrameSaver(Path(args.frames).expanduser(), args.frame_every))
        if export_dir is not None:
            export_dir.mkdir(parents=True, exist_ok=True)
            (export_dir / "scenario.json").write_text(
                json.dumps(scenario_payload, indent=2) + "\n", encoding="utf-8"
            )
            telemetry = TelemetryRecorder(
                scenario, export_dir, time_step=task.TIME_STEP, cameras=cameras
            )

        def frame_callback() -> None:
            for callback in frame_callbacks:
                callback()

        runner = EpisodeRunner(
            scenario,
            controller,
            controller_target,
            viewer,
            real_time=args.debugger or viewer is not None,
            allow_failed_grasp=args.allow_failed_grasp,
            frame_callback=frame_callback if frame_callbacks else None,
            step_callback=telemetry.record_sample if telemetry is not None else None,
        )
        home_pose = policy.home_pose()
        runner.reset_episode(home_pose)
        if args.debugger:
            runner.hold_pose(home_pose, 1.5)
        uncontrolled = np.ones(scenario.bot_info.actor.get_num_dofs(), dtype=bool)
        uncontrolled[scenario.bot_info.controlled_dofs] = False
        episode_number = 0
        while True:
            episode_number += 1
            completed = runner.run(policy)
            if args.debugger and completed:
                runner.hold_pose(home_pose, 1.0)
            final_sponge = runner.sponge_position()
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
            coverage = runner.coverage
            returned = scenario.specification.sponge_returned(final_sponge)
            success = coverage >= task.CLEAN_SUCCESS_COVERAGE and returned
            print(
                f"Episode {episode_number} {'complete' if completed else 'stopped'}; "
                f"plate coverage {100 * coverage:.0f}% (need {100 * task.CLEAN_SUCCESS_COVERAGE:.0f}%), "
                f"peak plate normal force {runner.cleanliness.peak_plate_normal_force:.2f} N, "
                f"sponge at {np.round(final_sponge, 3).tolist()}, returned={returned}, "
                f"clean={success}, parked-side drift={parked_drift:.2e} rad."
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
                (export_dir / "cleanliness.json").write_text(
                    json.dumps(runner.cleanliness.map.to_dict(), separators=(",", ":")) + "\n",
                    encoding="utf-8",
                )
            if (
                (args.headless or export_dir is not None)
                and completed
                and not success
                and not args.allow_failed_grasp
            ):
                raise RuntimeError("Headless episode completed without cleaning the plate.")
            if not (args.loop and completed and success and physics.debugger.is_attached()):
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


if __name__ == "__main__":
    main()
