"""Run, inspect, record, and evaluate organizer assembly and sorting."""

# Direct-script execution adds the repository root before local imports.
# ruff: noqa: E402

import argparse
import json
from pathlib import Path
import secrets
import sys

import numpy as np
from superdex import physics, robotics

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scenarios.organizer import scenario as task
from scenarios.organizer.episode import PolicyOptions
from scenarios.organizer.openarm_policy import OpenArmPolicy
from superdex_scenarios.simulation import PoseExecutor, create_pose_controller
from superdex_scenarios.rendering.blender.recorder import BlenderSceneRecorder
from scenarios.ball_bowl.cameras import camera_payloads

CAMERAS = {
    "front": {
        "kind": "fixed",
        "look_from": [0.65, -0.90, 0.95],
        "look_at": [0.00, -0.10, 0.42],
    },
    "top": {
        "kind": "fixed",
        "look_from": [0.02, -0.12, 1.35],
        "look_at": [0.02, -0.12, 0.38],
        "up_world": [1, 0, 0],
    },
}


class EpisodeRunner:
    def __init__(self, scenario, viewer=None, recorder=None):
        self.scenario = scenario
        self.info = scenario.bot_info
        self.events = []
        self.samples = []
        self.record = recorder is not None
        self.phase_name = "initial"
        self.objects = [*scenario.dividers, *scenario.parts]
        self.fingers = [
            self.info.link_actor(scenario.scene, f"/openarm_{side}_ee_link{i}")
            for side in ("right", "left")
            for i in (1, 2)
        ]
        self.queries = [
            (a, a.register_query(physics.QueryType.TOTAL_CONTACT_FORCE))
            for a in self.objects
        ]
        self.ee = self.info.link_actor(
            scenario.scene, f"/{self.info.end_effector_link}"
        )
        controller, target = create_pose_controller(self.info)
        self.executor = PoseExecutor(
            scenario.scene,
            controller,
            target,
            task.TIME_STEP,
            task.RENDER_EVERY_STEPS,
            viewer=viewer,
            frame_callback=recorder.capture if recorder else None,
            step_callback=self.sample,
        )

    def measured_pose(self):
        pose = physics.DynamicArrayReal(self.info.actor.get_num_dofs())
        self.info.actor.get_articulated_pose(pose)
        return np.asarray(pose, dtype=float).copy()

    def select_arm(self, side):
        self.ee = self.info.link_actor(
            self.scenario.scene, f"/openarm_{side}_ee_base_link"
        )

    def grasp_position(self):
        return np.asarray(
            (
                self.ee.get_root_transform()
                * physics.TransformRT(translation=self.info.grasp_point_local)
            ).translation,
            dtype=float,
        )

    def phase(self, name):
        self.phase_name = name
        self.event("phase", name=name)
        print(f"  {self.executor.step_count * task.TIME_STEP:6.2f}s {name}", flush=True)

    def event(self, event, **data):
        self.events.append(
            {
                "time_s": self.executor.step_count * task.TIME_STEP,
                "event": event,
                **data,
            }
        )

    def sample(self, step, target):
        if self.record and step % task.RENDER_EVERY_STEPS == 0:
            objects = [
                a.get_root_transform()
                for a in [*self.scenario.dividers, *self.scenario.parts]
            ]
            self.samples.append(
                (
                    step * task.TIME_STEP,
                    self.measured_pose(),
                    np.array(target),
                    np.array([[*t.translation, *t.rotation] for t in objects]),
                    np.array([a.get_contact_force_world() for a in self.objects]),
                    np.array(
                        [
                            [
                                a.get_contact_force_from_actor_world(f)
                                for f in self.fingers
                            ]
                            for a in self.objects
                        ]
                    ),
                )
            )

    def close(self):
        for actor, query in self.queries:
            actor.cancel_query(query)
        self.queries.clear()

    def follow(self, path, duration, hand, end_hand=None):
        targets = np.asarray([self.info.target_pose(q, hand) for q in path])
        if end_hand is not None:
            start = targets[0, self.info.hand_dofs]
            end = self.info.target_pose(path[-1], end_hand)[self.info.hand_dofs]
            targets[:, self.info.hand_dofs] = np.linspace(start, end, len(path))
        return self.executor.follow(targets, duration)

    def hold(self, q, hand, duration):
        return self.executor.hold(self.info.target_pose(q, hand), duration)


def create_viewer(scenario, camera="front", *, offscreen=False):
    from superdex.physics.utils.coordinate_systems import CoordinateSystem
    from superdex.physics.viewer import Viewer, ViewerCfg

    viewer = Viewer(
        ViewerCfg(
            coordinate_system=CoordinateSystem(right="-Y", up="+Z", forward="+X"),
            start_paused=False,
            size=(1280, 960),
            offscreen=offscreen,
        )
    )
    viewer.set_scene(scenario.scene)
    viewer.set_excluded_actors(
        [f"*{name}" for name in scenario.bot_info.hidden_render_link_names]
    )
    preset = CAMERAS[camera]
    viewer.set_camera_view(look_from=preset["look_from"], look_at=preset["look_at"])
    viewer.render()

    def color_actor(actor):
        color = scenario.colors.get(actor.get_name())
        renderer = viewer.get_actor_renderer(actor)
        if (
            color is not None
            and renderer is not None
            and hasattr(renderer, "set_front_face_color")
        ):
            renderer.set_front_face_color(color)

    scenario.scene.for_each_actor(color_actor)
    viewer.render()
    return viewer


def main(*, task_module=task, policy_class=OpenArmPolicy):
    task = task_module
    parser = argparse.ArgumentParser(description=task.__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--fixed", action="store_true")
    inputs.add_argument("--seed", type=int)
    inputs.add_argument("--layout", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true", help="Headless physics without recording"
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--no-collision-check",
        "--no-trajopt",
        action="store_true",
        help="Diagnostic: skip conservative free-space collision screening",
    )
    parser.add_argument("--export-dir", type=Path)
    parser.add_argument(
        "--record-blender",
        type=Path,
        help="Alias for --export-dir; exports physics and render data",
    )
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--camera", choices=CAMERAS, default="front")
    args = parser.parse_args()
    if args.export_dir and args.record_blender:
        parser.error("Use either --export-dir or --record-blender.")
    if args.layout:
        spec = task.ScenarioSpecification.from_layout(
            json.loads(args.layout.read_text())
        )
    elif args.fixed:
        spec = task.ScenarioSpecification.fixed()
    else:
        spec = task.ScenarioSpecification.randomized(
            args.seed if args.seed is not None else secrets.randbits(32)
        )
    export = args.export_dir or args.record_blender
    if args.headless and not args.dry_run and export is None:
        from datetime import datetime

        export = (
            Path(task.__file__).parent
            / "exports"
            / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        )
    physics.initialize(num_worker_threads=0)
    scenario = viewer = runner = recorder = None
    try:
        scenario = task.OrganizerScenario.build(robotics.create_context(), spec)
        policy = policy_class(
            scenario, PolicyOptions(check_collisions=not args.no_collision_check)
        )
        print(json.dumps(spec.to_dict(), indent=2), flush=True)
        policy.plan()
        if args.plan_only:
            print(
                f"Plan valid: {len(policy.trajectory_points())} Cartesian continuation samples."
            )
            return
        if args.snapshot or not (args.headless or args.dry_run or export):
            viewer = create_viewer(scenario, args.camera, offscreen=bool(args.snapshot))
        if args.snapshot:
            import polyscope as ps

            args.snapshot.parent.mkdir(parents=True, exist_ok=True)
            ps.screenshot(str(args.snapshot), transparent_bg=False, include_UI=False)
            print(f"Snapshot: {args.snapshot}")
            return
        if export:
            export.mkdir(parents=True, exist_ok=True)
            recorder = BlenderSceneRecorder(
                scenario.scene,
                time_step=task.TIME_STEP,
                render_every_steps=task.RENDER_EVERY_STEPS,
                cameras={
                    **CAMERAS,
                    **getattr(task, "CAMERAS", {}),
                    **{c["name"]: c for c in camera_payloads(scenario.bot_info)},
                },
                materials={a.get_name(): "wood" for a in scenario.desk_actors},
                colors=scenario.colors,
                hidden=scenario.bot_info.hidden_render_link_names,
                metadata=spec.to_dict(),
            )
            recorder.capture()
        runner = EpisodeRunner(scenario, viewer, recorder)
        completed, error = False, None
        try:
            completed = policy.run(runner)
        except RuntimeError as exc:
            error = str(exc)
        result = {
            **scenario.outcome(),
            "completed": bool(completed),
            "error": error,
            "seed": spec.seed,
            "simulation_time_s": runner.executor.step_count * task.TIME_STEP,
        }
        result["success"] = bool(completed and result["success"])
        if export:
            recorder.capture()
            recorder.save(export)
            for name, payload in (
                ("scenario", spec.to_dict()),
                ("result", result),
                ("phases", runner.events),
            ):
                (export / f"{name}.json").write_text(
                    json.dumps(payload, indent=2) + "\n"
                )
            if runner.samples:
                times, measured, commanded, objects, forces, finger_forces = zip(
                    *runner.samples
                )
                np.savez_compressed(
                    export / "telemetry.npz",
                    time_s=times,
                    measured_joints=measured,
                    target_joints=commanded,
                    object_poses_xyzw=objects,
                    object_contact_force_world_n=forces,
                    object_from_finger_force_world_n=finger_forces,
                    finger_names=[a.get_name() for a in runner.fingers],
                    joint_names=scenario.bot_info.dof_names,
                    object_names=[
                        a.get_name() for a in [*scenario.dividers, *scenario.parts]
                    ],
                )
        print(json.dumps(result, indent=2))
        if not result["success"]:
            raise SystemExit(1)
    finally:
        if runner is not None:
            runner.close()
        if viewer is not None:
            viewer.close()
        if scenario is not None:
            scenario.close()
        physics.shutdown()


if __name__ == "__main__":
    main()
