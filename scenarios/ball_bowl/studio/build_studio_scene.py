"""Build public SuperDex Studio scenes for either physical embodiment.

The public Studio does not expose its internal ``.mochi_bot_scene`` editor, so
this builder exports the robot and workcell together as one standard
``.mochi_scene``.  The result is self-contained and can be opened directly by
the released Studio application.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh
from superdex import physics, robotics
from superdex.physics.paths import resolve_asset
from superdex.physics.utils import render_model_registry

SIM_ROOT = Path(__file__).resolve().parents[3]
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

from scenarios.ball_bowl import scenario as task
from superdex_scenarios.embodiments.human_right_arm import (
    build_human_right_arm,
    destroy_human_right_arm,
)


def _safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "asset"


def _material(
    name: str,
    rgba: tuple[int, int, int, int],
    roughness: float,
    metallic: float = 0.0,
) -> trimesh.visual.material.PBRMaterial:
    return trimesh.visual.material.PBRMaterial(
        name=name,
        baseColorFactor=rgba,
        metallicFactor=metallic,
        roughnessFactor=roughness,
    )


def _box(
    extents: tuple[float, float, float],
    center: tuple[float, float, float],
    material: trimesh.visual.material.PBRMaterial,
) -> trimesh.Trimesh:
    transform = trimesh.transformations.translation_matrix(center)
    mesh = trimesh.creation.box(extents=extents, transform=transform)
    mesh.visual = trimesh.visual.TextureVisuals(material=material)
    return mesh


def _export_scene_glb(scene: trimesh.Scene, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(scene.export(file_type="glb"))


def _assert_studio_positive_octant(path: Path) -> None:
    """Ensure a GLB unit box occupies positive XYZ after Studio conversion."""
    scene = trimesh.load(path, force="scene")
    raw_min, raw_max = np.asarray(scene.bounds, dtype=float)
    # Studio maps glTF (x, y, z) to physics (x, -z, y).
    studio_min = np.array([raw_min[0], -raw_max[2], raw_min[1]])
    studio_max = np.array([raw_max[0], -raw_min[2], raw_max[1]])
    if np.any(studio_min < -1.0e-6) or not np.all(studio_max > 0.0):
        raise RuntimeError(
            f"{path.name} has invalid Studio-space bounds: "
            f"min={studio_min}, max={studio_max}"
        )


def _create_desk_models(render_dir: Path) -> tuple[Path, Path]:
    walnut = _material("Walnut", (105, 52, 22, 255), roughness=0.48)
    dark_grain = _material("Walnut grain", (58, 25, 10, 255), roughness=0.54)

    top_scene = trimesh.Scene()
    top_scene.add_geometry(
        _box((1.0, 1.0, 1.0), (0.5, 0.5, -0.5), walnut),
        node_name="desk_top",
    )
    # Subtle raised strips catch Studio's image-based lighting and read as wood
    # grain without requiring an external texture image.
    # Studio maps glTF (x, y, z) to physics (x, -z, y). Raw Z must therefore
    # span [-1, 0] so the rendered box shares the collision box's [0, 1] local
    # Y range. Raw +Y is the tabletop's local +Z/top face.
    for index, scene_y in enumerate(np.linspace(0.08, 0.92, 11)):
        top_scene.add_geometry(
            _box(
                (0.94, 0.006, 0.004),
                (0.5, 1.002, -float(scene_y)),
                dark_grain,
            ),
            node_name=f"grain_{index:02d}",
        )
    top_path = render_dir / "wood_desk_top.glb"
    _export_scene_glb(top_scene, top_path)
    _assert_studio_positive_octant(top_path)

    leg_scene = trimesh.Scene()
    leg_scene.add_geometry(
        _box((1.0, 1.0, 1.0), (0.5, 0.5, -0.5), walnut),
        node_name="desk_leg",
    )
    leg_path = render_dir / "wood_desk_leg.glb"
    _export_scene_glb(leg_scene, leg_path)
    _assert_studio_positive_octant(leg_path)
    return top_path, leg_path


def _create_ball_model(render_dir: Path) -> Path:
    ball = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    ball.visual = trimesh.visual.TextureVisuals(
        material=_material("Blue tennis ball", (22, 65, 225, 255), roughness=0.72)
    )
    scene = trimesh.Scene(ball)
    path = render_dir / "blue_tennis_ball.glb"
    _export_scene_glb(scene, path)
    return path


def _create_bowl_model(render_dir: Path) -> Path:
    source = Path(resolve_asset("prefabs/paper_cups/render/paper_cup.glb"))
    scene = trimesh.load(source, force="scene")
    ceramic = _material("Gray ceramic", (112, 118, 126, 255), roughness=0.30)
    for geometry in scene.geometry.values():
        geometry.visual = trimesh.visual.TextureVisuals(material=ceramic)
    path = render_dir / "gray_bowl.glb"
    _export_scene_glb(scene, path)
    return path


def _relative_render_path(path: Path, scene_dir: Path) -> str:
    return "./" + path.relative_to(scene_dir).as_posix()


def _attach_robot_render_models(
    prefab: physics.prefab.ScenePrefab,
    bot_prefab: robotics.BotPrefab,
    render_dir: Path,
    scene_dir: Path,
) -> None:
    source_links = {link.name: link for link in bot_prefab.links}
    copied: dict[Path, Path] = {}
    for articulated in prefab.actors.articulated:
        for index, link in enumerate(articulated.links):
            source_link = source_links.get(link.name)
            if source_link is None or not source_link.render_model_file:
                continue
            source = Path(source_link.render_model_file)
            target = (
                render_dir
                / "robot"
                / (f"{index:02d}_{_safe_filename(link.name)}{source.suffix.lower()}")
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            if source not in copied:
                shutil.copy2(source, target)
                copied[source] = target
            else:
                target = copied[source]
            link.render_model_file = _relative_render_path(target, scene_dir)
            link.render_model_scale = source_link.render_model_scale
            link.render_model_rotation = source_link.render_model_rotation
            link.render_model_translation = source_link.render_model_translation


def _attach_workcell_render_models(
    prefab: physics.prefab.ScenePrefab,
    scene_dir: Path,
) -> None:
    render_dir = scene_dir / "render"
    desk_top, desk_leg = _create_desk_models(render_dir)
    ball = _create_ball_model(render_dir)
    bowl = _create_bowl_model(render_dir)

    for actor in prefab.actors.rigid:
        if actor.name == "blue_ball":
            actor.render_model_file = _relative_render_path(ball, scene_dir)
            actor.render_model_scale = [task.BALL_RADIUS] * 3
            actor.density = None
            actor.mass = task.BALL_MASS
            actor.center_of_mass = [0.0, 0.0, 0.0]
            actor.moment_of_inertia = [
                task.BALL_SHELL_INERTIA,
                0.0,
                0.0,
                task.BALL_SHELL_INERTIA,
                0.0,
                task.BALL_SHELL_INERTIA,
            ]
            actor.contact = task.contact_params(task.BALL_FRICTION)
        elif actor.name == "gray_bowl":
            actor.render_model_file = _relative_render_path(bowl, scene_dir)
            actor.render_model_scale = task.BOWL_SCALE
            actor.contact = task.contact_params(task.CERAMIC_FRICTION)
        elif actor.name == "wooden_desk/top":
            actor.render_model_file = _relative_render_path(desk_top, scene_dir)
            actor.render_model_scale = task.DESK_SIZE
            actor.contact = task.contact_params(task.WOOD_FRICTION)
        elif actor.name.startswith("wooden_desk/leg_"):
            actor.render_model_file = _relative_render_path(desk_leg, scene_dir)
            actor.render_model_scale = task.DESK_LEG_SIZE
            actor.contact = task.contact_params(task.WOOD_FRICTION)
        elif actor.name == "ground":
            # Leave the infinite physics plane without a render model. A large
            # floor mesh dominates Studio's automatic scene bounds and makes the
            # initial camera nearly edge-on to the workcell; Studio's own grid is
            # the clearer visual ground reference.
            actor.contact = task.contact_params(0.60)


def build(*, human: bool = False) -> Path:
    """Export, decorate, validate, and install the Studio scene."""
    export_name = "human_ball_bowl_studio" if human else "openarm_ball_bowl_studio"
    output_dir = Path(__file__).resolve().parent / ("human_scene" if human else "scene")
    physics.initialize(num_worker_threads=0)
    display_name = "Human right arm" if human else "OpenArm v2"
    scene = physics.create_scene(f"{display_name}: blue ball into gray bowl")
    scene.set_gravity([0.0, 0.0, -9.80665])
    solver = scene.get_solver_params()
    solver.integration_method = physics.IntegrationMethod.BDF2
    solver.non_linear_solver.max_iter = 8
    scene.set_solver_params(solver)
    context = robotics.create_context()
    bot_info: task.BotInfo | None = None

    try:
        bot_info = (
            build_human_right_arm(scene, context, task.contact_params)
            if human
            else task.create_full_robot(scene, context)
        )
        task.create_workcell(scene)
        ground_shape = physics.create_plane_shape(normal=[0.0, 0.0, 1.0], distance=0.0)
        scene.create_rigid_actor(
            name="ground",
            layer="ground",
            shape=ground_shape,
            is_static=True,
            contact=task.contact_params(0.60),
        )
        physics.release_shape(ground_shape)

        with tempfile.TemporaryDirectory(prefix="openarm_studio_build_") as temp:
            temp_root = Path(temp)
            physics.prefab.export_scene(scene, export_name, str(temp_root))
            staged_dir = temp_root / export_name
            # Marks this self-contained folder as an asset root for Studio. The
            # scene itself uses prefab-relative paths, but the marker also keeps
            # the asset browser from warning about unresolved root references.
            (staged_dir / ".superdex_root").touch()
            scene_path = staged_dir / f"{export_name}.mochi_scene"
            prefab = physics.prefab.shallow_load_from_file(str(scene_path))

            if prefab.scene is not None:
                prefab.scene.description = (
                    f"{display_name} physical ball-and-bowl task; blue regulation-"
                    "size tennis ball, gray bowl, and walnut desk."
                )
            _attach_robot_render_models(
                prefab, bot_info.prefab, staged_dir / "render", staged_dir
            )
            _attach_workcell_render_models(prefab, staged_dir)
            physics.prefab.save_to_json_file(prefab, str(scene_path))

            # Exercise the released loader and instantiate the complete asset before
            # replacing the previous build.
            validation_scene = physics.create_scene("Studio scene validation")
            result = physics.prefab.add_to_scene(
                prefab_path=str(scene_path),
                root_path=str(staged_dir),
                scene=validation_scene,
                params=physics.prefab.PrefabParams(),
            )
            if len(result.actors) != 9:
                raise RuntimeError(
                    "Expected 9 top-level actors in Studio scene, got "
                    f"{len(result.actors)}"
                )

            if output_dir.name not in {"scene", "human_scene"}:
                raise RuntimeError(f"Refusing to replace unexpected path: {output_dir}")
            if output_dir.exists():
                shutil.rmtree(output_dir)
            shutil.copytree(staged_dir, output_dir)

        final_scene = output_dir / f"{export_name}.mochi_scene"
        print(f"Studio scene ready: {final_scene}")
        return final_scene
    finally:
        if bot_info is not None:
            if human:
                destroy_human_right_arm(scene, bot_info)
            else:
                render_model_registry.unregister_actors(
                    scene, bot_info.actor.get_nested_link_actors()
                )
                robotics.destroy_bot(scene, bot_info.bot)
        physics.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human", action="store_true")
    build(human=parser.parse_args().human)
