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

"""Record everything Blender needs to re-render an episode offline.

The recorder walks the physics scene once to describe every actor (which GLB
render model drives a robot link, the physics surface mesh of a rigid body
without one, the surface topology of a soft body) and then, on every captured
frame, stores each actor's root transform and each soft body's deformed
surface node positions in the world frame.  ``scene.json`` carries the static
description and ``frames.npz`` the per-frame arrays; ``render_episode.py``
consumes both inside Blender.

Coordinates are SuperDex world coordinates (metres, +X forward, +Y left, +Z
up), which is also Blender's convention, so no conversion is applied.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from superdex import physics
from superdex.physics.utils import render_model_registry


def quaternion_to_matrix(quaternion: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """3x3 rotation matrix from an (x, y, z, w) quaternion."""
    x, y, z, w = (float(v) for v in np.asarray(quaternion, dtype=float))
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _transform_vector(transform: physics.TransformRT) -> list[float]:
    return [
        *np.asarray(transform.translation, dtype=float).round(7).tolist(),
        *np.asarray(transform.rotation, dtype=float).round(7).tolist(),
    ]


class BlenderSceneRecorder:
    """Capture per-frame transforms and soft-body surfaces for Blender.

    Args:
        scene: The physics scene to record.
        time_step: Physics step [s].
        render_every_steps: Steps between captures (sets the recorded rate).
        cameras: Named fixed look-at presets or calibrated fixed/actor camera
            dictionaries. Actor cameras are resolved against recorded actors by
            their ``actor_suffix`` when rendered.
        materials: ``{actor_name: material_name}`` hints for the renderer.
        colors: ``{actor_name: [r, g, b]}`` base colours for generic materials.
        overrides: ``{actor_name: {...}}`` renderer overrides, for example
            ``{"obj": path, "offset": [x, y, z]}`` to draw a textured scan
            instead of the physics mesh.
        hidden: Actor names (or suffixes) that should not be drawn.
        metadata: Free-form task metadata copied into ``scene.json``.
    """

    FORMAT = "superdex-blender-scene-v1"

    def __init__(
        self,
        scene: physics.Scene,
        *,
        time_step: float,
        render_every_steps: int,
        cameras: dict[str, dict[str, Any]],
        materials: dict[str, str] | None = None,
        colors: dict[str, list[float]] | None = None,
        overrides: dict[str, dict[str, Any]] | None = None,
        hidden: set[str] | frozenset[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.scene = scene
        self.time_step = float(time_step)
        self.render_every_steps = int(render_every_steps)
        self.cameras = cameras
        self.materials = dict(materials or {})
        self.colors = {k: list(v) for k, v in (colors or {}).items()}
        self.overrides = dict(overrides or {})
        self.hidden = set(hidden or ())
        self.metadata = dict(metadata or {})
        self.actors: list[physics.Actor] = []
        self.soft_actors: list[physics.Actor] = []
        self.descriptions: list[dict[str, Any]] = []
        self.static_meshes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.soft_faces: dict[str, np.ndarray] = {}
        self.transform_frames: list[np.ndarray] = []
        self.soft_frames: dict[str, list[np.ndarray]] = {}
        self._describe_scene()

    # -- static description -------------------------------------------------

    def _describe_scene(self) -> None:
        actors: list[physics.Actor] = []
        self.scene.for_each_actor(
            lambda actor: actors.append(actor) if actor.has_root_transform() else None
        )
        actors.sort(key=lambda actor: actor.get_name())
        scene_handle = self.scene.get_handle()
        for index, actor in enumerate(actors):
            name = actor.get_name()
            actor_type = actor.get_type()
            entry: dict[str, Any] = {
                "index": index,
                "name": name,
                "type": str(actor_type).split(".")[-1],
                "hidden": any(name == h or name.endswith(h) for h in self.hidden),
                "material": self.materials.get(name),
                "color": self.colors.get(name),
                "override": self.overrides.get(name),
            }
            model = render_model_registry.get(scene_handle, actor.get_handle())
            if actor_type == physics.ActorType.SOFT:
                actor.register_query_and_compute(physics.QueryType.SURFACE_NODE_POSITIONS)
                surface = actor.get_surface_mesh()
                faces = np.asarray(surface.connectivity, dtype=np.int32).reshape(-1, 3)
                entry["kind"] = "soft"
                entry["soft"] = {"faces": f"soft/{name}/faces", "nodes": f"soft/{name}/nodes"}
                self.soft_faces[name] = faces
                self.soft_frames[name] = []
                self.soft_actors.append(actor)
            elif model is not None:
                entry["kind"] = "glb"
                entry["glb"] = {
                    "path": model.glb_path,
                    "local_transform": _transform_vector(model.local_transform),
                    "scale": np.asarray(model.scale, dtype=float).tolist(),
                }
            elif actor_type == physics.ActorType.ARTICULATED:
                # The articulation root carries no geometry of its own.
                entry["kind"] = "empty"
            else:
                surface = actor.get_surface_mesh()
                vertices = np.asarray(surface.coordinates, dtype=np.float32).reshape(-1, 3)
                faces = np.asarray(surface.connectivity, dtype=np.int32).reshape(-1, 3)
                if len(vertices) == 0 or len(faces) == 0:
                    entry["kind"] = "empty"
                else:
                    entry["kind"] = "mesh"
                    entry["mesh"] = {
                        "vertices": f"mesh/{name}/vertices",
                        "faces": f"mesh/{name}/faces",
                    }
                    self.static_meshes[name] = (vertices, faces)
            self.descriptions.append(entry)
        self.actors = actors

    # -- per-frame capture --------------------------------------------------

    def capture(self) -> None:
        frame = np.empty((len(self.actors), 7), dtype=np.float32)
        for index, actor in enumerate(self.actors):
            transform = actor.get_root_transform()
            frame[index, :3] = np.asarray(transform.translation, dtype=np.float32)
            frame[index, 3:] = np.asarray(transform.rotation, dtype=np.float32)
        self.transform_frames.append(frame)
        for actor in self.soft_actors:
            transform = actor.get_root_transform()
            rotation = quaternion_to_matrix(transform.rotation)
            local = np.asarray(
                actor.get_surface_mesh_node_positions_local(), dtype=np.float64
            ).reshape(-1, 3)
            world = local @ rotation.T + np.asarray(transform.translation, dtype=float)
            self.soft_frames[actor.get_name()].append(world.astype(np.float32))

    @property
    def frame_count(self) -> int:
        return len(self.transform_frames)

    # -- output ---------------------------------------------------------------

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        fps = 1.0 / (self.time_step * self.render_every_steps)
        arrays: dict[str, np.ndarray] = {
            "transforms": np.stack(self.transform_frames)
            if self.transform_frames
            else np.zeros((0, len(self.actors), 7), dtype=np.float32),
            "times": np.arange(self.frame_count, dtype=np.float64) / fps,
        }
        for name, (vertices, faces) in self.static_meshes.items():
            arrays[f"mesh/{name}/vertices"] = vertices
            arrays[f"mesh/{name}/faces"] = faces
        for name, faces in self.soft_faces.items():
            arrays[f"soft/{name}/faces"] = faces
            arrays[f"soft/{name}/nodes"] = (
                np.stack(self.soft_frames[name])
                if self.soft_frames[name]
                else np.zeros((0, 0, 3), dtype=np.float32)
            )
        np.savez_compressed(directory / "frames.npz", **arrays)
        payload = {
            "format": self.FORMAT,
            "coordinate_system": "world +X forward, +Y left, +Z up, metres",
            "fps": fps,
            "time_step": self.time_step,
            "render_every_steps": self.render_every_steps,
            "frame_count": self.frame_count,
            "cameras": self.cameras,
            "actors": self.descriptions,
            "metadata": self.metadata,
            "frames_file": "frames.npz",
        }
        (directory / "scene.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(
            f"Blender scene recorded: {directory} ({self.frame_count} frames at {fps:.1f} FPS, "
            f"{len(self.actors)} actors, {len(self.soft_actors)} soft)"
        )


__all__ = ["BlenderSceneRecorder", "quaternion_to_matrix"]
