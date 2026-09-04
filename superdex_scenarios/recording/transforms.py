"""Self-describing actor-transform replay recording."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from superdex import physics


class TransformRecorder:
    """Record all root transforms with task, embodiment, camera, and render metadata."""

    FORMAT = "superdex-transform-replay-v2"

    def __init__(
        self,
        scene: physics.Scene,
        *,
        time_step: float,
        render_every_steps: int,
        task: dict[str, object],
        embodiment: str,
        cameras: list[dict[str, object]],
        render_manifest: str,
        render_overrides: dict[str, dict[str, object]],
    ) -> None:
        self.scene = scene
        self.time_step = time_step
        self.render_every_steps = render_every_steps
        self.task = task
        self.embodiment = embodiment
        self.cameras = cameras
        self.render_manifest = render_manifest
        self.render_overrides = render_overrides
        self.actors: list[physics.Actor] = []
        scene.for_each_actor(
            lambda actor: (
                self.actors.append(actor) if actor.has_root_transform() else None
            )
        )
        self.actors.sort(key=lambda actor: actor.get_name())
        self.frames: list[list[list[float]]] = []

    def capture(self) -> None:
        frame: list[list[float]] = []
        for actor in self.actors:
            transform = actor.get_root_transform()
            frame.append(
                [
                    *np.round(
                        np.asarray(transform.translation, dtype=float), 7
                    ).tolist(),
                    *np.round(np.asarray(transform.rotation, dtype=float), 7).tolist(),
                ]
            )
        self.frames.append(frame)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fps = 1.0 / (self.time_step * self.render_every_steps)
        payload = {
            "format": self.FORMAT,
            "coordinateSystem": "FLU",
            "fps": fps,
            "embodiment": self.embodiment,
            "scenario": self.task,
            "cameras": self.cameras,
            "renderManifest": self.render_manifest,
            "renderOverrides": self.render_overrides,
            "actors": [actor.get_name() for actor in self.actors],
            "frames": self.frames,
        }
        path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        duration = max(0, len(self.frames) - 1) / fps
        print(
            f"Replay recorded: {path} ({len(self.frames)} frames, "
            f"{duration:.2f} s at {fps:.0f} FPS)"
        )
