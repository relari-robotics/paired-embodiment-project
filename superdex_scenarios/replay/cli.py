"""Runner-side helpers shared by every scenario for trajectory replay.

Each scenario runner keeps its own ``main``; these helpers add the replay
options to its parser, build the replay policy, park the task objects for
object-free runs, and write ``result.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .policy import DEFAULT_SETTLE_S, DEFAULT_TAIL_S, ReplayPolicy
from .trajectory import JointTrajectory

PARKED_OBJECT_POSITION = np.array([1.5, 1.5, 0.0], dtype=float)
"""Where ``--no-objects`` moves task objects: on the ground, far from the desk."""


def add_replay_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("trajectory replay")
    group.add_argument(
        "--replay",
        metavar="FILE",
        help=(
            "replay an open-loop joint trajectory (TRAJECTORY.md) instead of the "
            "scripted reference policy"
        ),
    )
    group.add_argument(
        "--kinematic",
        action="store_true",
        help=(
            "with --replay: set the articulation directly at every target instead "
            "of simulating the controller and contact physics (objects do not move)"
        ),
    )
    group.add_argument(
        "--no-objects",
        action="store_true",
        help=(
            "park the task objects away from the desk so the arm motion can be "
            "checked without any object interaction (success is not evaluated)"
        ),
    )
    group.add_argument(
        "--layout",
        metavar="FILE",
        help=(
            "JSON file overriding object placements (and other sampled values) of "
            "the fixed scenario; see each scenario's README for the keys"
        ),
    )
    group.add_argument(
        "--replay-speed",
        type=float,
        default=1.0,
        help="playback speed factor for --replay (default: 1.0)",
    )
    group.add_argument(
        "--replay-settle",
        type=float,
        default=DEFAULT_SETTLE_S,
        help=f"seconds to hold the first pose before replay (default: {DEFAULT_SETTLE_S})",
    )
    group.add_argument(
        "--replay-tail",
        type=float,
        default=DEFAULT_TAIL_S,
        help=f"seconds to hold the last pose after replay (default: {DEFAULT_TAIL_S})",
    )
    group.add_argument(
        "--result",
        metavar="PATH",
        help="also write the episode result JSON to this path (default: <export-dir>/result.json)",
    )


def validate_replay_arguments(args: argparse.Namespace) -> None:
    if args.kinematic and args.replay is None:
        raise SystemExit("--kinematic requires --replay")
    if args.replay is not None and getattr(args, "loop", False):
        raise SystemExit("--replay cannot be combined with --loop")
    if args.replay_speed <= 0.0:
        raise SystemExit("--replay-speed must be positive")
    if args.layout is not None and getattr(args, "seed", None) is not None:
        raise SystemExit("--layout replaces the fixed scenario; it cannot be combined with --seed")


def load_layout(path: str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Layout file {path} must contain a JSON object.")
    return payload


def make_replay_policy(scenario: Any, options: Any, args: argparse.Namespace) -> ReplayPolicy:
    trajectory = JointTrajectory.load(Path(args.replay).expanduser())
    return ReplayPolicy(
        scenario,
        options,
        trajectory,
        settle_s=args.replay_settle,
        tail_s=args.replay_tail,
        speed=args.replay_speed,
    )


def park_objects(actors: list[Any]) -> None:
    """Move task objects to the parking spot and zero their velocities."""
    from superdex import physics

    for index, actor in enumerate(actors):
        offset = PARKED_OBJECT_POSITION + np.array([0.5 * index, 0.0, 0.0])
        actor.set_root_transform(physics.TransformRT(translation=offset))
        if not actor.is_static():
            actor.set_velocity(linear_vel=[0.0, 0.0, 0.0], angular_vel=[0.0, 0.0, 0.0])


ROUTE_COLORS = ((0.0, 0.85, 0.95), (0.98, 0.55, 0.10), (0.65, 0.35, 0.95))
"""Viewer colours for successive grasp-point routes (right arm first)."""


def split_routes(points: Any) -> list[np.ndarray]:
    """Split an (N, 3) point array into routes at rows that are all NaN."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    routes: list[np.ndarray] = []
    current: list[np.ndarray] = []
    for row in points:
        if np.all(np.isnan(row)):
            if len(current) > 1:
                routes.append(np.asarray(current))
            current = []
        elif np.all(np.isfinite(row)):
            current.append(row)
    if len(current) > 1:
        routes.append(np.asarray(current))
    return routes


def add_route_curves(viewer: Any, points: Any, *, radius: float = 0.0025) -> None:
    """Draw every route in ``points`` as its own curve network in the viewer."""
    for index, route in enumerate(split_routes(points)):
        edges = np.column_stack([np.arange(len(route) - 1), np.arange(1, len(route))])
        viewer.add_curve_network(
            f"trajectory_{index + 1}",
            nodes=route,
            edges=edges,
            radius=radius,
            color=np.asarray(ROUTE_COLORS[index % len(ROUTE_COLORS)]),
        )


def write_result(
    payload: dict[str, Any],
    export_dir: Path | None,
    explicit_path: str | None,
) -> None:
    """Print the result and persist it next to the export (or wherever asked)."""
    text = json.dumps(payload, indent=2, default=_json_default)
    print("Episode result:")
    print(text)
    targets: list[Path] = []
    if export_dir is not None:
        targets.append(export_dir / "result.json")
    if explicit_path is not None:
        targets.append(Path(explicit_path).expanduser().resolve())
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + "\n", encoding="utf-8")
        print(f"Result written: {target}")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialise {type(value).__name__}")


__all__ = [
    "PARKED_OBJECT_POSITION",
    "ROUTE_COLORS",
    "add_replay_arguments",
    "add_route_curves",
    "load_layout",
    "make_replay_policy",
    "park_objects",
    "split_routes",
    "validate_replay_arguments",
    "write_result",
]
