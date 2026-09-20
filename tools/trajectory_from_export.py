#!/usr/bin/env python
"""Turn an exported episode bundle into an open-loop trajectory file.

Reads the commanded joint targets (``joint/<name>/target_rad``) from a bundle's
``telemetry.csv`` and its ``phases.json``, and writes a
``openarm-joint-trajectory-v1`` JSON file (see ``TRAJECTORY.md``).  Replaying
that file must reproduce the recorded episode, which is the round-trip check
for the replay pipeline and a worked example of the file format:

    python scenarios/ball_bowl/runner.py --fixed --skip-video --export-dir exports/ref
    python tools/trajectory_from_export.py exports/ref --output exports/ref/trajectory.json
    python scenarios/ball_bowl/runner.py --fixed --dry-run --replay exports/ref/trajectory.json

The script needs only NumPy.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from superdex_scenarios.replay.trajectory import (  # noqa: E402
    ARM_DOF_COUNT,
    JointTrajectory,
    PhaseMark,
)

SIDES = ("right", "left")


def _load_columns(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        header = next(csv.reader(stream))
    values = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64, ndmin=2)
    if values.shape[1] != len(header):
        raise SystemExit(f"{path}: {values.shape[1]} values per row for {len(header)} headers.")
    return header, values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("export_dir", type=Path, help="bundle written by runner.py --export-dir")
    parser.add_argument("--output", type=Path, default=None, help="trajectory JSON (default: <export_dir>/trajectory.json)")
    parser.add_argument("--rate", type=float, default=50.0, help="output sample rate in Hz (default: 50)")
    parser.add_argument(
        "--measured",
        action="store_true",
        help="use measured joint positions instead of the commanded targets",
    )
    args = parser.parse_args()

    export_dir = args.export_dir.expanduser().resolve()
    telemetry_path = export_dir / "telemetry.csv"
    if not telemetry_path.is_file():
        raise SystemExit(f"Missing {telemetry_path}; run runner.py --export-dir first.")
    header, values = _load_columns(telemetry_path)
    column = {name: index for index, name in enumerate(header)}
    time_s = values[:, column["time_s"]]
    suffix = "position_rad" if args.measured else "target_rad"

    stride = max(1, int(round((1.0 / args.rate) / float(np.median(np.diff(time_s))))))
    keep = np.arange(0, len(time_s), stride)
    if keep[-1] != len(time_s) - 1:
        keep = np.append(keep, len(time_s) - 1)

    payload: dict[str, object] = {"format": "openarm-joint-trajectory-v1"}
    arms: dict[str, np.ndarray] = {}
    grippers: dict[str, np.ndarray] = {}
    for side in SIDES:
        arm_columns = [f"joint/arm_openarm_{side}_joint{i}/{suffix}" for i in range(1, ARM_DOF_COUNT + 1)]
        finger_columns = [f"joint/openarm_{side}_finger_joint{i}/{suffix}" for i in (1, 2)]
        if not all(name in column for name in arm_columns):
            continue
        metadata_path = export_dir / "telemetry_metadata.json"
        controlled = None
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            joints = {j["name"]: j for j in metadata.get("joints", [])}
            controlled = all(joints.get(f"arm_openarm_{side}_joint{i}", {}).get("controlled") for i in range(1, ARM_DOF_COUNT + 1))
        if controlled is False:
            continue
        arms[side] = values[np.ix_(keep, [column[c] for c in arm_columns])]
        grippers[side] = values[np.ix_(keep, [column[c] for c in finger_columns])]
    if not arms:
        raise SystemExit("No OpenArm joint columns found in the telemetry.")

    phases: list[PhaseMark] = []
    phases_path = export_dir / "phases.json"
    if phases_path.is_file():
        log = json.loads(phases_path.read_text(encoding="utf-8"))
        for record in log.get("phases", []):
            start = record.get("start") or {}
            if "time_s" in start:
                phases.append(PhaseMark(str(record["name"]), float(start["time_s"])))
    t = time_s[keep]
    if phases:
        phases = [PhaseMark(p.name, float(np.clip(p.time_s, t[0], t[-1]))) for p in phases]

    scenario_path = export_dir / "scenario.json"
    metadata_payload: dict[str, object] = {
        "source": "tools/trajectory_from_export.py",
        "export_dir": str(export_dir),
        "signal": "measured" if args.measured else "commanded",
    }
    if scenario_path.is_file():
        metadata_payload["scenario"] = json.loads(scenario_path.read_text(encoding="utf-8"))
    trajectory = JointTrajectory(t, arms, grippers, phases, metadata_payload, str(telemetry_path))
    output = args.output or export_dir / "trajectory.json"
    trajectory.save(output)
    print(f"Wrote {output}: {json.dumps(trajectory.summary())}")


if __name__ == "__main__":
    main()
