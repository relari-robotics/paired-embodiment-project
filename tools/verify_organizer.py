"""Run real physics on the fixed, independently shifted, and seeded organizer layouts.

Usage: .venv/bin/python tools/verify_organizer.py --seeds 0 1 2 3 4 --output /tmp/organizer-validation.json
No simulator success is inferred from IK or from a commanded object target.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--task", choices=("organizer", "tea_sorting"), default="organizer"
    )
    args = parser.parse_args()
    cases = [
        ("fixed", ["--fixed"]),
        (
            "shifted",
            ["--layout", str(ROOT / f"scenarios/{args.task}/layouts/shifted.json")],
        ),
    ]
    cases += [(f"seed_{seed}", ["--seed", str(seed)]) for seed in args.seeds]
    results = []
    for name, options in cases:
        start = time.monotonic()
        process = subprocess.run(
            [sys.executable, "-m", f"scenarios.{args.task}", "--dry-run", *options],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
        text = process.stdout
        try:
            result = json.loads(text[text.rfind("\n{\n") + 1 :])
        except (ValueError, json.JSONDecodeError):
            result = {"success": False, "error": process.stderr[-4000:] or text[-4000:]}
        passed = process.returncode == 0 and result.get("success") is True
        results.append(
            {
                "case": name,
                "passed": passed,
                "exit_code": process.returncode,
                "wall_seconds": round(time.monotonic() - start, 2),
                "result": result,
            }
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {"passed": all(r["passed"] for r in results), "cases": results},
                indent=2,
            )
            + "\n"
        )
        print(
            f"{name}: {'PASS' if passed else 'FAIL'} ({results[-1]['wall_seconds']}s) {result.get('error') or ''}",
            flush=True,
        )
    raise SystemExit(0 if all(r["passed"] for r in results) else 1)


if __name__ == "__main__":
    main()
