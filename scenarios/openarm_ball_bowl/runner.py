"""Deprecated compatibility wrapper for the neutral ball-and-bowl runner."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scenarios.ball_bowl.runner import main

if __name__ == "__main__":
    warnings.warn(
        "scenarios/openarm_ball_bowl is deprecated; use scenarios/ball_bowl",
        DeprecationWarning,
        stacklevel=1,
    )
    main()
