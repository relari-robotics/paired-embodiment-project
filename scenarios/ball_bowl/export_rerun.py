#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "h5py>=3.11",
#   "numpy>=2.0",
#   "rerun-sdk==0.34.0",
# ]
# ///
"""Scenario-local entry point for the reusable Rerun exporter."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from superdex_scenarios.recording.export_rerun import main

if __name__ == "__main__":
    main()
