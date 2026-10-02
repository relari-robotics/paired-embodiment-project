"""Reusable planning, simulation, recording, rendering, and embodiments.

The top-level package intentionally performs no simulator imports so offline
tools such as the Rerun exporter remain usable without SuperDex installed.
"""

import os
from pathlib import Path

# SuperDex resolves its assets through SUPERDEX_ASSETS_PATH.  Default it to the
# documented layout (a `project_superdex` checkout next to this package) so a
# fresh install needs no environment configuration; an explicit value wins.
_DEFAULT_ASSETS = Path(__file__).resolve().parents[2] / "project_superdex" / "assets"
if "SUPERDEX_ASSETS_PATH" not in os.environ and _DEFAULT_ASSETS.is_dir():
    os.environ["SUPERDEX_ASSETS_PATH"] = str(_DEFAULT_ASSETS)
del _DEFAULT_ASSETS
